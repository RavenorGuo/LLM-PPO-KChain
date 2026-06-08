"""
多Agent博弈系统 — 基于LangGraph的改进合同网协议

参考:
- 论文4 (集群杀伤链): 改进合同网协议 (招标-投标-反提案-中标)
- 论文 (即时杀伤链): 派单-甩单-接单 资源调度流程
- 论文3 (分布式杀伤链): 方程(10) 多目标优化, Nash均衡

LangGraph 图结构:
  [协调者发布任务] → [4类Agent并行投标] → [协调者评分]
       ↕                                          ↓
  [反提案/调整] ← [不满足] ← [Nash均衡检查] → [满足] → [生成奖励参数]

5类Agent:
  SensorAgent: 优化探测覆盖率 & 信息质量
  WeaponAgent: 优化杀伤概率 & 弹药效率
  CommandAgent: 优化决策速度 & 指挥容量
  ResourceAgent: 优化全局资源利用率
  Coordinator: 仲裁 & 全局Pareto最优
"""
import numpy as np
from dataclasses import dataclass, field
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from model import (
    KillChainNetwork, Platform, Target, RequirementVector, RequirementParser,
    compute_node_info_sum, compute_info_measure, evaluate_kill_chain_effectiveness,
)
from deep_agents import LLMClient, DeepBidGenerator, DeepCoordinator, DeepRewardGenerator
import copy
import json


# ============================================================
# 1. LangGraph 状态定义
# ============================================================

class GameState(TypedDict):
    """多Agent博弈共享状态"""
    # 输入
    requirement: str                          # 用户原始需求
    req_vector: dict                          # RequirementVector 序列化
    network_nodes: int                        # 网络节点数
    n_targets: int                            # 目标数

    # 博弈状态
    round: int                                # 当前轮次
    max_rounds: int                           # 最大轮数
    converged: bool                           # 是否收敛

    # Agent投标
    bids: dict                                # {agent_type: bid_dict}
    all_bids_raw: list                        # 所有轮次的原始投标 (手动管理)

    # 协调者评估
    scores: dict                              # {agent_type: score}
    rankings: list                            # 排序后的agent列表

    # 谈判
    counter_proposals: dict                   # 反提案
    disagreement_points: dict                 # 各Agent的底线

    # 均衡结果
    nash_equilibrium: Optional[dict]          # Nash均衡方案
    pareto_front: list                        # Pareto前沿点

    # 输出
    reward_params: Optional[dict]             # PPO奖励参数
    messages: list                            # 日志消息 (手动管理，避免operator.add指数爆炸)
    done: bool                                # 完成标记
    llm_config: Optional[dict]                # DeepAgents LLM配置


# ============================================================
# 2. 基础Agent & 4类作战Agent
# ============================================================

@dataclass
class AgentConfig:
    """Agent配置"""
    agent_type: str
    reservation_price: float  # 保留效用 (disagreement point)
    negotiation_power: float  # 谈判力量 α_i
    risk_aversion: float      # 风险厌恶系数


class BaseAgent:
    """作战Agent基类"""
    def __init__(self, config: AgentConfig, platforms: list, targets: list,
                 network: KillChainNetwork):
        self.config = config
        self.platforms = platforms
        self.targets = targets
        self.network = network
        self.bid_history = []
        self.current_utility = config.reservation_price

    def compute_utility(self, assignment: np.ndarray, req_vec: RequirementVector) -> float:
        """计算个体效用函数 — 子类重写"""
        raise NotImplementedError

    def compute_bid(self, task_spec: dict, req_vec: RequirementVector) -> dict:
        """计算投标: 返回 {capability, cost, expected_utility}"""
        raise NotImplementedError

    def propose_counter(self, current_allocation: dict, req_vec: RequirementVector) -> dict:
        """生成反提案"""
        raise NotImplementedError


class SensorAgent(BaseAgent):
    """传感器Agent: 最大化探测覆盖率 & 信息质量 y_sensor

    效用: U_s = w1 * coverage + w2 * info_quality - w3 * resource_usage
    """
    def compute_utility(self, assignment: np.ndarray, req_vec: RequirementVector) -> float:
        s_idx = int(assignment[0]) % len(self.platforms)
        # 探测覆盖率: 传感器对目标位置的可见性
        sensor_plat = self.platforms[s_idx]
        coverage = 0.0
        for t in self.targets:
            dist = np.linalg.norm(sensor_plat.position - t.position)
            coverage += max(0, 1.0 - dist / sensor_plat.max_range)
        coverage /= max(1, len(self.targets))
        # 信息质量: y ∈ [0,1]
        info_qual = compute_node_info_sum(self.network, s_idx)
        # 资源消耗
        resource_use = 1.0 / max(1, sensor_plat.ammo)
        return 0.4 * coverage + 0.4 * info_qual - 0.2 * resource_use

    def compute_bid(self, task_spec: dict, req_vec: RequirementVector) -> dict:
        relevant = [p for p in self.platforms if p.platform_type == "sensor"]
        if not relevant:
            relevant = self.platforms[:5]
        best = max(relevant, key=lambda p: p.kill_prob)
        utility = compute_node_info_sum(self.network, best.platform_id)
        return {
            "agent_type": "sensor",
            "platform_id": int(best.platform_id),
            "capability": float(best.kill_prob),
            "info_quality": float(utility),
            "cost": float(best.cost_per_shot),
            "expected_utility": float(utility),
            "available": bool(best.is_available and best.ammo > 0),
        }

    def propose_counter(self, current_allocation: dict, req_vec: RequirementVector) -> dict:
        """如果不满意当前分配，提议更换传感器"""
        return {"action": "swap_sensor", "reason": "improve_coverage",
                "preferred": current_allocation.get("sensor", 0)}


class WeaponAgent(BaseAgent):
    """武器Agent: 最大化杀伤概率 & 最小化弹药消耗

    效用: U_w = w1 * P_kill * target_value - w2 * cost_per_shot - w3 * reload_penalty
    """
    def compute_utility(self, assignment: np.ndarray, req_vec: RequirementVector) -> float:
        w_idx = int(assignment[1]) % len(self.platforms)
        weapon = self.platforms[w_idx]
        # 杀伤效能
        kill_score = weapon.kill_prob * np.mean([t.value for t in self.targets])
        # 成本归一化
        cost_norm = weapon.cost_per_shot / 200.0
        # 时间惩罚
        time_penalty = weapon.reload_time / 60.0
        return kill_score - 0.3 * cost_norm - 0.2 * time_penalty

    def compute_bid(self, task_spec: dict, req_vec: RequirementVector) -> dict:
        relevant = [p for p in self.platforms if p.platform_type == "weapon" and p.is_available]
        if not relevant:
            relevant = [p for p in self.platforms if p.is_available][:5] or self.platforms[:5]
        best = max(relevant, key=lambda p: p.kill_prob / max(p.cost_per_shot, 1.0))
        return {
            "agent_type": "weapon",
            "platform_id": int(best.platform_id),
            "capability": float(best.kill_prob),
            "cost_per_shot": float(best.cost_per_shot),
            "range": float(best.max_range),
            "ammo": int(best.ammo),
            "expected_utility": float(best.kill_prob * task_spec.get("target_value", 0.7)),
            "available": bool(best.is_available and best.ammo > 0),
        }

    def propose_counter(self, current_allocation: dict, req_vec: RequirementVector) -> dict:
        return {"action": "swap_weapon", "reason": "improve_kill_prob",
                "preferred": current_allocation.get("weapon", 0)}


class CommandAgent(BaseAgent):
    """指控Agent: 最小化决策延迟 & 最大化指挥容量

    效用: U_c = w1 * cmd_coverage - w2 * decision_delay - w3 * overload_penalty
    """
    def compute_utility(self, assignment: np.ndarray, req_vec: RequirementVector) -> float:
        c_idx = int(assignment[2]) % len(self.platforms)
        cmd = self.platforms[c_idx]
        cmd_quality = compute_info_measure(self.network, c_idx, "command_control")
        # 延迟: 基于网络连通质量
        delay = 1.0 - float(self.network.C[c_idx].mean())
        return 0.6 * cmd_quality - 0.4 * delay

    def compute_bid(self, task_spec: dict, req_vec: RequirementVector) -> dict:
        relevant = [p for p in self.platforms if p.platform_type == "command"]
        if not relevant:
            relevant = self.platforms[:5]
        best = max(relevant, key=lambda p: compute_node_info_sum(self.network, p.platform_id))
        return {
            "agent_type": "command",
            "platform_id": int(best.platform_id),
            "cmd_quality": float(compute_info_measure(self.network, best.platform_id, "command_control")),
            "capacity": int(self.network.IndR["cmd_capacity"][best.platform_id]),
            "expected_utility": float(compute_node_info_sum(self.network, best.platform_id)),
            "available": bool(best.is_available),
        }

    def propose_counter(self, current_allocation: dict, req_vec: RequirementVector) -> dict:
        return {"action": "swap_command", "reason": "reduce_latency",
                "preferred": current_allocation.get("command", 0)}


class ResourceAgent(BaseAgent):
    """资源Agent: 最小化全局资源消耗 & 最大化资源利用率

    效用: U_r = - w1 * total_cost - w2 * waste_rate + w3 * utilization
    """
    def compute_utility(self, assignment: np.ndarray, req_vec: RequirementVector) -> float:
        total_cost = 0.0
        for i, idx in enumerate(assignment):
            p = self.platforms[int(idx) % len(self.platforms)]
            total_cost += p.cost_per_shot
        util = np.mean([1.0 if p.is_available else 0.0 for p in self.platforms])
        return - 0.5 * (total_cost / 500.0) + 0.5 * util

    def compute_bid(self, task_spec: dict, req_vec: RequirementVector) -> dict:
        cheapest_w = min([p for p in self.platforms if p.platform_type == "weapon"],
                         key=lambda p: p.cost_per_shot,
                         default=self.platforms[0])
        total_available = sum(1 for p in self.platforms if p.is_available)
        utilization = total_available / max(1, len(self.platforms))
        return {
            "agent_type": "resource",
            "total_resources": int(total_available),
            "utilization": float(utilization),
            "cheapest_option": int(cheapest_w.platform_id),
            "cheapest_cost": float(cheapest_w.cost_per_shot),
            "expected_utility": float(utilization),
            "available": True,
        }

    def propose_counter(self, current_allocation: dict, req_vec: RequirementVector) -> dict:
        return {"action": "reduce_cost", "reason": "budget_aware",
                "preferred": 0}


# ============================================================
# 3. LangGraph 节点函数
# ============================================================

def _deserialize_req(state: GameState) -> RequirementVector:
    """从state恢复RequirementVector"""
    d = state["req_vector"]
    return RequirementVector(
        w_kill=d["w_kill"], w_cost=d["w_cost"], w_time=d["w_time"],
        w_info=d["w_info"], w_network=d["w_network"],
        target_priority=d.get("target_priority", "all"),
        constraints=d.get("constraints", {}),
    )


def node_init_simulation(state: GameState) -> GameState:
    """初始化: 解析需求, 构建网络和实体"""
    # DeepAgents: 如果 state 中已带有 LLM 解析好的 req_vector，直接使用
    existing_req = state.get("req_vector")
    if existing_req and existing_req.get("w_kill") is not None:
        req_vec = RequirementVector(
            w_kill=existing_req["w_kill"], w_cost=existing_req["w_cost"],
            w_time=existing_req["w_time"], w_info=existing_req["w_info"],
            w_network=existing_req["w_network"],
            target_priority=existing_req.get("target_priority", "all"),
            constraints=existing_req.get("constraints", {}),
        )
    else:
        req_vec = RequirementParser.parse(state["requirement"])
    state["req_vector"] = {
        "w_kill": req_vec.w_kill, "w_cost": req_vec.w_cost,
        "w_time": req_vec.w_time, "w_info": req_vec.w_info,
        "w_network": req_vec.w_network,
        "target_priority": req_vec.target_priority,
        "constraints": req_vec.constraints,
    }
    state["round"] = 0
    state["converged"] = False
    state["done"] = False
    state["bids"] = {}
    state["counter_proposals"] = {}
    state["nash_equilibrium"] = None
    state["reward_params"] = None
    state["messages"] = [
        f"[Init] 需求解析完成: kill={req_vec.w_kill:.2f}, cost={req_vec.w_cost:.2f}, "
        f"time={req_vec.w_time:.2f}, info={req_vec.w_info:.2f}, net={req_vec.w_network:.2f}"
    ]
    return state


def node_coordinator_announce(state: GameState) -> GameState:
    """协调者发布任务招标 (论文2: 派单阶段)"""
    req_vec = _deserialize_req(state)
    task_spec = {
        "target_count": state["n_targets"],
        "target_priority": req_vec.target_priority,
        "time_window": req_vec.constraints.get("time_window", 300),
        "target_value": 0.7,
        "round": state["round"],
    }
    state["messages"].append(
        f"[Round {state['round']}] 协调者发布招标: targets={task_spec['target_count']}, "
        f"priority={task_spec['target_priority']}"
    )
    # 存task_spec到state供bid使用
    state["bids"]["_task_spec"] = task_spec
    return state


def _try_llm_bid(state: GameState, agent_type: str, generator_func) -> Optional[dict]:
    """尝试用LLM生成投标，失败返回None"""
    llm_cfg = state.get("llm_config")
    if not llm_cfg or not llm_cfg.get("api_key"):
        return None
    try:
        client = LLMClient.from_config(llm_cfg)
        if client is None:
            return None
        result = generator_func(state, client)
        if result:
            state["messages"].append(
                f"[LLM-{agent_type}] 生成投标: {json.dumps(result, ensure_ascii=False, default=str)[:120]}"
            )
        return result
    except Exception as e:
        state["messages"].append(f"[LLM-{agent_type}] 调用失败: {e}")
        return None


def node_sensor_bid(state: GameState) -> GameState:
    """传感器Agent投标"""
    seed = 42 + state["round"] * 7  # 每轮不同种子
    n_nodes = state.get("network_nodes", 50)
    n_targets = state.get("n_targets", 8)
    network = KillChainNetwork.create_default(n_nodes, seed=seed)
    platforms = _create_platforms(seed=seed, n_total=n_nodes)
    targets = _create_targets(seed=seed, n_total=n_targets)
    req_vec = _deserialize_req(state)
    task_spec = state["bids"].get("_task_spec", {})
    agent = SensorAgent(
        AgentConfig("sensor", reservation_price=0.3, negotiation_power=1.0, risk_aversion=0.5),
        platforms, targets, network)
    bid = agent.compute_bid(task_spec, req_vec)
    # DeepAgents: 尝试LLM增强投标
    llm_bid = _try_llm_bid(state, "sensor", DeepBidGenerator.sensor_bid)
    if llm_bid:
        bid.update({k: v for k, v in llm_bid.items() if k in bid and v is not None})
    state["bids"]["sensor"] = bid
    state["all_bids_raw"].append(("sensor", bid))
    state["messages"].append(
        f"[Sensor] 投标: platform={bid['platform_id']}, "
        f"info_quality={bid.get('info_quality', 0):.3f}, utility={bid['expected_utility']:.3f}"
    )
    return state


def node_weapon_bid(state: GameState) -> GameState:
    """武器Agent投标"""
    seed = 42 + state["round"] * 7
    n_nodes = state.get("network_nodes", 50)
    n_targets = state.get("n_targets", 8)
    network = KillChainNetwork.create_default(n_nodes, seed=seed)
    platforms = _create_platforms(seed=seed, n_total=n_nodes)
    targets = _create_targets(seed=seed, n_total=n_targets)
    req_vec = _deserialize_req(state)
    task_spec = state["bids"].get("_task_spec", {})
    agent = WeaponAgent(
        AgentConfig("weapon", reservation_price=0.25, negotiation_power=1.2, risk_aversion=0.3),
        platforms, targets, network)
    bid = agent.compute_bid(task_spec, req_vec)
    llm_bid = _try_llm_bid(state, "weapon", DeepBidGenerator.weapon_bid)
    if llm_bid:
        bid.update({k: v for k, v in llm_bid.items() if k in bid and v is not None})
    state["bids"]["weapon"] = bid
    state["all_bids_raw"].append(("weapon", bid))
    state["messages"].append(
        f"[Weapon] 投标: platform={bid['platform_id']}, "
        f"kill_prob={bid['capability']:.3f}, cost={bid['cost_per_shot']:.1f}"
    )
    return state


def node_command_bid(state: GameState) -> GameState:
    """指控Agent投标"""
    seed = 42 + state["round"] * 7
    n_nodes = state.get("network_nodes", 50)
    n_targets = state.get("n_targets", 8)
    network = KillChainNetwork.create_default(n_nodes, seed=seed)
    platforms = _create_platforms(seed=seed, n_total=n_nodes)
    targets = _create_targets(seed=seed, n_total=n_targets)
    req_vec = _deserialize_req(state)
    task_spec = state["bids"].get("_task_spec", {})
    agent = CommandAgent(
        AgentConfig("command", reservation_price=0.2, negotiation_power=0.8, risk_aversion=0.6),
        platforms, targets, network)
    bid = agent.compute_bid(task_spec, req_vec)
    llm_bid = _try_llm_bid(state, "command", DeepBidGenerator.command_bid)
    if llm_bid:
        bid.update({k: v for k, v in llm_bid.items() if k in bid and v is not None})
    state["bids"]["command"] = bid
    state["all_bids_raw"].append(("command", bid))
    state["messages"].append(
        f"[Command] 投标: platform={bid['platform_id']}, "
        f"cmd_quality={bid.get('cmd_quality', 0):.3f}, capacity={bid.get('capacity', 0)}"
    )
    return state


def node_resource_bid(state: GameState) -> GameState:
    """资源Agent投标"""
    seed = 42 + state["round"] * 7
    n_nodes = state.get("network_nodes", 50)
    n_targets = state.get("n_targets", 8)
    network = KillChainNetwork.create_default(n_nodes, seed=seed)
    platforms = _create_platforms(seed=seed, n_total=n_nodes)
    targets = _create_targets(seed=seed, n_total=n_targets)
    req_vec = _deserialize_req(state)
    task_spec = state["bids"].get("_task_spec", {})
    agent = ResourceAgent(
        AgentConfig("resource", reservation_price=0.3, negotiation_power=1.1, risk_aversion=0.7),
        platforms, targets, network)
    bid = agent.compute_bid(task_spec, req_vec)
    llm_bid = _try_llm_bid(state, "resource", DeepBidGenerator.resource_bid)
    if llm_bid:
        bid.update({k: v for k, v in llm_bid.items() if k in bid and v is not None})
    state["bids"]["resource"] = bid
    state["all_bids_raw"].append(("resource", bid))
    state["messages"].append(
        f"[Resource] 投标: utilization={bid['utilization']:.3f}, "
        f"cheapest={bid['cheapest_cost']:.1f}"
    )
    return state


def node_coordinator_evaluate(state: GameState) -> GameState:
    """协调者评分 & 排序 (论文2: 甩单阶段 — 自动生成多套方案)"""
    req_vec = _deserialize_req(state)
    bids = state["bids"]
    scores = {}

    for agent_type in ["sensor", "weapon", "command", "resource"]:
        bid = bids.get(agent_type, {})
        if not bid or not bid.get("available", False):
            scores[agent_type] = 0.0
            continue
        utility = bid.get("expected_utility", 0.0)
        # 加权: 用户需求权重 × Agent个体效用
        w_map = {"sensor": req_vec.w_info, "weapon": req_vec.w_kill,
                  "command": req_vec.w_time, "resource": req_vec.w_cost}
        scores[agent_type] = utility * w_map.get(agent_type, 0.5)

    # 归一化 & 转为Python原生float
    max_s = max(scores.values()) if scores else 1.0
    if max_s > 0:
        scores = {k: float(v / max_s) for k, v in scores.items()}

    state["scores"] = {k: float(v) for k, v in scores.items()}
    state["rankings"] = [str(r) for r in sorted(scores, key=scores.get, reverse=True)]

    # DeepAgents: 尝试LLM协调者评估
    llm_cfg = state.get("llm_config")
    if llm_cfg and llm_cfg.get("api_key"):
        try:
            client = LLMClient.from_config(llm_cfg)
            if client:
                llm_eval = DeepCoordinator.evaluate(state, client)
                if llm_eval and isinstance(llm_eval.get("scores"), dict):
                    state["scores"] = {k: float(v) for k, v in llm_eval["scores"].items()}
                    state["rankings"] = [str(r) for r in llm_eval.get("rankings", state["rankings"])]
                    # 注意：不覆盖 converged，由 check_convergence 独立判定
                    state["messages"].append(
                        f"[Coordinator] LLM评分: {', '.join(f'{k}={v:.3f}' for k, v in state['scores'].items())}"
                    )
        except Exception as e:
            state["messages"].append(f"[Coordinator] LLM评估失败: {e}")

    state["messages"].append(
        f"[Coordinator] 评分: {', '.join(f'{k}={v:.3f}' for k, v in state['scores'].items())}"
    )
    state["messages"].append(f"[Coordinator] 排名: {state['rankings']}")
    return state


def node_check_convergence(state: GameState) -> GameState:
    """检查Nash均衡 & 收敛条件"""
    state["round"] += 1
    scores = state["scores"]

    # 收敛条件:
    # 1. 达到最小博弈轮数后，评分标准差足够小 (各Agent达成一致)
    # 2. 或达到最大轮数 (强制收敛)
    min_rounds = 5  # 最少博弈5轮
    max_rounds = int(state.get("max_rounds", 20))
    score_std = float(np.std(list(scores.values()))) if scores else 1.0
    epsilon = 0.15  # 放宽收敛阈值

    converged_by_score = (state["round"] >= min_rounds and score_std < epsilon)
    converged_by_rounds = state["round"] >= max_rounds

    if converged_by_score or converged_by_rounds:
        state["converged"] = True
        reason = "score" if converged_by_score else "max_rounds"
        state["messages"].append(
            f"[Convergence] 收敛! reason={reason}, round={state['round']}, std={score_std:.4f}"
        )
        # 构建Nash均衡方案 (全部转换为Python原生类型)
        state["nash_equilibrium"] = {
            "scores": {k: float(v) for k, v in scores.items()},
            "rankings": [str(r) for r in state["rankings"]],
            "round": int(state["round"]),
            "nash_product": float(np.prod([max(s, 0.01) for s in scores.values()])),
        }
    else:
        state["messages"].append(
            f"[Check] 未收敛: round={state['round']}, std={score_std:.4f}"
        )

    return state


def node_counter_proposal(state: GameState) -> GameState:
    """反提案阶段: 不满意的Agent提出调整方案 (论文4: 改进合同网)"""
    if state["converged"]:
        return state

    scores = state["scores"]
    rankings = state["rankings"]

    # 得分最低的Agent提出反提案
    lowest_agent = rankings[-1] if rankings else "sensor"
    req_vec = _deserialize_req(state)

    counter = {
        "agent": lowest_agent,
        "current_score": scores.get(lowest_agent, 0),
        "proposal": f"improve_{lowest_agent}_allocation",
        "request": "re-bid_with_adjusted_weights",
    }

    # 调整权重: 给最低分Agent更多谈判力量
    adjustment = 0.1 * (1.0 - scores.get(lowest_agent, 0))
    state["counter_proposals"] = {"lowest_agent": lowest_agent, "adjustment": adjustment}
    state["messages"].append(
        f"[Counter] {lowest_agent} 提出反提案, adjustment={adjustment:.3f}"
    )
    return state


def node_generate_reward_params(state: GameState) -> GameState:
    """生成PPO奖励参数 (博弈结果 → α,β,γ,δ,ε)"""
    req_vec = _deserialize_req(state)
    equilibrium = state.get("nash_equilibrium") or {}
    scores = equilibrium.get("scores", {"sensor": 0.5, "weapon": 0.5, "command": 0.5, "resource": 0.5})

    # 核心映射逻辑:
    # α (kill):    weapon得分 × 用户杀伤权重
    # β (cost):    resource得分 × 用户成本权重
    # γ (time):    command得分 × 用户时间权重
    # δ (info):    sensor得分 × 用户信息权重
    # ε (network): sensor×command混合 × 用户网络权重

    # 解析约束标签 → 数值
    constraints = req_vec.constraints
    max_cost = 1000.0
    if isinstance(constraints.get("max_cost"), (int, float)):
        max_cost = float(constraints["max_cost"])
    elif "max_cost_500" in constraints:
        max_cost = 500.0
    elif "max_cost_1000" in constraints:
        max_cost = 1000.0

    time_window = 300.0
    if isinstance(constraints.get("time_window"), (int, float)):
        time_window = float(constraints["time_window"])
    elif "time_window_180" in constraints:
        time_window = 180.0
    elif "time_window_300" in constraints:
        time_window = 300.0

    min_kill_prob = 0.5
    if isinstance(constraints.get("min_kill_prob"), (int, float)):
        min_kill_prob = float(constraints["min_kill_prob"])
    elif "min_kill_08" in constraints:
        min_kill_prob = 0.8
    elif "min_kill_09" in constraints:
        min_kill_prob = 0.9

    reward_params = {
        "alpha":    round(scores.get("weapon", 0.5)   * req_vec.w_kill * 2.0, 4),
        "beta":     round(scores.get("resource", 0.5) * req_vec.w_cost * 2.0, 4),
        "gamma":    round(scores.get("command", 0.5)  * req_vec.w_time * 2.0, 4),
        "delta":    round(scores.get("sensor", 0.5)   * req_vec.w_info * 2.0, 4),
        "epsilon":  round(0.5 * (scores.get("sensor", 0.5) + scores.get("command", 0.5))
                          * req_vec.w_network * 2.0, 4),
        "min_kill_prob": min_kill_prob,
        "max_cost": max_cost,
        "time_window": time_window,
        "nash_product": equilibrium.get("nash_product", 0.0),
        "rounds_to_converge": equilibrium.get("round", state["round"]),
    }

    # DeepAgents: 尝试LLM生成奖励参数，成功则覆盖公式结果
    llm_cfg = state.get("llm_config")
    if llm_cfg and llm_cfg.get("api_key"):
        try:
            client = LLMClient.from_config(llm_cfg)
            if client:
                llm_reward = DeepRewardGenerator.generate(state, client)
                if llm_reward and "alpha" in llm_reward:
                    for k in ["alpha", "beta", "gamma", "delta", "epsilon",
                              "min_kill_prob", "max_cost", "time_window"]:
                        if k in llm_reward:
                            reward_params[k] = llm_reward[k]
                    state["messages"].append("[RewardParams] LLM生成参数已应用")
        except Exception as e:
            state["messages"].append(f"[RewardParams] LLM生成失败: {e}")

    state["reward_params"] = reward_params
    state["done"] = True
    state["messages"].append(
        f"[RewardParams] α={reward_params['alpha']:.4f} β={reward_params['beta']:.4f} "
        f"γ={reward_params['gamma']:.4f} δ={reward_params['delta']:.4f} ε={reward_params['epsilon']:.4f}"
    )
    state["messages"].append(
        f"[Complete] Nash乘积={reward_params['nash_product']:.4f}, "
        f"收敛轮数={reward_params['rounds_to_converge']}"
    )
    return state


# ============================================================
# 4. 辅助函数
# ============================================================

def _create_platforms(seed: int = 42, n_total: int = 50):
    """创建默认平台列表，按比例分配类型"""
    from model import Platform
    rng = np.random.default_rng(seed)
    n_sensor = max(1, n_total * 3 // 10)
    n_weapon = max(1, n_total * 3 // 10)
    n_command = max(1, n_total * 2 // 10)
    n_platform = n_total - n_sensor - n_weapon - n_command
    type_dist = (["sensor"] * n_sensor + ["weapon"] * n_weapon +
                 ["command"] * n_command + ["platform"] * max(1, n_platform))
    plats = []
    for i, t in enumerate(type_dist):
        plats.append(Platform(
            platform_id=i, platform_type=t,
            capabilities=rng.choice(["radar", "ir", "sar", "missile", "laser", "ew"], size=2, replace=False).tolist(),
            kill_prob=rng.uniform(0.3, 0.95) if t == "weapon" else 0.1,
            cost_per_shot=rng.uniform(10, 200) if t == "weapon" else 5.0,
            reload_time=rng.uniform(5, 60),
            max_range=rng.uniform(20, 80),
            position=rng.uniform(0, 100, 2),
            ammo=rng.integers(5, 20),
        ))
    return plats


def _create_targets(seed: int = 43, n_total: int = 16):
    """创建目标列表: 高价值(2-4) + 普通(其余)"""
    from model import Target
    rng = np.random.default_rng(seed)
    tgts = []
    n_high = min(max(2, n_total // 6), 4)
    for i in range(n_high):
        tgts.append(Target(
            target_id=i, value=rng.uniform(0.8, 1.0),
            is_time_sensitive=rng.random() > 0.3,
            time_window=rng.uniform(60, 300),
            position=rng.uniform(0, 100, 2), velocity=rng.uniform(-5, 5, 2),
            priority=1,
        ))
    for i in range(n_high, n_total):
        tgts.append(Target(
            target_id=i, value=rng.uniform(0.2, 0.6),
            is_time_sensitive=False, time_window=np.inf,
            position=rng.uniform(0, 100, 2), velocity=rng.uniform(-3, 3, 2),
            priority=rng.integers(2, 4),
        ))
    return tgts


# ============================================================
# 5. 构建 LangGraph 图
# ============================================================

def build_game_graph() -> StateGraph:
    """构建多Agent博弈的LangGraph状态图

    图结构:
      init → coordinator_announce → [sensor|weapon|command|resource]_bid
          → coordinator_evaluate → check_convergence
          → (if not converged) counter_proposal → coordinator_announce (loop)
          → (if converged) generate_reward_params → END
    """
    graph = StateGraph(GameState)

    # 添加节点
    graph.add_node("init", node_init_simulation)
    graph.add_node("coordinator_announce", node_coordinator_announce)
    graph.add_node("sensor_bid", node_sensor_bid)
    graph.add_node("weapon_bid", node_weapon_bid)
    graph.add_node("command_bid", node_command_bid)
    graph.add_node("resource_bid", node_resource_bid)
    graph.add_node("coordinator_evaluate", node_coordinator_evaluate)
    graph.add_node("check_convergence", node_check_convergence)
    graph.add_node("counter_proposal", node_counter_proposal)
    graph.add_node("generate_reward_params", node_generate_reward_params)

    # 边: init → coordinator_announce
    graph.set_entry_point("init")
    graph.add_edge("init", "coordinator_announce")

    # 协调者发布后 → 4个Agent并行投标 (在LangGraph中顺序执行)
    graph.add_edge("coordinator_announce", "sensor_bid")
    graph.add_edge("sensor_bid", "weapon_bid")
    graph.add_edge("weapon_bid", "command_bid")
    graph.add_edge("command_bid", "resource_bid")

    # 投标完成 → 协调者评估
    graph.add_edge("resource_bid", "coordinator_evaluate")

    # 评估 → 检查收敛
    graph.add_edge("coordinator_evaluate", "check_convergence")

    # 条件路由: 收敛 → 生成参数; 否则 → 反提案 → 重新招标
    def should_continue(state: GameState) -> str:
        if state.get("converged", False):
            return "generate_reward_params"
        if state.get("round", 0) >= state.get("max_rounds", 20):
            return "generate_reward_params"
        return "counter_proposal"

    graph.add_conditional_edges("check_convergence", should_continue, {
        "generate_reward_params": "generate_reward_params",
        "counter_proposal": "counter_proposal",
    })

    # 反提案后 → 重新招标 (循环)
    graph.add_edge("counter_proposal", "coordinator_announce")

    # 奖励参数生成 → 结束
    graph.add_edge("generate_reward_params", END)

    return graph


# ============================================================
# 6. GameEngine 封装
# ============================================================

class GameEngine:
    """多Agent博弈引擎 — 对外统一接口

    用法:
        engine = GameEngine(max_rounds=20)
        result = engine.run("最大化杀伤高价值目标，控制弹药成本")
        # result["reward_params"] → PPO奖励参数
    """

    def __init__(self, max_rounds: int = 20):
        self.max_rounds = max_rounds
        self.graph = build_game_graph()
        self.app = self.graph.compile()  # 不使用checkpointer避免msgpack序列化问题

    def run(self, requirement: str, n_nodes: int = 50, n_targets: int = 8,
            llm_config: dict = None) -> dict:
        """运行完整博弈流程

        Args:
            llm_config: DeepAgents LLM 配置 {base_url, api_key, model}
        """
        initial_state: GameState = {
            "requirement": requirement,
            "req_vector": {},
            "network_nodes": n_nodes,
            "n_targets": n_targets,
            "round": 0,
            "max_rounds": self.max_rounds,
            "converged": False,
            "bids": {},
            "all_bids_raw": [],
            "scores": {},
            "rankings": [],
            "counter_proposals": {},
            "disagreement_points": {},
            "nash_equilibrium": None,
            "pareto_front": [],
            "reward_params": None,
            "messages": [],
            "done": False,
            "llm_config": llm_config,
        }

        config = {"configurable": {"thread_id": "game_session_1"}}
        final_state = self.app.invoke(initial_state, config)

        return {
            "reward_params": final_state["reward_params"],
            "nash_equilibrium": final_state["nash_equilibrium"],
            "messages": final_state["messages"],
            "rounds": final_state["round"],
            "converged": final_state["converged"],
        }


# ============================================================
# 7. 独立测试入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("多Agent博弈系统测试 (LangGraph)")
    print("=" * 60)

    engine = GameEngine(max_rounds=20)

    test_requirements = [
        "最大化杀伤高价值目标，同时控制弹药成本在500万以内",
        "优先打击时敏目标，要求3分钟内完成杀伤链闭环",
        "在预算约束下摧毁尽可能多的敌方指挥节点，注重信息优势",
    ]

    for req in test_requirements:
        print(f"\n{'─' * 50}")
        print(f"需求: {req}")
        result = engine.run(req)
        rp = result["reward_params"]
        print(f"  奖励参数: α={rp['alpha']:.4f} β={rp['beta']:.4f} "
              f"γ={rp['gamma']:.4f} δ={rp['delta']:.4f} ε={rp['epsilon']:.4f}")
        print(f"  Nash乘积: {rp['nash_product']:.4f}")
        print(f"  收敛轮数: {rp['rounds_to_converge']}")
        print(f"  消息数: {len(result['messages'])}")
