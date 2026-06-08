"""
DeepAgents 架构 — 基于 LLM 的多 Agent 增强模块

核心设计:
  1. 保留原有硬编码逻辑作为 fallback（流程不变、结构不变）
  2. 每个 Agent 投标 / 协调者评估 / 奖励参数生成 先尝试 LLM
  3. LLM 失败或超时 → 自动回退到规则逻辑
  4. 默认使用 DeepSeek 兼容接口 (OpenAI API 格式)

用法:
    from deep_agents import LLMClient, DeepBidGenerator, DeepRewardGenerator
    client = LLMClient(base_url="...", api_key="...", model="deepseek-chat")
    bid = DeepBidGenerator.sensor_bid(state, client)  # 失败返回 None
"""
import json
import os
from urllib import request, error
from typing import Optional, Dict, Any


# ============================================================
# 1. LLM 客户端
# ============================================================

class LLMClient:
    """通用 OpenAI 兼容 LLM 客户端"""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 20):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """单轮对话，返回模型生成的纯文本"""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 1024,
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(url, data=data, headers=headers, method="POST")

        with request.urlopen(req, timeout=self.timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"]

    def chat_parse_json(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        """对话并尝试解析返回的 JSON，失败返回 None"""
        try:
            content = self.chat(system_prompt, user_prompt)
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            return json.loads(content)
        except Exception:
            return None

    @classmethod
    def from_config(cls, config: Optional[Dict[str, str]]) -> Optional["LLMClient"]:
        """从配置字典创建客户端，配置无效返回 None"""
        if not config or not config.get("api_key"):
            return None
        return cls(
            base_url=config.get("base_url", "https://api.deepseek.com/v1"),
            api_key=config["api_key"],
            model=config.get("model", "deepseek-chat"),
            timeout=int(config.get("timeout", 20)),
        )


# ============================================================
# 2. Prompt 模板
# ============================================================

SYSTEM_SENSOR = """你是传感器Agent（SensorAgent），负责在分布式杀伤链中为任务提供最优探测方案。
你的核心目标是：最大化探测覆盖率 + 信息质量，同时控制资源消耗。
你必须以 JSON 格式返回投标结果，不要包含任何解释文字。"""

SYSTEM_WEAPON = """你是武器Agent（WeaponAgent），负责在分布式杀伤链中为任务提供最优打击方案。
你的核心目标是：最大化杀伤概率 × 目标价值，同时控制弹药成本和时间延迟。
你必须以 JSON 格式返回投标结果，不要包含任何解释文字。"""

SYSTEM_COMMAND = """你是指控Agent（CommandAgent），负责在分布式杀伤链中为任务提供最优指挥控制方案。
你的核心目标是：最大化指挥质量 + 决策速度，同时最小化网络延迟。
你必须以 JSON 格式返回投标结果，不要包含任何解释文字。"""

SYSTEM_RESOURCE = """你是资源Agent（ResourceAgent），负责在分布式杀伤链中优化全局资源配置。
你的核心目标是：最小化总成本 + 最大化资源利用率。
你必须以 JSON 格式返回投标结果，不要包含任何解释文字。"""

SYSTEM_COORDINATOR = """你是协调者Agent（Coordinator），负责评估多Agent博弈中的投标并判断收敛。
你需要根据各Agent的投标质量和用户需求权重，给出公正评分。
你必须以 JSON 格式返回评估结果，不要包含任何解释文字。"""

SYSTEM_REWARD = """你是PPO奖励参数生成专家。你的任务是根据Nash均衡结果和用户需求，生成定制化的PPO强化学习奖励参数。
参数含义：α=杀伤效能, β=成本控制, γ=时间效率, δ=信息优势, ε=网络质量。
你必须以 JSON 格式返回结果，不要包含任何解释文字。"""


# ============================================================
# 3. Deep Bid Generator — 各Agent LLM 投标生成
# ============================================================

class DeepBidGenerator:
    """LLM 增强的 Agent 投标生成器

    每个方法先尝试 LLM 生成结构化投标，失败返回 None，调用方负责 fallback。
    """

    @classmethod
    def _build_context(cls, state: Dict[str, Any]) -> str:
        """从 state 构建通用上下文"""
        req = state.get("req_vector", {})
        task = state.get("bids", {}).get("_task_spec", {})
        ctx = f"""【当前需求向量】
- 杀伤效能权重: {req.get('w_kill', 0.5):.2f}
- 成本控制权重: {req.get('w_cost', 0.5):.2f}
- 时间效率权重: {req.get('w_time', 0.5):.2f}
- 信息优势权重: {req.get('w_info', 0.5):.2f}
- 网络质量权重: {req.get('w_network', 0.5):.2f}
- 目标优先级: {req.get('target_priority', 'all')}

【任务规格】
- 目标数量: {task.get('target_count', 8)}
- 时间窗口: {task.get('time_window', 300)} 秒
- 博弈轮次: {state.get('round', 0)}/{state.get('max_rounds', 20)}
"""
        return ctx

    @classmethod
    def sensor_bid(cls, state: Dict[str, Any], client: LLMClient) -> Optional[Dict[str, Any]]:
        prompt = (
            cls._build_context(state)
            + "\n请生成传感器Agent的投标，JSON格式如下：\n"
            + '{"platform_id": 整数, "capability": 0.0-1.0, "info_quality": 0.0-1.0, "cost": 数值, "expected_utility": 0.0-1.0, "available": true}'
        )
        return client.chat_parse_json(SYSTEM_SENSOR, prompt)

    @classmethod
    def weapon_bid(cls, state: Dict[str, Any], client: LLMClient) -> Optional[Dict[str, Any]]:
        prompt = (
            cls._build_context(state)
            + "\n请生成武器Agent的投标，JSON格式如下：\n"
            + '{"platform_id": 整数, "capability": 0.0-1.0, "cost_per_shot": 数值, "range": 数值, "ammo": 整数, "expected_utility": 0.0-1.0, "available": true}'
        )
        return client.chat_parse_json(SYSTEM_WEAPON, prompt)

    @classmethod
    def command_bid(cls, state: Dict[str, Any], client: LLMClient) -> Optional[Dict[str, Any]]:
        prompt = (
            cls._build_context(state)
            + "\n请生成指控Agent的投标，JSON格式如下：\n"
            + '{"platform_id": 整数, "cmd_quality": 0.0-1.0, "capacity": 整数, "expected_utility": 0.0-1.0, "available": true}'
        )
        return client.chat_parse_json(SYSTEM_COMMAND, prompt)

    @classmethod
    def resource_bid(cls, state: Dict[str, Any], client: LLMClient) -> Optional[Dict[str, Any]]:
        prompt = (
            cls._build_context(state)
            + "\n请生成资源Agent的投标，JSON格式如下：\n"
            + '{"total_resources": 整数, "utilization": 0.0-1.0, "cheapest_option": 整数, "cheapest_cost": 数值, "expected_utility": 0.0-1.0, "available": true}'
        )
        return client.chat_parse_json(SYSTEM_RESOURCE, prompt)


# ============================================================
# 4. Deep Coordinator — 协调者 LLM 评估
# ============================================================

class DeepCoordinator:
    """LLM 增强的协调者评估"""

    @classmethod
    def evaluate(cls, state: Dict[str, Any], client: LLMClient) -> Optional[Dict[str, Any]]:
        req = state.get("req_vector", {})
        bids = state.get("bids", {})
        bids_json = json.dumps({k: v for k, v in bids.items() if not k.startswith("_")},
                                ensure_ascii=False, default=str)
        prompt = (
            f"【需求向量】\n"
            f"- 杀伤权重: {req.get('w_kill', 0.5):.2f}\n"
            f"- 成本权重: {req.get('w_cost', 0.5):.2f}\n"
            f"- 时间权重: {req.get('w_time', 0.5):.2f}\n"
            f"- 信息权重: {req.get('w_info', 0.5):.2f}\n"
            f"- 网络权重: {req.get('w_network', 0.5):.2f}\n\n"
            f"【各Agent投标】\n{bids_json}\n\n"
            f"请评估并返回JSON：\n"
            f'{{"scores": {{"sensor": 0.0-1.0, "weapon": 0.0-1.0, "command": 0.0-1.0, "resource": 0.0-1.0}}, '
            f'"rankings": ["sensor", "weapon", "command", "resource"], '
            f'"converged": true/false, '
            f'"counter_proposal": {{"lowest_agent": "xxx", "adjustment": 0.0-0.5}}'
            f'}}'
        )
        return client.chat_parse_json(SYSTEM_COORDINATOR, prompt)


# ============================================================
# 5. Deep Reward Generator — 奖励参数 LLM 生成
# ============================================================

class DeepRewardGenerator:
    """LLM 增强的 PPO 奖励参数生成"""

    @classmethod
    def generate(cls, state: Dict[str, Any], client: LLMClient) -> Optional[Dict[str, Any]]:
        req = state.get("req_vector", {})
        eq = state.get("nash_equilibrium", {})
        scores = eq.get("scores", {})
        prompt = (
            f"【Nash均衡结果】\n"
            f"- Agent评分: {json.dumps(scores, ensure_ascii=False)}\n"
            f"- Nash乘积: {eq.get('nash_product', 0):.4f}\n"
            f"- 收敛轮数: {eq.get('round', 0)}\n\n"
            f"【需求向量】\n"
            f"- 杀伤权重: {req.get('w_kill', 0.5):.2f}\n"
            f"- 成本权重: {req.get('w_cost', 0.5):.2f}\n"
            f"- 时间权重: {req.get('w_time', 0.5):.2f}\n"
            f"- 信息权重: {req.get('w_info', 0.5):.2f}\n"
            f"- 网络权重: {req.get('w_network', 0.5):.2f}\n"
            f"- 目标优先级: {req.get('target_priority', 'all')}\n"
            f"- 约束: {json.dumps(req.get('constraints', {}), ensure_ascii=False)}\n\n"
            f"请生成PPO奖励参数，JSON格式：\n"
            f'{{"alpha": 0.0-2.0, "beta": 0.0-2.0, "gamma": 0.0-2.0, "delta": 0.0-2.0, "epsilon": 0.0-2.0, '
            f'"min_kill_prob": 0.0-1.0, "max_cost": 数值, "time_window": 数值}}'
        )
        result = client.chat_parse_json(SYSTEM_REWARD, prompt)
        if result:
            # 确保数值类型正确
            for k in ["alpha", "beta", "gamma", "delta", "epsilon",
                      "min_kill_prob", "max_cost", "time_window"]:
                if k in result:
                    try:
                        result[k] = round(float(result[k]), 4) if k != "max_cost" and k != "time_window" else float(result[k])
                    except (ValueError, TypeError):
                        pass
            result["nash_product"] = eq.get("nash_product", 0.0)
            result["rounds_to_converge"] = eq.get("round", state.get("round", 0))
        return result
