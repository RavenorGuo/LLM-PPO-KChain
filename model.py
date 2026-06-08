"""
模型与仿真环境 — 基于论文3 "分布式杀伤链中网络与信息聚优数学问题探析"
核心模型: G = (V, C, R, P) + 信息度量 + F2T2EA活动模型 + Gym环境

参考:
- 论文3 (分布式杀伤链): G=(V,C,R,P), 网络行为范式, 信息度量 y∈[0,1], 方程(10)
- 论文 (即时杀伤链): F2T2EA 六阶段, 时敏目标优先, 派单-甩单-接单
- 论文4 (集群杀伤链): 改进合同网协议任务分配
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from collections import OrderedDict
import gymnasium as gym
from gymnasium import spaces


# ============================================================
# 1. 核心数学模型 (论文3: G=(V,C,R,P))
# ============================================================

@dataclass
class KillChainNetwork:
    """论文3: 数据链网络图 G = (V, C, R, P)
    V: 节点集 (M个作战节点)
    C: 连通矩阵 (M×M) — 物理连通性
    R: 指挥/协同关系矩阵 (M×M)
    P: 信息交互矩阵 (M×M)
    """
    V: np.ndarray          # (M,) 节点ID数组
    C: np.ndarray          # (M,M) 连通矩阵, c_mn ∈ {0,1}
    R: np.ndarray          # (M,M) 指挥/协同关系矩阵, r_mn ∈ {0,1}
    P: np.ndarray          # (M,M) 信息交互矩阵, p_mn ∈ {0,1}

    # 节点类型映射: 0=传感器, 1=武器, 2=指控, 3=平台
    node_types: np.ndarray = None
    # 节点能力向量 (每个节点能同时参与的任务数)
    node_capacities: np.ndarray = None
    # 指标体系
    IndC: dict = field(default_factory=dict)   # 连通指标: {入网时间, 信噪比, 误码率, 丢包率}
    IndR: dict = field(default_factory=dict)   # 指挥指标: {指挥容量, 协同子网容量}
    IndP: dict = field(default_factory=dict)   # 信息交互指标: {功能域数, 信息种类, 交互频次, 信息容量}

    @classmethod
    def create_default(cls, n_nodes: int = 50, seed: int = 42):
        """创建默认网络拓扑，含传感器/武器/指控/平台四类节点"""
        rng = np.random.default_rng(seed)
        V = np.arange(n_nodes)
        # 随机分配节点类型
        node_types = rng.choice([0, 1, 2, 3], size=n_nodes, p=[0.3, 0.3, 0.2, 0.2])
        # 连通矩阵: 距离越远连通概率越低
        positions = rng.uniform(0, 100, (n_nodes, 2))
        dist = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
        C = (dist < rng.uniform(20, 40)).astype(np.float32) * (1 - np.eye(n_nodes))
        # 指挥关系: 指控节点指挥其连通范围内的武器和传感器
        R = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        cmd_nodes = np.where(node_types == 2)[0]
        for cmd in cmd_nodes:
            subordinates = np.where((C[cmd] > 0) & (node_types != 2))[0]
            R[cmd, subordinates] = 1.0
        # 信息交互矩阵: 连通节点间有概率信息交互
        P = (C * rng.uniform(0, 1, (n_nodes, n_nodes)) > 0.5).astype(np.float32)
        # 节点能力
        capacities = rng.integers(1, 6, size=n_nodes).astype(np.float32)

        return cls(
            V=V, C=C, R=R, P=P,
            node_types=node_types,
            node_capacities=capacities,
            IndC={"snr": rng.uniform(5, 20, (n_nodes, n_nodes)),
                  "ber": rng.uniform(1e-6, 1e-3, (n_nodes, n_nodes)),
                  "delay": rng.uniform(1, 50, (n_nodes, n_nodes))},
            IndR={"cmd_capacity": rng.integers(1, 10, size=n_nodes)},
            IndP={"info_types": rng.integers(1, 5, (n_nodes, n_nodes))},
        )


# ============================================================
# 2. 网络行为范式 & 信息度量 (论文3: 公式7/8/9)
# ============================================================

# 论文3 图7: 5种网络行为范式
NETWORK_PARADIGMS = {
    "intel_distribution":    ["p2p", "chain", "star"],      # 情报分发
    "situation_sharing":     ["full_connect", "relay", "star"],  # 态势共享
    "command_control":       ["p2p", "star", "hierarchy"],  # 指挥控制
    "tactical_coordination": ["formation"],                  # 战术协同
    "weapon_coordination":   ["full_connect", "star"],       # 武器协同
}


def compute_info_measure(network: KillChainNetwork, node_idx: int,
                         paradigm_type: str) -> float:
    """论文3 公式(7)(8)(9): θ: {IndC, IndR, IndP} → y ∈ [0,1]
    计算节点vm在第b种网络行为范式下的信息度量 y^b_m
    """
    if paradigm_type not in NETWORK_PARADIGMS:
        raise ValueError(f"未知范式类型: {paradigm_type}")

    # 与该节点连通的邻居
    neighbors = np.where(network.C[node_idx] > 0)[0]
    if len(neighbors) == 0:
        return 0.0

    connected_nodes = np.concatenate([neighbors, [node_idx]])
    # 公式(8): y^b_m = (1/d_m) * Σ_j y^b_{j,m}
    snr_norm = float(np.clip(network.IndC["snr"][node_idx, neighbors].mean() / 20.0, 0, 1))
    ber_val = float(network.IndC["ber"][node_idx, neighbors].mean())
    ber_score = 1.0 - np.clip(np.log10(1.0 / max(ber_val, 1e-12)) / 6.0, 0, 1)
    delay_score = 1.0 - float(np.clip(network.IndC["delay"][node_idx, neighbors].mean() / 50.0, 0, 1))
    info_score = network.P[node_idx, neighbors].mean()

    # 加权平均 → y ∈ [0,1]
    y = 0.25 * snr_norm + 0.25 * ber_score + 0.25 * delay_score + 0.25 * info_score
    return float(np.clip(y, 0.0, 1.0))


def compute_node_info_sum(network: KillChainNetwork, node_idx: int) -> float:
    """论文3 公式(9): 节点vm的总信息度量 y_m = (1/x_m) * Σ_b y^b_m"""
    y_sum = 0.0
    for paradigm in NETWORK_PARADIGMS:
        y_sum += compute_info_measure(network, node_idx, paradigm)
    return y_sum / len(NETWORK_PARADIGMS)


def compute_subnet_quality(network: KillChainNetwork, task_subnet: np.ndarray) -> float:
    """计算任务子网 G_t 的整体信息质量，用于方程(10)约束检查"""
    selected = np.where(task_subnet > 0)[0]
    if len(selected) == 0:
        return 0.0
    return np.mean([compute_node_info_sum(network, i) for i in selected])


# ============================================================
# 3. 作战实体: 目标 & 平台
# ============================================================

@dataclass
class Target:
    """时敏目标 (论文2: 高价值/高威胁/时间窗口紧迫)"""
    target_id: int
    value: float            # 目标价值 (高价值=1.0, 普通=0.5)
    is_time_sensitive: bool  # 是否时敏
    time_window: float       # 可打击时间窗口 (秒)
    position: np.ndarray     # (2,) 或 (3,) 位置
    velocity: np.ndarray     # 速度向量
    required_capability: list = field(default_factory=list)  # 所需杀伤能力
    priority: int = 1        # 优先级: 1=高, 2=中, 3=低
    destroyed: bool = False  # 击毁模式用：是否已被摧毁


@dataclass
class Platform:
    """作战平台 (传感器/武器/指控节点)"""
    platform_id: int
    platform_type: str       # "sensor" | "weapon" | "command" | "platform"
    capabilities: list       # 能力标签: ["radar","ir","sar","missile","laser"]
    kill_prob: float         # 对目标的杀伤概率 p_kill ∈ [0,1]
    cost_per_shot: float     # 单次射击成本
    reload_time: float       # 再装填时间 (秒)
    max_range: float         # 最大射程/探测距离
    position: np.ndarray     # 当前位置
    ammo: int = 10           # 弹药余量
    is_available: bool = True


# ============================================================
# 4. 需求解析器
# ============================================================

@dataclass
class RequirementVector:
    """用户需求 → 结构化权重向量"""
    w_kill: float = 0.5      # 杀伤效能权重
    w_cost: float = 0.5      # 成本意识权重 (越高=越注重省钱)
    w_time: float = 0.5      # 时间效率权重
    w_info: float = 0.5      # 信息优势权重
    w_network: float = 0.5   # 网络质量权重
    target_priority: str = "all"  # "high_value" | "time_sensitive" | "all"
    constraints: dict = field(default_factory=dict)

    def to_array(self) -> np.ndarray:
        return np.array([self.w_kill, self.w_cost, self.w_time, self.w_info, self.w_network])

    def normalize(self):
        arr = self.to_array()
        if arr.sum() > 0:
            arr = arr / arr.sum()
        self.w_kill, self.w_cost, self.w_time, self.w_info, self.w_network = arr


class RequirementParser:
    """关键词规则映射: 自然语言需求 → RequirementVector

    示例输入:
    - "最大化杀伤高价值目标，同时控制弹药成本"
    - "优先打击时敏目标，保证快速闭环杀伤链"
    - "在预算500万内摧毁尽可能多的敌方指挥节点"
    """

    # 关键词→维度映射表
    KILL_KEYWORDS = [
        "杀伤", "摧毁", "打击", "消灭", "毁伤", "致命",
        "高价值", "高威胁", "最大化", "尽可能多", "更多目标",
        "重点目标", "指挥节点", "关键目标"
    ]
    COST_KEYWORDS = [
        "费用", "成本", "预算", "省钱", "节约", "消耗",
        "弹药", "资源消耗", "经济", "廉价", "低消耗", "控制成本"
    ]
    TIME_KEYWORDS = [
        "快速", "时间", "时敏", "时效", "速度", "分钟",
        "秒", "立刻", "立即", "快速响应", "敏捷", "及时",
        "窗口", "紧迫", "第一时间"
    ]
    INFO_KEYWORDS = [
        "信息", "情报", "态势", "感知", "探测", "识别",
        "定位", "跟踪", "监视", "侦察", "信息优势", "数据"
    ]
    NETWORK_KEYWORDS = [
        "网络", "通信", "连通", "链路", "数据链", "组网",
        "通联", "带宽", "网络质量", "抗干扰"
    ]

    @classmethod
    def parse(cls, requirement: str) -> RequirementVector:
        """解析需求文本 → RequirementVector"""
        text = requirement.lower()

        def count_hits(keywords):
            return sum(1 for kw in keywords if kw in text)

        # 原始计数
        raw_kill = count_hits(cls.KILL_KEYWORDS)
        raw_cost = count_hits(cls.COST_KEYWORDS)
        raw_time = count_hits(cls.TIME_KEYWORDS)
        raw_info = count_hits(cls.INFO_KEYWORDS)
        raw_network = count_hits(cls.NETWORK_KEYWORDS)

        # 映射到 [0.1, 1.0]，基础值0.1确保最低权重
        total = raw_kill + raw_cost + raw_time + raw_info + raw_network
        if total == 0:
            # 无匹配 → 均匀权重
            return RequirementVector()

        base = 0.1
        scale = 0.9
        vec = RequirementVector(
            w_kill=base + scale * raw_kill / max(total, 1),
            w_cost=base + scale * raw_cost / max(total, 1),
            w_time=base + scale * raw_time / max(total, 1),
            w_info=base + scale * raw_info / max(total, 1),
            w_network=base + scale * raw_network / max(total, 1),
        )

        # 目标优先级推断
        if "高价值" in text or "高威胁" in text:
            vec.target_priority = "high_value"
        elif "时敏" in text or "时间窗口" in text:
            vec.target_priority = "time_sensitive"
        else:
            vec.target_priority = "all"

        # 约束提取
        for kw, tag in [("500万", "max_cost_500"), ("1000万", "max_cost_1000"),
                        ("3分钟", "time_window_180"), ("5分钟", "time_window_300"),
                        ("杀伤概率0.8", "min_kill_08"), ("杀伤概率0.9", "min_kill_09")]:
            if kw in requirement:
                vec.constraints[tag] = True

        return vec


# ============================================================
# 5. Gym仿真环境: KillChainEnv
# ============================================================

class KillChainEnv(gym.Env):
    """
    分布式杀伤链仿真环境

    状态空间: [各节点信息度量y_i, 目标特征, 网络全局质量]
    动作空间: MultiDiscrete — 为每个目标选择(传感器, 武器, 指控)三元组
    奖励函数: 由外部 reward_params 定制 (来自Agent博弈结果)

    六阶段 F2T2EA 闭环:
      Find → Fix → Track → Target → Engage → Assess
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, network: KillChainNetwork = None,
                 platforms: list = None, targets: list = None,
                 reward_params: dict = None, max_steps: int = 100,
                 count_mode: str = "delivery"):
        super().__init__()

        self.network = network or KillChainNetwork.create_default(50)
        self.n_nodes = len(self.network.V)

        self.platforms = platforms or self._create_default_platforms()
        self.targets = targets or self._create_default_targets()
        self.max_steps = max_steps
        self.reward_params = reward_params or {
            "alpha": 1.0, "beta": 0.5, "gamma": 0.5,
            "delta": 0.3, "epsilon": 0.3,
            "min_kill_prob": 0.5, "max_cost": 1000.0, "time_window": 300.0
        }

        self.n_targets = len(self.targets)
        self.n_platforms = len(self.platforms)

        # 保存初始弹药快照，用于 reset 时正确恢复
        self._initial_ammo = {p.platform_id: int(p.ammo) for p in self.platforms}

        # 若传入外部 targets，确保高价值目标至少有一个落在某武器射程内
        if targets is not None:
            weapons = [p for p in self.platforms if p.platform_type == "weapon"]
            for t in self.targets:
                if t.priority == 1 and weapons:
                    in_range = any(
                        np.linalg.norm(t.position - w.position) <= w.max_range
                        for w in weapons
                    )
                    if not in_range:
                        w = min(weapons, key=lambda w: np.linalg.norm(t.position - w.position))
                        direction = t.position - w.position
                        dist = np.linalg.norm(direction)
                        if dist > 0:
                            t.position = w.position + direction / dist * (w.max_range * 0.6)
                        else:
                            t.position = w.position + np.array([5.0, 0.0])

        # 状态: 节点信息度量 + 目标状态 + 网络全局
        self.obs_dim = self.n_nodes + self.n_targets * 4 + 3

        # 动作: Discrete — 选择哪个武器平台打击当前最高优先级目标
        # 传感器和指控由环境自动匹配（选择连通且信息质量最高的）
        self.weapon_indices = [i for i, p in enumerate(self.platforms) if p.platform_type == "weapon"]
        if not self.weapon_indices:
            self.weapon_indices = list(range(len(self.platforms)))  # fallback
        self.action_space = spaces.Discrete(len(self.weapon_indices))
        self.observation_space = spaces.Box(
            low=-1.0, high=10.0, shape=(self.obs_dim,), dtype=np.float32
        )

        self.count_mode = count_mode  # "delivery" | "destruction"

        self.current_step = 0
        self.total_cost = 0.0
        self.kills = 0
        self.episode_rewards = []

    def _create_default_platforms(self):
        rng = np.random.default_rng(42)
        plats = []
        n_total = self.n_nodes
        n_sensor = max(1, n_total * 3 // 10)
        n_weapon = max(1, n_total * 3 // 10)
        n_command = max(1, n_total * 2 // 10)
        n_platform = max(0, n_total - n_sensor - n_weapon - n_command)
        types = (["sensor"] * n_sensor + ["weapon"] * n_weapon +
                 ["command"] * n_command + ["platform"] * n_platform)
        for i, t in enumerate(types):
            if t == "sensor":
                cap = rng.choice(["radar", "ir", "sar", "eoir"], size=2, replace=False).tolist()
                kill_prob = rng.uniform(0.6, 0.95)  # 传感器探测概率高
            elif t == "weapon":
                cap = rng.choice(["missile", "laser", "bomb", "torpedo"], size=2, replace=False).tolist()
                kill_prob = rng.uniform(0.4, 0.95)
            elif t == "command":
                cap = rng.choice(["c2", "datalink", "fusion"], size=2, replace=False).tolist()
                kill_prob = 0.1
            else:
                cap = rng.choice(["transport", "refuel", "relay"], size=2, replace=False).tolist()
                kill_prob = 0.1
            plats.append(Platform(
                platform_id=i, platform_type=t,
                capabilities=cap,
                kill_prob=kill_prob,
                cost_per_shot=rng.uniform(10, 200) if t == "weapon" else 5.0,
                reload_time=rng.uniform(5, 60),
                max_range=rng.uniform(20, 80),
                position=rng.uniform(0, 100, 2),
                ammo=rng.integers(5, 20),
            ))
        return plats

    def _create_default_targets(self):
        rng = np.random.default_rng(43)
        tgts = []
        # 高价值目标: 2-3个, value=0.8-1.0, priority=1
        n_high = rng.integers(2, 4)
        for i in range(n_high):
            tgts.append(Target(
                target_id=i,
                value=rng.uniform(0.8, 1.0),
                is_time_sensitive=rng.random() > 0.3,  # 高价值常为时敏
                time_window=rng.uniform(180, 600),
                position=rng.uniform(20, 80, 2),
                velocity=rng.uniform(-5, 5, 2),
                priority=1,
                required_capability=rng.choice(["missile", "laser"], size=1).tolist(),
            ))
        # 普通目标: 10-15个, value=0.2-0.6, priority=2-3
        n_normal = rng.integers(10, 16)
        for i in range(n_high, n_high + n_normal):
            tgts.append(Target(
                target_id=i,
                value=rng.uniform(0.2, 0.6),
                is_time_sensitive=False,
                time_window=np.inf,
                position=rng.uniform(20, 80, 2),
                velocity=rng.uniform(-3, 3, 2),
                priority=rng.integers(2, 4),
                required_capability=[],
            ))
        return tgts

    def _get_obs(self):
        """构建观测向量"""
        # 节点信息度量
        node_info = np.array([compute_node_info_sum(self.network, i) for i in range(self.n_nodes)])
        # 目标状态: [value, is_time_sensitive, time_window_norm, priority_norm]
        target_feats = []
        for t in self.targets:
            target_feats.extend([
                t.value,
                1.0 if t.is_time_sensitive else 0.0,
                t.time_window / 600.0,
                (4 - t.priority) / 3.0,
            ])
        target_feats = np.array(target_feats, dtype=np.float32)
        # 全局网络质量
        global_quality = np.array([
            float(self.network.C.sum() / (self.n_nodes * self.n_nodes)),
            float(self.network.R.sum() / (self.n_nodes * self.n_nodes)),
            float(self.network.P.sum() / (self.n_nodes * self.n_nodes)),
        ], dtype=np.float32)
        obs = np.concatenate([node_info, target_feats, global_quality])
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-1.0)
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.total_cost = 0.0
        self.kills = 0
        self.episode_rewards = []
        # 重置平台状态 (弹药恢复到初始值)
        for p in self.platforms:
            p.is_available = True
            p.ammo = self._initial_ammo.get(p.platform_id, int(p.ammo))
        # 重置目标摧毁状态
        for t in self.targets:
            t.destroyed = False
        return self._get_obs(), {}

    def step(self, action):
        """执行一步: 用选中的武器打击当前最高优先级可用目标

        action: int → 武器平台在 weapon_indices 中的索引
        环境自动选择: 与该武器连通的最佳传感器 + 最佳指控节点
        """
        self.current_step += 1

        # 获取实际武器平台ID
        w_idx = self.weapon_indices[action % len(self.weapon_indices)]
        weapon = self.platforms[w_idx]

        # 选择最高优先级且可打击的目标
        visible = []
        for t in self.targets:
            if self.count_mode == "destruction" and t.destroyed:
                continue
            dist = np.linalg.norm(weapon.position - t.position)
            if dist <= weapon.max_range:
                visible.append((t.priority, -t.value, t.target_id, t))
        if not visible:
            # 无可打击目标 → 小惩罚
            obs = self._get_obs()
            return obs, 0.0, self.current_step >= self.max_steps, False, {
                "kill_score": 0, "cost": 0, "engagement_time": 0,
                "info_score": 0, "network_score": 0,
                "total_kills": self.kills, "total_cost": self.total_cost,
            }
        target = sorted(visible)[0][3]  # 按(priority, -value, id)排序取最高

        if not weapon.is_available or weapon.ammo <= 0:
            # 自动切换到有弹药的可用武器 (fallback)
            available = [(i, p) for i, p in enumerate(self.platforms)
                         if p.platform_type == "weapon" and p.is_available and p.ammo > 0]
            if available:
                w_idx = available[0][0]
                weapon = self.platforms[w_idx]
            else:
                # 真的没武器了
                obs = self._get_obs()
                return obs, 0.0, self.current_step >= self.max_steps, False, {
                    "kill_score": 0, "cost": 0, "engagement_time": 0,
                    "info_score": 0, "network_score": 0,
                    "total_kills": self.kills, "total_cost": self.total_cost,
                }

        # 自动选择最佳传感器: 与武器连通 + 信息质量最高 + 探测能力强
        candidates_s = [(compute_node_info_sum(self.network, i), i)
                        for i, p in enumerate(self.platforms)
                        if p.platform_type == "sensor" and self.network.C[i, w_idx] > 0]
        if not candidates_s:
            candidates_s = [(compute_node_info_sum(self.network, i), i)
                            for i, p in enumerate(self.platforms) if p.platform_type == "sensor"]
        if not candidates_s:
            candidates_s = [(0, 0)]
        s_idx = max(candidates_s)[1]
        sensor = self.platforms[s_idx]

        # 自动选择最佳指控: 与武器和传感器连通 + 指挥质量最高
        candidates_c = [(compute_info_measure(self.network, i, "command_control"), i)
                        for i, p in enumerate(self.platforms)
                        if p.platform_type == "command"
                        and self.network.C[i, w_idx] > 0]
        if not candidates_c:
            candidates_c = [(compute_info_measure(self.network, i, "command_control"), i)
                            for i, p in enumerate(self.platforms) if p.platform_type == "command"]
        if not candidates_c:
            candidates_c = [(0, 0)]
        c_idx = max(candidates_c)[1]
        cmd = self.platforms[c_idx]

        # ── 杀伤计算 ──
        detect_prob = sensor.kill_prob if any(c in sensor.capabilities for c in ["radar", "ir", "sar", "eoir"]) else 0.7
        effective_kill = detect_prob * weapon.kill_prob
        kill_score = effective_kill * target.value

        # 杀伤惩罚: 有效杀伤概率偏低时打折
        if effective_kill < self.reward_params.get("min_kill_prob", 0.2):
            kill_score *= 0.7

        # 距离惩罚
        dist = np.linalg.norm(weapon.position - target.position)
        engagement_time = dist / 15.0 + weapon.reload_time

        # 时敏窗口检查（惩罚从 0.2 放宽到 0.6，避免高价值时敏目标完全无法计入）
        if target.is_time_sensitive and engagement_time > target.time_window:
            kill_score *= 0.6

        # 成本
        cost = weapon.cost_per_shot

        # 信息度量
        info_score = (compute_node_info_sum(self.network, s_idx) +
                      compute_node_info_sum(self.network, c_idx)) * 0.5

        # 网络连通质量
        net_qual = (float(self.network.C[s_idx, w_idx]) +
                    float(self.network.C[c_idx, w_idx]) +
                    float(self.network.C[s_idx, c_idx])) / 3.0

        # 消耗弹药
        weapon.ammo -= 1
        if weapon.ammo <= 0:
            weapon.is_available = False

        # ── 奖励计算 ──
        rp = self.reward_params
        # 高价值目标额外加成
        is_high_value = target.value > 0.7 or target.priority == 1
        value_mult = 1.5 if is_high_value else 1.0

        base_reward = 1.0
        kill_reward = rp["alpha"] * kill_score * 3.0 * value_mult  # 高价值x1.5
        cost_reward = rp["beta"] * (1.0 - cost / rp.get("max_cost", 1000.0))
        time_reward = rp["gamma"] * (1.0 - engagement_time / rp.get("time_window", 300.0))
        info_reward = rp["delta"] * info_score
        net_reward = rp["epsilon"] * net_qual

        reward = base_reward + kill_reward + cost_reward + time_reward + info_reward + net_reward

        self.total_cost += cost
        # 区分高价值和普通杀伤（阈值统一为 0.15，避免高价值目标因效能波动被漏统计）
        is_high_value_kill = False
        if kill_score > 0.15 and is_high_value:
            if self.count_mode == "destruction" and not target.destroyed:
                target.destroyed = True
            self.kills += 1  # 高价值杀伤
            is_high_value_kill = True
        elif kill_score > 0.15:
            if self.count_mode == "destruction" and not target.destroyed:
                target.destroyed = True
            self.kills += 1  # 普通杀伤

        done = self.current_step >= self.max_steps
        info = {
            "kill_score": kill_score,
            "cost": cost,
            "engagement_time": engagement_time,
            "info_score": info_score,
            "network_score": net_qual,
            "total_kills": self.kills,
            "total_cost": self.total_cost,
            "is_high_value_kill": is_high_value_kill,
        }

        return self._get_obs(), reward, done, False, info

    def render(self, mode="human"):
        if mode == "human":
            print(f"Step {self.current_step}: kills={self.kills}, "
                  f"cost={self.total_cost:.1f}, reward={self.episode_rewards[-1]:.3f}")


# ============================================================
# 6. F2T2EA 杀伤链执行效果评估 (论文2)
# ============================================================

def evaluate_kill_chain_effectiveness(network: KillChainNetwork,
                                      assignment: np.ndarray,
                                      platforms: list, targets: list) -> dict:
    """评估一条杀伤链的F2T2EA各阶段效果

    assignment: (3,) 数组 [sensor_idx, weapon_idx, cmd_idx]
    返回: {find, fix, track, target, engage, assess} 评分
    """
    s_idx, w_idx, c_idx = assignment.astype(int)
    sensor = platforms[s_idx % len(platforms)]
    weapon = platforms[w_idx % len(platforms)]
    cmd = platforms[c_idx % len(platforms)]

    # Find: 传感器探测能力
    find_score = compute_info_measure(network, s_idx, "intel_distribution")
    # Fix: 目标定位精度
    fix_score = compute_info_measure(network, s_idx, "situation_sharing")
    # Track: 持续跟踪
    track_score = 0.5 * compute_info_measure(network, s_idx, "situation_sharing") + \
                  0.5 * compute_info_measure(network, c_idx, "command_control")
    # Target: 瞄准决策
    target_score = compute_info_measure(network, c_idx, "command_control")
    # Engage: 武器打击
    engage_score = weapon.kill_prob * float(network.C[w_idx, s_idx] > 0)
    # Assess: 评估
    assess_score = compute_info_measure(network, c_idx, "tactical_coordination")

    return {
        "find": find_score,
        "fix": fix_score,
        "track": track_score,
        "target": target_score,
        "engage": engage_score,
        "assess": assess_score,
        "overall": np.mean([find_score, fix_score, track_score, target_score, engage_score, assess_score]),
    }
