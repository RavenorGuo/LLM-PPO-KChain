"""
Streamlit 前端 — 多Agent博弈 + PPO 军事决策优化系统

提供:
  1. 用户需求输入界面
  2. 多Agent博弈过程可视化
  3. PPO训练实时曲线
  4. 杀伤效果 & 成本分析图表
  5. 详细结果导出
"""
import sys
import os
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 强制重载 model 模块，避免 Streamlit 缓存旧版本
import importlib
import model
importlib.reload(model)

from model import (
    KillChainNetwork, KillChainEnv, Platform, Target,
    RequirementParser, RequirementVector,
    compute_node_info_sum, evaluate_kill_chain_effectiveness,
)
from agent import GameEngine
from ppo import PPOTrainer, ActorCritic, inference
from llm_parser import LLMRequirementParser, LLMConfig

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(
    page_title="军事决策优化系统",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 环境配置读取
# ============================================================

def _load_llm_env() -> dict:
    """从项目根目录 .env 文件读取 LLM 配置，同时兼容系统环境变量"""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    env = {}
    # 1. 先读 .env 文件（如果存在）
    if os.path.isfile(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                env[key.strip()] = val.strip()
    # 2. 系统环境变量优先级更高，覆盖 .env
    for key in ["LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"]:
        val = os.getenv(key)
        if val is not None:
            env[key] = val
    return {
        "base_url": env.get("LLM_BASE_URL", "https://api.deepseek.com/v1"),
        "api_key": env.get("LLM_API_KEY", ""),
        "model": env.get("LLM_MODEL", "deepseek-chat"),
    }


# ============================================================
# 缓存函数
# ============================================================

def run_game_engine(requirement: str, max_rounds: int, n_nodes: int, n_targets: int,
                    llm_config=None) -> dict:
    """运行多Agent博弈"""
    engine = GameEngine(max_rounds=max_rounds)
    # 将 LLMConfig dataclass 转为 dict，供 agent.py 使用
    config_dict = None
    if llm_config is not None:
        if hasattr(llm_config, "api_key"):
            config_dict = {
                "base_url": getattr(llm_config, "base_url", "https://api.deepseek.com/v1"),
                "api_key": getattr(llm_config, "api_key", ""),
                "model": getattr(llm_config, "model", "deepseek-chat"),
            }
        elif isinstance(llm_config, dict):
            config_dict = llm_config
    result = engine.run(requirement, n_nodes, n_targets, llm_config=config_dict)
    return result


def build_targets(n_targets: int, n_high: int = None, seed: int = 43):
    """构建目标列表: 高价值 + 普通"""
    rng = np.random.default_rng(seed)
    targets = []
    n_high = max(1, min(n_high or max(2, n_targets // 6), n_targets))
    for i in range(n_high):
        targets.append(Target(
            target_id=i, value=rng.uniform(0.8, 1.0),
            is_time_sensitive=rng.random() > 0.3,
            time_window=rng.uniform(180, 600),
            position=rng.uniform(20, 80, 2), velocity=rng.uniform(-5, 5, 2),
            priority=1,
        ))
    for i in range(n_high, n_targets):
        targets.append(Target(
            target_id=i, value=rng.uniform(0.2, 0.6),
            is_time_sensitive=False, time_window=np.inf,
            position=rng.uniform(20, 80, 2), velocity=rng.uniform(-3, 3, 2),
            priority=rng.integers(2, 4),
        ))
    return targets


def run_ppo_training(reward_params: dict, n_nodes: int, n_targets: int, n_high_targets: int,
                     episodes: int, progress_bar, status_text,
                     count_mode: str = "delivery") -> dict:
    """运行PPO训练 (带进度回调)"""
    network = KillChainNetwork.create_default(n_nodes)
    targets = build_targets(n_targets, n_high=n_high_targets)
    env = KillChainEnv(network=network, targets=targets, reward_params=reward_params,
                       max_steps=50, count_mode=count_mode)

    n_actions = env.action_space.n
    trainer = PPOTrainer(
        reward_params, obs_dim=env.obs_dim, act_dim=n_actions,
        hidden_dim=128, lr=3e-4, gamma=0.99, ppo_epochs=10, verbose=False,
    )

    episode_rewards = []
    best_reward = -float("inf")

    for ep in range(1, episodes + 1):
        ep_reward, ep_len = trainer.collect_rollout(env, max_steps=50)
        avg_loss = trainer.update()
        episode_rewards.append(ep_reward)
        trainer.episode_rewards.append(ep_reward)
        trainer.episode_lengths.append(ep_len)
        if ep_reward > best_reward:
            best_reward = ep_reward

        if ep % 50 == 0 or ep == episodes:
            progress_bar.progress(ep / episodes)
            recent = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
            status_text.text(f"Ep {ep}/{episodes} | Avg Reward: {recent:.2f} | Best: {best_reward:.2f}")

    # 推理评估
    env_test = KillChainEnv(network=network, targets=targets, reward_params=reward_params,
                            max_steps=100, count_mode=count_mode)
    infer_result = inference(trainer.policy, env_test, n_steps=100, deterministic=True)

    # 用deterministic策略收集评估数据用于可视化
    eval_data = run_evaluation_rollout(network, reward_params, trainer.policy, targets,
                                       n_steps=100, count_mode=count_mode)

    return {
        "policy": trainer.policy,
        "trainer": trainer,
        "episode_rewards": episode_rewards,
        "best_reward": best_reward,
        "inference": infer_result,
        "eval_data": eval_data,
        "count_mode": count_mode,
    }


def run_evaluation_rollout(network, reward_params, policy, targets, n_steps=100, count_mode="delivery"):
    """运行评估rollout，收集详细数据用于图表"""
    env = KillChainEnv(network=network, targets=targets, reward_params=reward_params,
                       max_steps=n_steps, count_mode=count_mode)
    device = next(policy.parameters()).device

    obs, _ = env.reset()
    data = {
        "steps": [], "rewards": [], "kill_scores": [], "costs": [],
        "engagement_times": [], "info_scores": [], "network_scores": [],
        "cumulative_kills": [], "cumulative_cost": [],
        "cumulative_high_value_kills": [],
        "high_value_kill_scores": [],
    }

    total_kills = 0
    total_high_kills = 0
    total_cost = 0.0

    for step in range(n_steps):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _, value, action_probs = policy.get_action(obs_tensor, deterministic=True)

        action_scalar = action.item()
        next_obs, reward, done, truncated, info = env.step(action_scalar)

        ks = info.get("kill_score", 0)
        total_kills += int(ks > 0.15)
        total_high_kills += int(info.get("is_high_value_kill", False))
        total_cost += info.get("cost", 0)

        data["steps"].append(step)
        data["rewards"].append(reward)
        data["kill_scores"].append(ks)
        data["high_value_kill_scores"].append(ks if info.get("is_high_value_kill", False) else 0.0)
        data["costs"].append(info.get("cost", 0))
        data["engagement_times"].append(info.get("engagement_time", 0))
        data["info_scores"].append(info.get("info_score", 0))
        data["network_scores"].append(info.get("network_score", 0))
        data["cumulative_kills"].append(total_kills)
        data["cumulative_high_value_kills"].append(total_high_kills)
        data["cumulative_cost"].append(total_cost)

        obs = next_obs
        if done or truncated:
            break

    return data


# ============================================================
# 图表绘制
# ============================================================

def plot_agent_scores(game_result: dict) -> plt.Figure:
    """Agent评分雷达图"""
    eq = game_result.get("nash_equilibrium", {})
    scores = eq.get("scores", {})
    if not scores:
        scores = {"sensor": 0.5, "weapon": 0.5, "command": 0.5, "resource": 0.5}

    labels = list(scores.keys())
    values = list(scores.values())

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    ax.fill(angles, values, alpha=0.25, color="#1f77b4")
    ax.plot(angles, values, "o-", color="#1f77b4", linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["传感器", "武器", "指控", "资源"], fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title("多Agent博弈均衡评分", fontsize=14, fontweight="bold", pad=20)
    return fig


def plot_training_curve(episode_rewards: list) -> plt.Figure:
    """PPO训练曲线"""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(episode_rewards, alpha=0.3, color="#1f77b4", linewidth=0.5, label="Episode Reward")

    if len(episode_rewards) >= 20:
        window = min(50, len(episode_rewards) // 4)
        ma = np.convolve(episode_rewards, np.ones(window)/window, mode="valid")
        ax.plot(range(window-1, len(episode_rewards)), ma, color="#d62728", linewidth=2, label=f"MA({window})")

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Reward", fontsize=11)
    ax.set_title("PPO 训练奖励曲线", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_kill_cost_chart(eval_data: dict) -> plt.Figure:
    """杀伤 & 成本 双轴图 (MA10平滑)"""
    fig, ax1 = plt.subplots(figsize=(8, 4))
    color_kill = "#d62728"
    color_cost = "#1f77b4"
    steps = eval_data["steps"]
    kills = np.array(eval_data["kill_scores"])
    costs = np.array(eval_data["costs"])

    # MA10平滑，前9步保持原值 (conv mode='valid' 丢9个点)
    def ma10(arr):
        if len(arr) <= 10:
            return arr
        ma = np.convolve(arr, np.ones(10)/10, mode="valid")  # len - 9
        return np.concatenate([arr[:9], ma])  # 9 + (len-9) = len ✓

    kills_sm = ma10(kills)
    costs_sm = ma10(costs)
    # 高价值杀伤分数
    hv_kills = np.array(eval_data["high_value_kill_scores"])
    hv_kills_sm = ma10(hv_kills)

    ax1.set_xlabel("Step", fontsize=11)
    ax1.set_ylabel("杀伤分数 (MA10)", color=color_kill, fontsize=11)
    line1, = ax1.plot(steps[:len(kills_sm)], kills_sm,
                       color=color_kill, linewidth=2, label="Kill Score (MA10)")
    line3, = ax1.plot(steps[:len(hv_kills_sm)], hv_kills_sm,
                       color="#9467bd", linewidth=2, linestyle="--",
                       label="High-Value Kill (MA10)")
    ax1.tick_params(axis="y", labelcolor=color_kill)

    ax2 = ax1.twinx()
    ax2.set_ylabel("单步成本 (MA10)", color=color_cost, fontsize=11)
    line2, = ax2.plot(steps[:len(costs_sm)], costs_sm,
                       color=color_cost, linewidth=2, label="Cost (MA10)")
    ax2.tick_params(axis="y", labelcolor=color_cost)

    lines = [line1, line2, line3]
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right")
    ax1.set_title("杀伤效果 & 资源消耗 (MA10 平滑)", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    return fig


def plot_cumulative(eval_data: dict, count_mode: str = "delivery") -> plt.Figure:
    """累计杀伤 (总+高价值) & 累计成本"""
    label_total = "总击毁" if count_mode == "destruction" else "总杀伤"
    label_high = "高价值击毁" if count_mode == "destruction" else "高价值杀伤"
    title = "累计击毁 (总 vs 高价值)" if count_mode == "destruction" else "累计杀伤 (总 vs 高价值)"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # 总杀伤 + 高价值杀伤
    ax1.plot(eval_data["steps"], eval_data["cumulative_kills"],
             color="#2ca02c", linewidth=2, label=label_total)
    ax1.plot(eval_data["steps"], eval_data["cumulative_high_value_kills"],
             color="#d62728", linewidth=2, label=label_high)
    ax1.fill_between(eval_data["steps"], eval_data["cumulative_high_value_kills"],
                     alpha=0.3, color="#d62728")
    ax1.set_xlabel("Step", fontsize=11)
    ax1.set_ylabel("Cumulative Kills", fontsize=11)
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(eval_data["steps"], eval_data["cumulative_cost"],
                     alpha=0.4, color="#ff7f0e")
    ax2.plot(eval_data["steps"], eval_data["cumulative_cost"],
             color="#ff7f0e", linewidth=2)
    ax2.set_xlabel("Step", fontsize=11)
    ax2.set_ylabel("Cumulative Cost", fontsize=11)
    ax2.set_title("累计资源消耗", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_info_network(eval_data: dict) -> plt.Figure:
    """信息优势 & 网络质量 趋势"""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(eval_data["steps"], eval_data["info_scores"],
            color="#9467bd", alpha=0.7, linewidth=1.5, label="信息优势")
    ax.plot(eval_data["steps"], eval_data["network_scores"],
            color="#8c564b", alpha=0.7, linewidth=1.5, label="网络质量")
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("信息优势 & 网络质量 变化", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_reward_params(reward_params: dict) -> plt.Figure:
    """奖励参数柱状图"""
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["α (杀伤)", "β (成本)", "γ (时间)", "δ (信息)", "ε (网络)"]
    keys = ["alpha", "beta", "gamma", "delta", "epsilon"]
    values = [reward_params.get(k, 0) for k in keys]
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#1f77b4"]

    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.2)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylabel("Weight Value", fontsize=11)
    ax.set_title("PPO 定制奖励参数", fontsize=14, fontweight="bold")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.grid(True, alpha=0.2, axis="y")
    return fig


# ============================================================
# UI 组件
# ============================================================

def sidebar_config():
    """侧边栏配置"""
    with st.sidebar:
        st.title("⚙️ 系统配置")

        n_nodes = st.slider("网络节点数", 20, 200, 50, 10,
                            help="分布式杀伤链网络中的作战节点数量",
                            key="config_n_nodes")
        n_targets = st.slider("目标数量", 3, 30, 8, 1,
                              help="需要打击的敌方目标数量",
                              key="config_n_targets")
        n_high_targets = st.slider("高价值目标数", 1, n_targets, min(2, n_targets), 1,
                                   help="高价值目标数量（优先级=1，价值0.8-1.0）",
                                   key="config_n_high_targets")
        count_mode_label = st.radio("杀伤统计模式", ["投送模式", "击毁模式"], index=0,
                                    help="投送模式=统计成功打击次数；击毁模式=统计被摧毁的独特目标数",
                                    key="config_count_mode")
        max_rounds = st.slider("博弈最大轮数", 5, 50, 12, 1,
                               help="多Agent合同网协议谈判的最大轮数",
                               key="config_max_rounds")
        episodes = st.slider("PPO训练回合", 100, 2000, 400, 100,
                             help="PPO强化学习的训练episode数",
                             key="config_episodes")

        st.divider()
        st.subheader("🤖 需求解析配置")
        parser_mode = st.radio("解析方式", ["规则匹配", "LLM大模型"], index=0,
                               help="规则匹配=关键词计数；LLM=大语言模型语义理解",
                               key="config_parser_mode")

        # 从 .env 文件读取 LLM 配置（无需在 Web 输入敏感信息）
        llm_env = _load_llm_env()
        if parser_mode == "LLM大模型":
            if llm_env.get("api_key"):
                st.success(f"✅ LLM 已配置 ({llm_env.get('model', 'deepseek-chat')})")
                st.caption(f"Base: {llm_env.get('base_url', 'https://api.deepseek.com/v1')}")
            else:
                st.warning("⚠️ LLM API Key 未配置")
                st.caption("请修改项目根目录 `.env` 文件，填入 LLM_API_KEY 后重启 Streamlit")

        st.divider()
        st.subheader("📋 预置需求模板")
        templates = {
            "杀伤优先": "最大化杀伤高价值目标，控制弹药成本在500万以内",
            "时敏优先": "优先打击时敏目标，要求3分钟内完成杀伤链闭环，不惜成本",
            "信息优势": "注重信息优势和网络质量，摧毁敌方指挥节点和传感器",
            "经济作战": "降低资源消耗，用最低成本消耗敌方力量，保证生存率",
            "速战速决": "快速响应，最大化时间效率，优先处理高威胁时敏目标",
        }
        for name, req in templates.items():
            if st.button(f"📌 {name}", use_container_width=True):
                st.session_state.requirement = req
                st.rerun()

        st.divider()
        if st.button("🔄 清空缓存", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
        st.caption("基于分布式杀伤链数学模型 G=(V,C,R,P)")
        st.caption("西安交通大学 | 论文参考实现")

    llm_env = _load_llm_env()
    return {
        "n_nodes": n_nodes,
        "n_targets": n_targets,
        "n_high_targets": n_high_targets,
        "max_rounds": max_rounds,
        "episodes": episodes,
        "count_mode": "destruction" if count_mode_label == "击毁模式" else "delivery",
        "parser_mode": parser_mode,
        "llm_config": llm_env if parser_mode == "LLM大模型" and llm_env.get("api_key") else None,
    }


def display_game_process(game_result: dict):
    """展示博弈过程"""
    with st.expander("🔄 多Agent博弈过程详情", expanded=False):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("收敛状态", "✅ 已收敛" if game_result["converged"] else "⚠️ 未收敛")
            st.metric("博弈轮数", game_result["rounds"])
            eq = game_result.get("nash_equilibrium", {})
            st.metric("Nash乘积", f"{eq.get('nash_product', 0):.4f}")
        with col2:
            st.subheader("Agent均衡评分")
            scores = eq.get("scores", {})
            for agent, score in scores.items():
                st.progress(float(score), text=f"{agent}: {float(score):.3f}")

        st.subheader("博弈消息流")
        messages = game_result.get("messages", [])
        for msg in messages[-20:]:  # 最近20条
            if "[Init]" in msg or "[Coordinator]" in msg:
                st.info(msg)
            elif "[Convergence]" in msg or "[Complete]" in msg:
                st.success(msg)
            elif "[RewardParams]" in msg:
                st.warning(msg)
            else:
                st.text(f"  {msg}")


def display_reward_params(reward_params: dict):
    """展示奖励参数"""
    with st.expander("🎁 PPO奖励参数详情", expanded=True):
        cols = st.columns(5)
        metrics = [
            ("α 杀伤效能", "alpha", "🔴"),
            ("β 成本控制", "beta", "🟠"),
            ("γ 时间效率", "gamma", "🟢"),
            ("δ 信息优势", "delta", "🟣"),
            ("ε 网络质量", "epsilon", "🔵"),
        ]
        for col, (label, key, icon) in zip(cols, metrics):
            col.metric(f"{icon} {label}", f"{reward_params.get(key, 0):.4f}")

        col1, col2, col3 = st.columns(3)
        col1.metric("最低杀伤概率", f"{reward_params.get('min_kill_prob', 0.5):.2f}")
        col2.metric("最大成本预算", f"{reward_params.get('max_cost', 1000):.0f}")
        col3.metric("时间窗口(秒)", f"{reward_params.get('time_window', 300):.0f}")

        fig = plot_reward_params(reward_params)
        st.pyplot(fig)


def display_ppo_results(ppo_result: dict):
    """展示PPO训练结果"""
    count_mode = ppo_result.get("count_mode", "delivery")
    st.subheader("📈 PPO 训练结果")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("最佳奖励", f"{ppo_result['best_reward']:.2f}")
    col2.metric("最终平均奖励",
                f"{np.mean(ppo_result['episode_rewards'][-50:]):.2f}" if len(ppo_result['episode_rewards']) >= 50
                else f"{np.mean(ppo_result['episode_rewards']):.2f}")
    col3.metric("推理总奖励", f"{ppo_result['inference']['total_reward']:.2f}")
    col4.metric("使用平台数", len(ppo_result['inference'].get('platform_usage', {})))

    # 训练曲线
    fig1 = plot_training_curve(ppo_result["episode_rewards"])
    st.pyplot(fig1)

    # 详细数据
    eval_data = ppo_result["eval_data"]

    # 杀伤 & 成本
    fig2 = plot_kill_cost_chart(eval_data)
    st.pyplot(fig2)

    # 累计统计
    fig3 = plot_cumulative(eval_data, count_mode)
    st.pyplot(fig3)

    # 信息 & 网络
    fig4 = plot_info_network(eval_data)
    st.pyplot(fig4)

    # 汇总统计
    st.subheader("📊 评估汇总")
    label_total = "🎯 总击毁" if count_mode == "destruction" else "🎯 总杀伤"
    label_high = "💎 高价值击毁" if count_mode == "destruction" else "💎 高价值杀伤"
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(label_total, f"{eval_data['cumulative_kills'][-1]}")
    c2.metric(label_high, f"{eval_data['cumulative_high_value_kills'][-1]}")
    c3.metric("💰 总消耗", f"{eval_data['cumulative_cost'][-1]:.0f}")
    c4.metric("⚔️ 平均杀伤/步", f"{np.mean(eval_data['kill_scores']):.3f}")
    c5.metric("💵 平均成本/步", f"{np.mean(eval_data['costs']):.0f}")


def display_export_section(full_result: dict):
    """导出结果"""
    with st.expander("📦 导出完整结果 (JSON)", expanded=False):
        export_data = {
            "requirement": full_result.get("requirement", ""),
            "req_vector": full_result.get("req_vector", {}),
            "reward_params": full_result.get("reward_params", {}),
            "nash_equilibrium": full_result.get("game_result", {}).get("nash_equilibrium", {}),
            "ppo_best_reward": full_result.get("ppo_training", {}).get("best_reward", 0),
            "inference_total_reward": full_result.get("inference", {}).get("total_reward", 0),
        }
        st.json(export_data)
        st.download_button(
            "📥 下载 JSON",
            json.dumps(export_data, indent=2, ensure_ascii=False, default=str),
            "military_decision_result.json",
            "application/json",
        )


# ============================================================
# 主页面
# ============================================================

def main():
    st.title("🎯 多Agent博弈 + PPO 军事决策优化系统")
    st.markdown("""
    > 基于 **合同网协议** 的多Agent博弈协商 → 定制化 **PPO强化学习** 奖励函数 → 分布式杀伤链最优资源调度
    >
    > 参考论文: 分布式动态规划杀伤链生成 | 即时杀伤链构建 | 网络与信息聚优数学模型
    """)

    cfg = sidebar_config()

    # 主输入区
    st.subheader("📝 输入军事需求")
    default_req = st.session_state.get("requirement", "最大化杀伤高价值目标，控制弹药成本，优先时敏目标")
    requirement = st.text_area(
        "请描述您的作战需求（自然语言）:",
        value=default_req,
        height=80,
        placeholder="例如: 最大化杀伤高价值目标，降低资源消耗，3分钟内完成打击闭环...",
        help="系统将自动解析关键词并转化为优化权重",
    )

    # 运行按钮
    run_clicked = st.button("🚀 开始优化决策", type="primary", use_container_width=True)

    if not run_clicked:
        st.info("👆 输入作战需求后，点击「开始优化决策」运行完整流程:")
        st.markdown("""
        1. **需求解析** → 自然语言 → 5维权重向量
        2. **多Agent博弈** → 合同网协议招标-投标 → Nash均衡
        3. **奖励参数生成** → 均衡方案 → PPO定制奖励 {α,β,γ,δ,ε}
        4. **PPO训练** → 定制奖励强化学习 → 最优调度策略
        5. **结果评估** → 杀伤效果 & 成本分析 & 策略可视化
        """)
        return

    # ── 运行完整流程 ──
    with st.spinner("正在运行..."):
        # 显示当前生效的配置
        st.info(f"⚙️ 当前配置: 节点={cfg['n_nodes']} | 目标={cfg['n_targets']} | 博弈轮={cfg['max_rounds']} | PPO回合={cfg['episodes']}")

        # Phase 1: 需求解析
        st.markdown("### Phase 1: 需求解析")
        if cfg["parser_mode"] == "LLM大模型":
            with st.spinner("LLM 解析中..."):
                try:
                    req_vector = LLMRequirementParser.parse(requirement, cfg["llm_config"])
                    st.success("✅ LLM 解析成功")
                except Exception as e:
                    st.warning(f"LLM 解析失败: {e}，已自动回退到规则匹配")
                    req_vector = RequirementParser.parse(requirement)
        else:
            req_vector = RequirementParser.parse(requirement)
            st.info("使用规则匹配解析")
        cols = st.columns(5)
        cols[0].metric("🔴 杀伤效能", f"{req_vector.w_kill:.3f}")
        cols[1].metric("🟠 成本意识", f"{req_vector.w_cost:.3f}")
        cols[2].metric("🟢 时间效率", f"{req_vector.w_time:.3f}")
        cols[3].metric("🟣 信息优势", f"{req_vector.w_info:.3f}")
        cols[4].metric("🔵 网络质量", f"{req_vector.w_network:.3f}")
        st.info(f"目标优先级: **{req_vector.target_priority}** | 约束: {req_vector.constraints}")

        # Phase 2: 多Agent博弈
        st.markdown("### Phase 2: 多Agent博弈 (LangGraph + 合同网协议)")
        progress_game = st.progress(0, "博弈进行中...")
        game_result = run_game_engine(requirement, cfg["max_rounds"],
                                      cfg["n_nodes"], cfg["n_targets"],
                                      llm_config=cfg.get("llm_config"))
        progress_game.progress(100, "博弈完成 ✅")
        display_game_process(game_result)

        # Phase 3: 奖励参数
        st.markdown("### Phase 3: PPO 定制奖励参数")
        reward_params = game_result["reward_params"]
        display_reward_params(reward_params)

        # Phase 4 & 5: PPO训练
        st.markdown("### Phase 4 & 5: PPO 训练 (定制奖励)")
        progress_ppo = st.progress(0, "PPO训练中...")
        status_ppo = st.empty()
        ppo_result = run_ppo_training(reward_params, cfg["n_nodes"], cfg["n_targets"],
                                      cfg["n_high_targets"],
                                      cfg["episodes"], progress_ppo, status_ppo,
                                      cfg["count_mode"])
        status_ppo.text(f"训练完成! Best Reward: {ppo_result['best_reward']:.2f}")

        # Phase 6: 结果评估
        st.markdown("### Phase 6: 综合评估 & 可视化")
        display_ppo_results(ppo_result)

        # 导出
        full_result = {
            "requirement": requirement,
            "req_vector": {
                "w_kill": req_vector.w_kill, "w_cost": req_vector.w_cost,
                "w_time": req_vector.w_time, "w_info": req_vector.w_info,
                "w_network": req_vector.w_network,
                "target_priority": req_vector.target_priority,
            },
            "reward_params": reward_params,
            "game_result": game_result,
            "ppo_training": {
                "best_reward": ppo_result["best_reward"],
                "final_avg_reward": float(np.mean(ppo_result["episode_rewards"][-50:])),
            },
            "inference": {
                "total_reward": ppo_result["inference"]["total_reward"],
                "steps": len(ppo_result["eval_data"]["steps"]),
                "total_kills": ppo_result["eval_data"]["cumulative_kills"][-1],
                "total_cost": ppo_result["eval_data"]["cumulative_cost"][-1],
            },
        }
        display_export_section(full_result)

        st.success("✅ 全流程完成!")
        st.balloons()


if __name__ == "__main__":
    main()
