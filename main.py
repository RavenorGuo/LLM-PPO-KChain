"""
主入口 — 多Agent博弈 + PPO 军事决策优化系统

端到端流程:
  1. 用户输入军事需求 (自然语言)
  2. model.py::RequirementParser 解析为权重向量
  3. agent.py::GameEngine LangGraph多Agent博弈 → Nash均衡
  4. agent.py::RewardGenerator 输出 PPO奖励参数
  5. model.py::KillChainEnv 构建仿真环境 (G=(V,C,R,P) + F2T2EA)
  6. ppo.py::PPOTrainer 定制奖励训练 PPO
  7. 输出最优资源调度策略 + 评估报告

用法:
    python main.py
    python main.py --requirement "最大化高价值杀伤，控制成本"
    python main.py --interactive
"""
import argparse
import json
import sys
import numpy as np

# 设置UTF-8编码
sys.stdout.reconfigure(encoding='utf-8')

from model import (
    KillChainNetwork, KillChainEnv, RequirementParser, RequirementVector,
    Target, Platform, compute_info_measure, compute_node_info_sum,
    evaluate_kill_chain_effectiveness,
)
from agent import GameEngine
from ppo import PPOTrainer, ActorCritic, inference


# ============================================================
# 系统主流程
# ============================================================

class MilitaryDecisionSystem:
    """军事决策优化系统主类

    封装完整的 需求→博弈→PPO 管道
    """

    def __init__(self, n_nodes: int = 50, n_targets: int = 8,
                 ppo_episodes: int = 500, max_rounds: int = 20,
                 verbose: bool = True):
        self.n_nodes = n_nodes
        self.n_targets = n_targets
        self.ppo_episodes = ppo_episodes
        self.max_rounds = max_rounds
        self.verbose = verbose

        # 组件
        self.network: KillChainNetwork = None
        self.env: KillChainEnv = None
        self.req_vector: RequirementVector = None
        self.reward_params: dict = None
        self.game_result: dict = None
        self.policy: ActorCritic = None
        self.trainer: PPOTrainer = None

    def run(self, requirement: str) -> dict:
        """执行完整流程"""
        print("\n" + "=" * 70)
        print(f"  多Agent博弈 + PPO 军事决策优化系统")
        print(f"  需求: {requirement}")
        print("=" * 70)

        # ── Phase 1: 需求解析 ──
        self._phase1_parse(requirement)

        # ── Phase 2: 多Agent博弈 ──
        self._phase2_game()

        # ── Phase 3: 生成奖励参数 ──
        self._phase3_reward()

        # ── Phase 4: 构建仿真环境 ──
        self._phase4_env()

        # ── Phase 5: PPO训练 ──
        self._phase5_ppo()

        # ── Phase 6: 推理评估 ──
        result = self._phase6_evaluate()

        return result

    def _phase1_parse(self, requirement: str):
        """Phase 1: 需求解析"""
        if self.verbose:
            print(f"\n{'─' * 50}")
            print("Phase 1: 需求解析")
            print(f"{'─' * 50}")

        self.req_vector = RequirementParser.parse(requirement)
        if self.verbose:
            print(f"  杀伤效能权重: {self.req_vector.w_kill:.3f}")
            print(f"  成本意识权重: {self.req_vector.w_cost:.3f}")
            print(f"  时间效率权重: {self.req_vector.w_time:.3f}")
            print(f"  信息优势权重: {self.req_vector.w_info:.3f}")
            print(f"  网络质量权重: {self.req_vector.w_network:.3f}")
            print(f"  目标优先级:   {self.req_vector.target_priority}")
            print(f"  约束条件:     {self.req_vector.constraints}")

    def _phase2_game(self):
        """Phase 2: LangGraph 多Agent博弈"""
        if self.verbose:
            print(f"\n{'─' * 50}")
            print("Phase 2: LangGraph 多Agent博弈 (改进合同网协议)")
            print(f"{'─' * 50}")

        engine = GameEngine(max_rounds=self.max_rounds)
        self.game_result = engine.run(
            self._req_to_text(), self.n_nodes, self.n_targets
        )

        if self.verbose:
            print(f"  收敛状态: {self.game_result['converged']}")
            print(f"  博弈轮数: {self.game_result['rounds']}")
            eq = self.game_result.get("nash_equilibrium", {})
            print(f"  Nash乘积: {eq.get('nash_product', 0):.4f}")
            scores = eq.get("scores", {})
            print(f"  Agent评分: {', '.join(f'{k}={v:.3f}' for k, v in scores.items())}")
            # 打印关键消息
            for msg in self.game_result.get("messages", [])[-8:]:
                print(f"  {msg}")

    def _phase3_reward(self):
        """Phase 3: 生成PPO奖励参数"""
        if self.verbose:
            print(f"\n{'─' * 50}")
            print("Phase 3: 生成PPO奖励参数")
            print(f"{'─' * 50}")

        self.reward_params = self.game_result["reward_params"]
        if self.verbose:
            print(f"  α (杀伤效能):  {self.reward_params['alpha']:.4f}")
            print(f"  β (成本控制):  {self.reward_params['beta']:.4f}")
            print(f"  γ (时间效率):  {self.reward_params['gamma']:.4f}")
            print(f"  δ (信息优势):  {self.reward_params['delta']:.4f}")
            print(f"  ε (网络质量):  {self.reward_params['epsilon']:.4f}")
            print(f"  最低杀伤概率:  {self.reward_params.get('min_kill_prob', 0.5)}")
            print(f"  最大成本:      {self.reward_params.get('max_cost', 1000)}")
            print(f"  时间窗口:      {self.reward_params.get('time_window', 300)}秒")

    def _phase4_env(self):
        """Phase 4: 构建仿真环境"""
        if self.verbose:
            print(f"\n{'─' * 50}")
            print("Phase 4: 构建仿真环境")
            print(f"{'─' * 50}")

        self.network = KillChainNetwork.create_default(self.n_nodes)
        env_targets = self._build_targets()
        self.env = KillChainEnv(
            network=self.network,
            targets=env_targets,
            reward_params=self.reward_params,
            max_steps=50,
        )
        if self.verbose:
            print(f"  网络节点数: {self.n_nodes}")
            print(f"  目标数:     {len(env_targets)}")
            print(f"  平台数:     {len(self.env.platforms)}")
            print(f"  观测维度:   {self.env.obs_dim}")
            print(f"  动作维度:   {self.env.action_space.n} (可选武器数)")

    def _phase5_ppo(self):
        """Phase 5: PPO训练"""
        if self.verbose:
            print(f"\n{'─' * 50}")
            print("Phase 5: PPO训练 (定制奖励)")
            print(f"{'─' * 50}")

        n_actions = self.env.action_space.n
        self.trainer = PPOTrainer(
            self.reward_params,
            obs_dim=self.env.obs_dim,
            act_dim=n_actions,
            hidden_dim=256,
            lr=3e-4,
            gamma=0.99,
            ppo_epochs=10,
            verbose=self.verbose,
        )
        self.policy = self.trainer.train(
            self.env, episodes=self.ppo_episodes, log_interval=100, collect_steps=50
        )

    def _phase6_evaluate(self) -> dict:
        """Phase 6: 推理评估"""
        if self.verbose:
            print(f"\n{'─' * 50}")
            print("Phase 6: 推理评估")
            print(f"{'─' * 50}")

        env_test = KillChainEnv(
            network=self.network,
            targets=self.env.targets,
            platforms=self.env.platforms,
            reward_params=self.reward_params,
            max_steps=30,
        )
        result = inference(self.policy, env_test, n_steps=30, deterministic=True)

        if self.verbose:
            print(f"  总奖励:      {result['total_reward']:.2f}")
            print(f"  平均杀伤分数: {np.mean(result['kill_scores']):.3f}")
            print(f"  平均成本:     {np.mean(result['costs']):.1f}")
            print(f"  使用平台数:   {len(result['platform_usage'])}")
            print(f"  执行步数:     {len(result['rewards'])}")

        # 构建完整输出
        final_output = {
            "requirement": self._req_to_text(),
            "req_vector": {
                "w_kill": self.req_vector.w_kill,
                "w_cost": self.req_vector.w_cost,
                "w_time": self.req_vector.w_time,
                "w_info": self.req_vector.w_info,
                "w_network": self.req_vector.w_network,
                "target_priority": self.req_vector.target_priority,
                "constraints": self.req_vector.constraints,
            },
            "game_result": {
                "converged": self.game_result["converged"],
                "rounds": self.game_result["rounds"],
                "nash_equilibrium": self.game_result["nash_equilibrium"],
            },
            "reward_params": self.reward_params,
            "ppo_training": {
                "episodes": self.ppo_episodes,
                "final_avg_reward": float(np.mean(self.trainer.episode_rewards[-100:])),
                "best_reward": float(np.max(self.trainer.episode_rewards)),
                "final_loss": float(np.mean(self.trainer.loss_history[-100:])),
            },
            "inference": {
                "total_reward": result["total_reward"],
                "avg_kill_score": float(np.mean(result["kill_scores"])),
                "avg_cost": float(np.mean(result["costs"])),
                "platforms_used": len(result["platform_usage"]),
                "steps": len(result["rewards"]),
            },
        }

        # 保存结果
        output_path = "output/reward_params.json"
        import os
        os.makedirs("output", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n  完整结果已保存至 {output_path}")

        return final_output

    def _req_to_text(self) -> str:
        """从RequirementVector反向构建需求文本"""
        if self.req_vector is None:
            return "默认需求"
        parts = []
        if self.req_vector.w_kill > 0.4:
            parts.append("最大化杀伤效能")
        if self.req_vector.w_cost > 0.4:
            parts.append("控制资源成本")
        if self.req_vector.w_time > 0.4:
            parts.append("提升时间效率")
        parts.append(f"优先级={self.req_vector.target_priority}")
        return ", ".join(parts)

    def _build_targets(self) -> list:
        """根据需求向量构建目标列表: 高价值 + 普通"""
        rng = np.random.default_rng(42)
        targets = []
        n_high = max(2, self.n_targets // 6)
        # 高价值目标
        for i in range(n_high):
            targets.append(Target(
                target_id=i, value=rng.uniform(0.8, 1.0),
                is_time_sensitive=rng.random() > 0.3,
                time_window=rng.uniform(60, 300),
                position=rng.uniform(0, 100, 2), velocity=rng.uniform(-5, 5, 2),
                priority=1,
            ))
        # 普通目标
        for i in range(n_high, self.n_targets):
            targets.append(Target(
                target_id=i, value=rng.uniform(0.2, 0.6),
                is_time_sensitive=False, time_window=np.inf,
                position=rng.uniform(0, 100, 2), velocity=rng.uniform(-3, 3, 2),
                priority=rng.integers(2, 4),
            ))
        return targets


# ============================================================
# CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="多Agent博弈 + PPO 军事决策优化系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py
  python main.py --requirement "最大化杀伤高价值目标，控制弹药成本"
  python main.py --requirement "优先打击时敏目标，3分钟内闭环" --episodes 800
  python main.py --interactive
        """,
    )
    parser.add_argument(
        "-r", "--requirement", type=str,
        default="最大化杀伤高价值目标，同时控制弹药成本和资源消耗",
        help="用户军事需求 (自然语言)",
    )
    parser.add_argument(
        "--episodes", type=int, default=500,
        help="PPO训练回合数 (默认: 500)",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=20,
        help="博弈最大轮数 (默认: 20)",
    )
    parser.add_argument(
        "--nodes", type=int, default=50,
        help="网络节点数 (默认: 50)",
    )
    parser.add_argument(
        "--targets", type=int, default=8,
        help="目标数 (默认: 8)",
    )
    parser.add_argument(
        "-i", "--interactive", action="store_true",
        help="交互模式",
    )
    parser.add_argument(
        "--skip-ppo", action="store_true",
        help="跳过PPO训练，仅输出博弈结果",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="output/reward_params.json",
        help="输出JSON路径",
    )

    args = parser.parse_args()

    if args.interactive:
        run_interactive(args)
    else:
        system = MilitaryDecisionSystem(
            n_nodes=args.nodes,
            n_targets=args.targets,
            ppo_episodes=0 if args.skip_ppo else args.episodes,
            max_rounds=args.max_rounds,
        )
        result = system.run(args.requirement)

        # 如果 skip-ppo，只运行到博弈阶段
        if args.skip_ppo:
            print("\n[INFO] --skip-ppo 模式: 仅输出博弈结果")
            output = {
                "requirement": args.requirement,
                "req_vector": result.get("req_vector", {}),
                "reward_params": result.get("reward_params", {}),
                "nash_equilibrium": result.get("game_result", {}).get("nash_equilibrium", {}),
            }
            import os
            os.makedirs("output", exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False, default=str)
            print(f"博弈结果已保存至 {args.output}")


def run_interactive(args):
    """交互模式"""
    print("\n" + "=" * 60)
    print("  交互模式 — 输入军事需求，获取PPO奖励参数")
    print("  输入 'quit' 退出, 'demo' 运行预置示例")
    print("=" * 60 + "\n")

    demo_requirements = [
        "最大化杀伤高价值目标，同时控制弹药成本在500万以内",
        "优先打击时敏目标，要求3分钟内完成杀伤链闭环",
        "在预算约束下摧毁尽可能多的敌方指挥节点，注重信息优势",
        "降低资源消耗，提高网络通信质量，打击普通目标即可",
        "快速响应时敏目标，最大化杀伤概率，不惜成本",
    ]

    while True:
        try:
            requirement = input("\n请输入军事需求 > ").strip()
            if not requirement:
                continue
            if requirement.lower() == "quit":
                print("退出。")
                break
            if requirement.lower() == "demo":
                print("\n预置示例:")
                for i, r in enumerate(demo_requirements):
                    print(f"  [{i+1}] {r}")
                choice = input("选择示例编号 (1-5): ").strip()
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(demo_requirements):
                        requirement = demo_requirements[idx]
                    else:
                        continue
                except ValueError:
                    continue

            print(f"\n处理需求: {requirement}")
            system = MilitaryDecisionSystem(
                n_nodes=args.nodes, n_targets=args.targets,
                ppo_episodes=200,  # 交互模式用较少的episodes
                max_rounds=args.max_rounds,
            )
            result = system.run(requirement)
            rp = result["reward_params"]
            print(f"\n{'=' * 60}")
            print(f"PPO奖励参数字段:")
            print(f"  {{")
            print(f'    "alpha":    {rp["alpha"]:.4f},  // 杀伤效能权重')
            print(f'    "beta":     {rp["beta"]:.4f},  // 成本控制权重')
            print(f'    "gamma":    {rp["gamma"]:.4f},  // 时间效率权重')
            print(f'    "delta":    {rp["delta"]:.4f},  // 信息优势权重')
            print(f'    "epsilon":  {rp["epsilon"]:.4f},  // 网络质量权重')
            print(f'    "min_kill_prob": {rp.get("min_kill_prob", 0.5)},')
            print(f'    "max_cost":      {rp.get("max_cost", 1000)},')
            print(f'    "time_window":   {rp.get("time_window", 300)}')
            print(f"  }}")
            print(f"{'=' * 60}")

        except KeyboardInterrupt:
            print("\n退出。")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
