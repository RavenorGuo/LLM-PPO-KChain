"""
PPO 强化学习模块 — 定制奖励的 Actor-Critic 网络

核心: 接收 agent.py 博弈生成的 reward_params,
     构建定制奖励函数 R(s,a) = α·kill + β·cost_saving + γ·time_saving
                               + δ·info + ε·network,
     用 PPO 算法求解分布式杀伤链最优资源调度策略

参考论文:
- 论文3 方程(10): 多目标优化 → PPO 奖励函数映射
- 论文4: 集群杀伤链任务分配 (动作空间设计)
"""
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam

from model import KillChainEnv, KillChainNetwork, RequirementVector


# ============================================================
# 1. Actor-Critic 网络
# ============================================================

class ActorCritic(nn.Module):
    """PPO的Actor-Critic网络

    共享特征提取层 + 独立Actor/Critic头
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, act_dim), nn.Softmax(dim=-1),
        )
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        obs = torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-1.0)
        features = self.shared(obs)
        action_probs = self.actor_head(features)
        # 防止NaN: 用极小值替代NaN
        action_probs = torch.nan_to_num(action_probs, nan=1e-8, posinf=1.0, neginf=0.0)
        action_probs = torch.clamp(action_probs, min=1e-8, max=1.0)
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
        value = self.critic_head(features)
        return action_probs, value

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """采样动作"""
        obs = torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-1.0)
        action_probs, value = self.forward(obs)
        # 确保概率分布有效
        action_probs = torch.clamp(action_probs, min=1e-8, max=1.0)
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            dist = Categorical(action_probs)
            action = dist.sample()
        log_prob = torch.log(action_probs.gather(1, action.unsqueeze(-1)) + 1e-8).squeeze(-1)
        return action, log_prob, value.squeeze(-1), action_probs

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        """评估动作的 log_prob, entropy, value"""
        action_probs, value = self.forward(obs)
        dist = Categorical(action_probs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, value.squeeze(-1)


# ============================================================
# 2. 经验缓冲区
# ============================================================

@dataclass
class RolloutBuffer:
    """PPO经验回放缓冲"""
    obs: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    values: list = field(default_factory=list)
    dones: list = field(default_factory=list)

    def clear(self):
        for attr in ["obs", "actions", "log_probs", "rewards", "values", "dones"]:
            getattr(self, attr).clear()

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def get_batch(self):
        return {
            "obs": torch.FloatTensor(np.array(self.obs)),
            "actions": torch.LongTensor(np.array(self.actions)),
            "log_probs": torch.FloatTensor(np.array(self.log_probs)),
            "rewards": torch.FloatTensor(np.array(self.rewards)),
            "values": torch.FloatTensor(np.array(self.values)),
            "dones": torch.FloatTensor(np.array(self.dones)),
        }

    def __len__(self):
        return len(self.obs)


# ============================================================
# 3. PPO 训练器 (核心: 定制奖励)
# ============================================================

class PPOTrainer:
    """定制奖励 PPO 训练器

    接收 reward_params (来自agent.py博弈结果),
    构建 R = α·kill + β·cost_saving + γ·time_saving + δ·info + ε·network

    用法:
        trainer = PPOTrainer(reward_params, obs_dim=env.obs_dim, act_dim=env.action_space.nvec[0])
        policy = trainer.train(env, episodes=1000)
    """

    def __init__(self, reward_params: dict, obs_dim: int, act_dim: int,
                 hidden_dim: int = 256, lr: float = 3e-4, gamma: float = 0.99,
                 gae_lambda: float = 0.95, clip_epsilon: float = 0.2,
                 value_coef: float = 0.5, entropy_coef: float = 0.01,
                 ppo_epochs: int = 10, batch_size: int = 64,
                 device: str = None, verbose: bool = True):
        self.reward_params = reward_params
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.verbose = verbose

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic(obs_dim, act_dim, hidden_dim).to(self.device)
        self.optimizer = Adam(self.policy.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

        # 训练统计
        self.episode_rewards: list = []
        self.episode_lengths: list = []
        self.loss_history: list = []

        if verbose:
            print(f"[PPO] 初始化: obs_dim={obs_dim}, act_dim={act_dim}, device={self.device}")
            print(f"[PPO] 奖励参数: α={reward_params.get('alpha', 1.0):.4f} "
                  f"β={reward_params.get('beta', 0.5):.4f} "
                  f"γ={reward_params.get('gamma', 0.5):.4f} "
                  f"δ={reward_params.get('delta', 0.3):.4f} "
                  f"ε={reward_params.get('epsilon', 0.3):.4f}")

    def compute_gae(self, rewards, values, dones):
        """计算 Generalized Advantage Estimation"""
        advantages = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values)]
        return torch.FloatTensor(advantages), torch.FloatTensor(returns)

    def collect_rollout(self, env: KillChainEnv, max_steps: int = 200):
        """收集一轮经验"""
        obs, _ = env.reset()
        self.buffer.clear()
        episode_reward = 0.0

        for step in range(max_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, log_prob, value, _ = self.policy.get_action(obs_tensor)

            action_scalar = action.item()
            next_obs, reward, done, truncated, info = env.step(action_scalar)
            done_flag = done or truncated

            self.buffer.add(obs, action_scalar, log_prob.item(), reward, value.item(), done_flag)
            episode_reward += reward
            obs = next_obs

            if done_flag:
                break

        return episode_reward, step + 1

    def update(self):
        """PPO 更新 (clip surrogate)"""
        if len(self.buffer) < self.batch_size:
            return 0.0

        batch = self.buffer.get_batch()
        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        old_log_probs = batch["log_probs"].to(self.device)
        rewards = batch["rewards"].tolist()
        values = batch["values"].tolist()
        dones = batch["dones"].tolist()

        advantages, returns = self.compute_gae(rewards, values, dones)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        # 归一化 advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        n_updates = 0

        for _ in range(self.ppo_epochs):
            # Mini-batch
            indices = torch.randperm(len(obs))
            for start in range(0, len(obs), self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]

                batch_obs = obs[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]

                # 评估
                log_probs, entropy, values_pred = self.policy.evaluate(batch_obs, batch_actions)

                # PPO Clip Loss
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value Loss
                value_loss = F.mse_loss(values_pred, batch_returns)

                # Entropy Bonus
                entropy_bonus = entropy.mean()

                # Total
                loss = (policy_loss
                        + self.value_coef * value_loss
                        - self.entropy_coef * entropy_bonus)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_loss += loss.item()
                n_updates += 1

        avg_loss = total_loss / max(n_updates, 1)
        self.loss_history.append(avg_loss)
        return avg_loss

    def train(self, env: KillChainEnv, episodes: int = 1000,
              log_interval: int = 100, collect_steps: int = 200):
        """PPO 训练主循环"""
        print(f"\n[PPO] 开始训练 {episodes} episodes...")
        best_reward = -float("inf")

        for ep in range(1, episodes + 1):
            ep_reward, ep_len = self.collect_rollout(env, collect_steps)
            avg_loss = self.update()

            self.episode_rewards.append(ep_reward)
            self.episode_lengths.append(ep_len)

            if ep_reward > best_reward:
                best_reward = ep_reward

            if self.verbose and ep % log_interval == 0:
                recent = np.mean(self.episode_rewards[-log_interval:])
                print(f"  Ep {ep:4d}/{episodes} | "
                      f"Reward: {recent:7.2f} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Steps: {ep_len:3d} | "
                      f"Best: {best_reward:7.2f}")

        print(f"\n[PPO] 训练完成! Best reward: {best_reward:.2f}")
        return self.policy

    def save(self, path: str):
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "reward_params": self.reward_params,
            "episode_rewards": self.episode_rewards,
            "loss_history": self.loss_history,
        }, path)
        print(f"[PPO] 模型已保存至 {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"[PPO] 模型已从 {path} 加载")


# ============================================================
# 4. 推理 & 策略输出
# ============================================================

def inference(policy: ActorCritic, env: KillChainEnv, n_steps: int = 50,
              deterministic: bool = True) -> dict:
    """使用训练好的策略推理"""
    device = next(policy.parameters()).device
    obs, _ = env.reset()
    results = {
        "assignments": [], "rewards": [], "total_reward": 0.0,
        "kill_scores": [], "costs": [], "platform_usage": {},
    }

    for step in range(n_steps):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _, value, action_probs = policy.get_action(obs_tensor, deterministic=deterministic)

        action_scalar = action.item()
        next_obs, reward, done, truncated, info = env.step(action_scalar)

        results["assignments"].append({
            "step": step,
            "action": action_scalar,
            "action_probs": action_probs.squeeze(0).tolist(),
            "value": value.item(),
        })
        results["rewards"].append(reward)
        results["kill_scores"].append(info.get("kill_score", 0))
        results["costs"].append(info.get("cost", 0))
        results["total_reward"] += reward

        p_id = env.weapon_indices[action_scalar % len(env.weapon_indices)]
        results["platform_usage"][p_id] = results["platform_usage"].get(p_id, 0) + 1

        obs = next_obs
        if done or truncated:
            break

    return results


# ============================================================
# 5. 独立测试入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PPO 训练器测试")
    print("=" * 60)

    # 测试奖励参数 (模拟博弈输出)
    test_reward_params = {
        "alpha": 1.0, "beta": 0.5, "gamma": 0.5,
        "delta": 0.3, "epsilon": 0.3,
        "min_kill_prob": 0.5, "max_cost": 1000.0, "time_window": 300.0,
        "nash_product": 0.0625, "rounds_to_converge": 8,
    }

    # 创建环境
    network = KillChainNetwork.create_default(50)
    env = KillChainEnv(network=network, reward_params=test_reward_params, max_steps=50)

    # 训练
    n_weapons = env.action_space.n
    trainer = PPOTrainer(
        test_reward_params,
        obs_dim=env.obs_dim,
        act_dim=n_weapons,
        hidden_dim=128,
        lr=3e-4,
    )
    policy = trainer.train(env, episodes=500, log_interval=100, collect_steps=50)

    # 推理
    print("\n[PPO] 推理测试...")
    env_test = KillChainEnv(network=network, reward_params=test_reward_params, max_steps=20)
    result = inference(policy, env_test, n_steps=20)
    print(f"  Total Reward: {result['total_reward']:.2f}")
    print(f"  Avg Kill Score: {np.mean(result['kill_scores']):.3f}")
    print(f"  Avg Cost: {np.mean(result['costs']):.1f}")
    print(f"  Platforms Used: {len(result['platform_usage'])}")
