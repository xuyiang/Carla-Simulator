from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Independent


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 6, act_dim: int = 1):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )

        self.actor_mean = nn.Linear(64, act_dim)

        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))

        self.critic = nn.Linear(64, 1)

    def forward(self, obs):
        if obs.dim()==1:
            obs=obs.unsqueeze(0)
        #obs-action 分支
        feats=self.shared(obs)

        #右边critic
        mean = self.actor_mean(feats)
        std=torch.exp(self.actor_log_std.expand_as(mean))
        value=self.critic(feats)

        return mean,std,value
    @torch.no_grad()
    def get_action(self,obs_np,deterministic=False):
        obs_t=torch.as_tensor(obs_np,dtype=torch.float32)

        single=(obs_t.dim() == 1)
        #(5,1)->(1,5)
        mean,std,value = self.forward(obs_t)
        dist = Independent(Normal(mean,std),reinterpreted_batch_ndims=1)

        if deterministic:
            action=mean
        else:
            action=dist.sample()

        log_prob = dist.log_prob(action) #(B,)
        entropy = dist.entropy() #(B,)
        value = value.squeeze(-1) #(B,1)-> (B,)
        # (1, act_dim) -> (act_dim,) so env.step 里 action[0], action[1] 是油门/转向标量
        action_np = action.detach().cpu().numpy()      # (B, act_dim)
        log_prob_np = log_prob.detach().cpu().numpy()  # (B,)
        entropy_np  = entropy.detach().cpu().numpy()
        value_np    = value.detach().cpu().numpy()
        
        if single:
        # 单条样本：剥掉 batch 维，让 env.step(action[0], action[1]) 直接能用
            return action_np[0], float(log_prob_np[0]), float(entropy_np[0]), float(value_np[0])
        else:
        # batch 输入：原样返回 numpy 数组
            return action_np, log_prob_np, entropy_np, value_np

    def evaluate(self,obs,actions):
        """
        obs: (B, obs_dim) tensor
        actions: (B, act_dim) tensor
        return: log_probs (B,), entropy (B,), values (B,)
        """
        
        mean,std,values=self.forward(obs)
        dist =Independent(Normal(mean,std),reinterpreted_batch_ndims=1)
        log_probs=dist.log_prob(actions)
        entropy=dist.entropy()
        return log_probs, entropy , values.squeeze(-1)


class RolloutBuffer:
    """On-policy rollout storage for vector observations."""

    def __init__(self, obs_dim: int, act_dim: int, device: torch.device, capacity: int):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        self.capacity = capacity
        self.obs_buf: list = []
        self.act_buf: list = []
        self.logp_buf: list = []
        self.rew_buf: list = []
        self.done_buf: list = []
        self.val_buf: list = []

    def reset(self) -> None:
        self.obs_buf.clear()
        self.act_buf.clear()
        self.logp_buf.clear()
        self.rew_buf.clear()
        self.done_buf.clear()
        self.val_buf.clear()

    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        logp: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        self.obs_buf.append(np.asarray(obs, dtype=np.float32).reshape(-1).copy())
        self.act_buf.append(np.asarray(action, dtype=np.float32).reshape(-1).copy())
        self.logp_buf.append(float(logp))
        self.rew_buf.append(float(reward))
        self.done_buf.append(float(done))
        self.val_buf.append(float(value))


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generalized Advantage Estimation (Schulman et al.).
    rewards, dones, values: shape (T,)
    next_value: bootstrap V(s_{T}) when the rollout slice did not end in terminal state.
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
    for t in range(T - 1, -1, -1):
        if t == T - 1:
            next_nonterminal = 1.0 - dones[t]
            next_v = next_value
        else:
            next_nonterminal = 1.0 - dones[t]
            next_v = values[t + 1]
        delta = rewards[t] + gamma * next_v * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def ppo_update(
    ac: ActorCritic,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_range: float = 0.2,
    value_coef: float = 0.5,
    ent_coef: float = 0.01,
    max_grad_norm: float = 0.5,
) -> Tuple[float, float, float]:
    """
    One PPO minibatch update. Returns (policy_loss, value_loss, entropy_mean).
    """
    mean, std, values = ac.forward(obs)
    dist = Independent(Normal(mean, std), reinterpreted_batch_ndims=1)
    log_probs = dist.log_prob(actions)
    entropy = dist.entropy()

    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    value_pred = values.squeeze(-1)
    value_loss = 0.5 * ((value_pred - returns) ** 2).mean()

    loss = policy_loss + value_coef * value_loss - ent_coef * entropy.mean()

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(ac.parameters(), max_grad_norm)
    optimizer.step()

    return policy_loss.item(), value_loss.item(), entropy.mean().item()

