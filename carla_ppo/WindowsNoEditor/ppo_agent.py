import torch
import torch.nn as nn
from torch.distributions import Normal, Independent


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 5, act_dim: int = 2):
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

        

