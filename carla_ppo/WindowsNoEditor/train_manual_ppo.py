"""
手写 PPO 训练脚本：CarlaEnv + RolloutBuffer + GAE + clipped surrogate。
与 stable-baselines3 的 train_ppo.py 并行存在；依赖 ppo_agent.py 中的网络与工具函数。
"""
from __future__ import annotations

import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim

from carla_env import CarlaEnv
from ppo_agent import ActorCritic, RolloutBuffer, compute_gae, ppo_update


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(
    total_timesteps: int = 300_000,
    num_steps: int = 2048,
    n_epochs: int = 10,
    batch_size: int = 64,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    value_coef: float = 0.5,
    ent_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    norm_adv: bool = True,
    device: str = "cpu",
    seed: int = 42,
    save_path: str | None = None,
    num_npcs: int = 12,
    debug_lidar: bool = False,
) -> None:
    set_seed(seed)
    device_t = torch.device(device)

    env = CarlaEnv(
        num_npcs=num_npcs,
        debug_lidar=debug_lidar,
    )
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    low = env.action_space.low.copy()
    high = env.action_space.high.copy()

    ac = ActorCritic(obs_dim, act_dim).to(device_t)
    optimizer = optim.Adam(ac.parameters(), lr=lr, eps=1e-5)
    buffer = RolloutBuffer(obs_dim, act_dim, device_t, num_steps)

    global_step = 0
    obs, _ = env.reset()
    rollout_count = 0

    out_path = save_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ppo_manual_carla.pt",
    )

    try:
        while global_step < total_timesteps:
            buffer.reset()
            last_done = False

            for _ in range(num_steps):
                if global_step >= total_timesteps:
                    break

                action, logp, _, val = ac.get_action(obs, deterministic=False)
                action = np.clip(action, low, high)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                buffer.push(obs, action, logp, reward, done, val)
                global_step += 1
                obs = next_obs
                last_done = done
                if done:
                    obs, _ = env.reset()

            if len(buffer.rew_buf) == 0:
                break

            rollout_count += 1
            n = len(buffer.rew_buf)

            with torch.no_grad():
                if last_done:
                    next_value = torch.tensor(0.0, device=device_t)
                else:
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device_t)
                    if obs_t.dim() == 1:
                        obs_t = obs_t.unsqueeze(0)
                    _, _, v = ac.forward(obs_t)
                    next_value = v.squeeze()

                rewards = torch.tensor(
                    buffer.rew_buf, dtype=torch.float32, device=device_t
                )
                dones = torch.tensor(
                    buffer.done_buf, dtype=torch.float32, device=device_t
                )
                values = torch.tensor(
                    buffer.val_buf, dtype=torch.float32, device=device_t
                )

            advantages, returns = compute_gae(
                rewards, dones, values, next_value, gamma, gae_lambda
            )
            if norm_adv:
                adv_std = advantages.std()
                if adv_std > 1e-8:
                    advantages = (advantages - advantages.mean()) / adv_std

            obs_t = torch.tensor(
                np.stack(buffer.obs_buf), dtype=torch.float32, device=device_t
            )
            act_t = torch.tensor(
                np.stack(buffer.act_buf), dtype=torch.float32, device=device_t
            )
            logp_old = torch.tensor(
                buffer.logp_buf, dtype=torch.float32, device=device_t
            )

            total_pi, total_v, total_ent = 0.0, 0.0, 0.0
            num_updates = 0
            idx = np.arange(n)

            for _ in range(n_epochs):
                np.random.shuffle(idx)
                for start in range(0, n, batch_size):
                    end = min(start + batch_size, n)
                    mb = idx[start:end]
                    mb_obs = obs_t[mb]
                    mb_act = act_t[mb]
                    mb_logp = logp_old[mb]
                    mb_adv = advantages[mb]
                    mb_ret = returns[mb]

                    pi, vloss, ent = ppo_update(
                        ac,
                        optimizer,
                        mb_obs,
                        mb_act,
                        mb_logp,
                        mb_adv,
                        mb_ret,
                        clip_range=clip_range,
                        value_coef=value_coef,
                        ent_coef=ent_coef,
                        max_grad_norm=max_grad_norm,
                    )
                    total_pi += pi
                    total_v += vloss
                    total_ent += ent
                    num_updates += 1

            if num_updates > 0:
                print(
                    f"[manual_ppo] rollout {rollout_count} | step {global_step}/{total_timesteps} | "
                    f"pi_loss {total_pi / num_updates:.4f} | "
                    f"v_loss {total_v / num_updates:.4f} | "
                    f"entropy {total_ent / num_updates:.4f}",
                    flush=True,
                )

        d = os.path.dirname(os.path.abspath(out_path))
        if d:
            os.makedirs(d, exist_ok=True)
        torch.save(
            {
                "actor_critic": ac.state_dict(),
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "seed": seed,
            },
            out_path,
        )
        print(f"[manual_ppo] saved -> {out_path}", flush=True)

    except KeyboardInterrupt:
        interrupt_path = out_path.replace(".pt", "_interrupted.pt")
        d = os.path.dirname(os.path.abspath(interrupt_path))
        if d:
            os.makedirs(d, exist_ok=True)
        torch.save(
            {
                "actor_critic": ac.state_dict(),
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "seed": seed,
            },
            interrupt_path,
        )
        print(f"[manual_ppo] KeyboardInterrupt; saved -> {interrupt_path}", flush=True)
        raise
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Manual PPO training on CarlaEnv")
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--num-steps", type=int, default=2048, help="Rollout length per update")
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--no-norm-adv", action="store_true", help="Disable advantage whitening")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--save",
        type=str,
        default=None,
        help="Output .pt path (default: ./ppo_manual_carla.pt next to this script)",
    )
    p.add_argument("--num-npcs", type=int, default=12)
    p.add_argument(
        "--debug-lidar",
        action="store_true",
        help="OpenCV LiDAR BEV (slows training)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        total_timesteps=args.timesteps,
        num_steps=args.num_steps,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        value_coef=args.value_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        norm_adv=not args.no_norm_adv,
        device=args.device,
        seed=args.seed,
        save_path=args.save,
        num_npcs=args.num_npcs,
        debug_lidar=args.debug_lidar,
    )
