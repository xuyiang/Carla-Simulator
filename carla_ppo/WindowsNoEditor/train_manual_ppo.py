"""5. Training loop — wires CarlaEnv + RolloutBuffer + GAE + PPO updates."""
import numpy as np
import torch
import torch.optim as optim

from carla_env import CarlaEnv
from ppo_agent import ActorCritic, RolloutBuffer, compute_gae, ppo_update


def train(
    total_timesteps: int = 100_000,
    num_steps: int = 2048,
    n_epochs: int = 10,
    batch_size: int = 64,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    value_coef: float = 0.5,
    ent_coef: float = 0.01,
    device: str = "cpu",
):
    device = torch.device(device)
    env = CarlaEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    low = env.action_space.low.copy()
    high = env.action_space.high.copy()

    ac = ActorCritic(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(ac.parameters(), lr=lr, eps=1e-5)
    buffer = RolloutBuffer(obs_dim, act_dim, device, num_steps)

    global_step = 0
    obs, _ = env.reset()
    rollout_count = 0

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
                next_value = torch.tensor(0.0, device=device)
            else:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                _, _, v = ac.forward(obs_t)
                next_value = v.squeeze()

            rewards = torch.tensor(buffer.rew_buf, dtype=torch.float32, device=device)
            dones = torch.tensor(buffer.done_buf, dtype=torch.float32, device=device)
            values = torch.tensor(buffer.val_buf, dtype=torch.float32, device=device)

        advantages, returns = compute_gae(
            rewards, dones, values, next_value, gamma, gae_lambda
        )

        obs_t = torch.tensor(np.stack(buffer.obs_buf), dtype=torch.float32, device=device)
        act_t = torch.tensor(np.stack(buffer.act_buf), dtype=torch.float32, device=device)
        logp_old = torch.tensor(buffer.logp_buf, dtype=torch.float32, device=device)

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
                )
                total_pi += pi
                total_v += vloss
                total_ent += ent
                num_updates += 1

        if num_updates > 0:
            print(
                f"rollout {rollout_count} | step {global_step} | "
                f"pi {total_pi/num_updates:.4f} | v {total_v/num_updates:.4f} | ent {total_ent/num_updates:.4f}"
            )

    env.close()
    torch.save(ac.state_dict(), "ppo_manual_carla.pt")
    print("Saved ppo_manual_carla.pt")


if __name__ == "__main__":
    train()
