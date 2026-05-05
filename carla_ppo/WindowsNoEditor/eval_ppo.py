from stable_baselines3 import PPO

from carla_env import CarlaEnv

# Eval 用一张训练时没见过的地图 + 大量 NPC，测泛化能力
# Town03：城区 + 环岛 + 多车道路口，spawn point ~150 个，足够塞下 80 辆 NPC
# 想再换可以试 Town05（路口超多）或 Town04（带高速）
EVAL_MAP = "Town02"
EVAL_NPCS = 80
MODEL_PATH = r".\models\stage2_rewardv2_20260503_204159_final.zip"

env = CarlaEnv(
    map_name=EVAL_MAP,
    num_npcs=EVAL_NPCS,
    spectator_follow=True,
    spectator_distance=12.0,
    spectator_height=4.0,
    spectator_pitch=-15.0,
    debug_lidar=False,
)

model = PPO.load(MODEL_PATH)
obs, info = env.reset()

episode = 0
ep_reward = 0.0
ep_steps = 0
collisions = 0

try:
    for i in range(300):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        ep_steps += 1

        print(
            f"step={i:4d}  ep={episode}  v={env.speed:5.2f}m/s  "
            f"lat={env.lateral_norm:+.2f}  d={env.lidar_distance:5.1f}m  "
            f"thr={info.get('throttle', 0):.2f}  brk={info.get('brake', 0):.2f}  "
            f"r={reward:+.2f}"
        )

        if terminated or truncated:
            if terminated:
                collisions += 1
            print(
                f"--- episode {episode} end: ep_reward={ep_reward:.1f}, "
                f"len={ep_steps}, terminated={terminated}, total_collisions={collisions} ---"
            )
            obs, info = env.reset()
            episode += 1
            ep_reward = 0.0
            ep_steps = 0
finally:
    print(f"[eval] map={EVAL_MAP}, npcs={EVAL_NPCS}, episodes={episode}, collisions={collisions}")
    env.close()
