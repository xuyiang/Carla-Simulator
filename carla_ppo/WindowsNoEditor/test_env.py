from carla_env import CarlaEnv

env = CarlaEnv(debug_lidar=True)

obs, info = env.reset()
print("reset 成功，obs shape:", obs.shape)
print("obs:", obs)

# Gymnasium 契约：step 返回 terminated/truncated=True 后，调用方必须自己 reset()
# 否则 env 处于「上一回合的终态」，再 step 行为是未定义/奇怪的
# （在我们这里就是：collision_happened 还是 True，每步立刻 terminated）
episode = 0
ep_steps = 0
ep_reward = 0.0
try:
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        ep_steps += 1
        print(
            f"step {i + 1}: ep={episode} ep_step={ep_steps} "
            f"r={reward:+.2f} term={terminated} trunc={truncated}"
        )
        if terminated or truncated:
            print(
                f"--- episode {episode} 结束: ep_reward={ep_reward:.1f}, "
                f"len={ep_steps}, terminated={terminated}, truncated={truncated} ---"
            )
            obs, info = env.reset()
            episode += 1
            ep_steps = 0
            ep_reward = 0.0
finally:
    env.close()
    print("close 成功")
