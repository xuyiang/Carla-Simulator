from carla_env import CarlaEnv

env = CarlaEnv(debug_lidar=True)

obs, info = env.reset()
print("reset 成功，obs shape:", obs.shape)
print("obs:", obs)

for i in range(100):
    action = env.action_space.sample()   # 随机采样一个动作
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"step {i+1}: action={action}, reward={reward}, terminated={terminated}")

env.close()
print("close 成功")