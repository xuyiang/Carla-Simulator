import os
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from carla_env import CarlaEnv

# 每次训练用独立 run 名（时间戳 + 阶段标签），避免覆盖旧模型 / 旧 TB 日志
# 改 STAGE_TAG 区分课程阶段：stage1_lanekeep / stage2_brake_npc / stage3_traffic_light
STAGE_TAG = "stage_acc_hybrid_v3"
RUN_NAME = f"{STAGE_TAG}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

MODELS_DIR = "./models"
CKPT_DIR = os.path.join(MODELS_DIR, RUN_NAME, "checkpoints")
TB_DIR = "./logs"
os.makedirs(CKPT_DIR, exist_ok=True)

# debug_lidar=True 会弹出 OpenCV 俯视雷达窗口（拖慢仿真）；正式训练保持 False
env = CarlaEnv(
    spectator_follow=True,
    spectator_distance=12.0,
    spectator_height=4.0,
    spectator_pitch=-15.0,
    num_npcs=20,
)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=TB_DIR,
    device="cpu",
    seed=42
)

# 每 20k step 存一次中间权重，崩了/想回滚都救得回来
checkpoint_cb = CheckpointCallback(
    save_freq=20_000,
    save_path=CKPT_DIR,
    name_prefix="ppo",
)

# try/finally 保证 Ctrl+C 也能 close env（销毁 NPC、释放 TM 端口）
# KeyboardInterrupt 时额外存一份 interrupted 权重，免得卡在两个 checkpoint 之间白训
try:
    model.learn(
        total_timesteps=300_000,
        tb_log_name=RUN_NAME,
        callback=checkpoint_cb,
    )
    final_path = os.path.join(MODELS_DIR, f"{RUN_NAME}_final.zip")
    model.save(final_path)
    print(f"[train_ppo] saved final model -> {final_path}")
except KeyboardInterrupt:
    interrupted_path = os.path.join(MODELS_DIR, f"{RUN_NAME}_interrupted.zip")
    model.save(interrupted_path)
    print(f"[train_ppo] interrupted; saved -> {interrupted_path}")
finally:
    env.close()
