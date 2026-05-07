# Carla-Simulator

Autonomous driving experiments in **CARLA** using RL (PPO), with room for MPC/PID-style control. The aim is stable end-to-end driving behavior in simulation.

---

## Stack

| Piece | Choice |
|--------|--------|
| Simulator | [CARLA](https://carla.org/) |
| RL | [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) PPO |
| Python API | `carla` client + Gym-style env wrapper |

---

## Setup

1. Install and run a CARLA release that matches your client API version.
2. Create the Python environment from `carla_ppo/`:

   ```bash
   cd carla_ppo
   conda env create -f environment.yml
   conda activate rl-driving
   ```

   Or with pip: `pip install -r requirements.txt` (see comments in that file for GPU PyTorch).

3. Training scripts live under **`carla_ppo/WindowsNoEditor/`** (e.g. `carla_env.py`, `train_ppo.py`).

---

## Training versions

### V1 — Lane keeping (PPO)



- **Task:** Lane keeping **without** brake control in the action space.
- **Actions** (`carla_env.py`): `Box(low=[0, -1], high=[1, 1])` — `[throttle, steer]`.
- **Training:** `train_ppo.py`, **100k** environment steps, PPO from Stable-Baselines3.
- **Reward (concept):** `0.3 * velocity - 0.3 * lateral_norm`, plus **-50** on collision; extra shaping for smooth steering (penalty on sharp steer).
- **Note:** In V1, **one RL decision step = one CARLA control step** (no frame-skip between action and sim step).
-**Challenge** I train 3 model to achivel the final model, all using PPO. However, Model is only perform well in simple task like Lane Keeping. Once more NPC car is adding into the environment. Using PPO policy's action directly apply control to the vehicle is too challenge. Especially when i try to set action[0] as throttle and Brake at the same time. The reward function of moving forward and brake when NPC infornt seems contridict to each other. Even the steer sharp penalty & action repeat methods works well for smooth. The vechicle still hard to apply great break control. This lead to my next stage Model plan. Acc+ PPO for steering. I am trying to align this more with real Autonomous Driving Plan.
![V1 lane keeping in CARLA](./carla_ppo/videos/IMG_2368.gif)

**Setup:** Town02, **0 NPC** vehicles, relatively simple observations + custom reward, full RL PPO.

**Outcome:** Smooth steering and **lane following** in CARLA; policy learns throttle/steer coordination without explicit brake dimension.

### Demo



[V1 lane following in CARLA — YouTube](https://www.youtube.com/watch?v=ElbHdQus8k0)
![V1 Demo](./carla_ppo/videos/Task%20three%20result.gif)
---

## License

See [`LICENSE`](LICENSE) in the repository root.
