# Carla-Simulator

Autonomous driving experiments in **CARLA** using RL (PPO), with room for MPC/PID-style control. The aim is stable end-to-end driving behavior in simulation.
Note the main code is located in **"Carla-Simulator/carla_ppo/WindowsNoEditor"**

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

### V2 — ACC+P"ID" for Throttle&Brake, RL PPO for Steering

I am still tunning this model, facing some challenge for throttle and brake control.

Failing Note:
Mutiple mistake i made during training:
1. Single Lidar Channel:1 too weak to reflect any **Actor** in front 30 degree. Need to set back to default as leat for a better perfomance
2. I realized Town02 is too small and contain a lot left turn. So easy to lower my lidar distance， this lead another bad consquence
3. Acc over sensitive to lidar when V_ego is too low because bad design of clip range.
            denom = max(d_safe + 5.0 - self.acc_d_min, 1e-3)
            ratio = (d_lead - self.acc_d_min) / denom
            v_target = self.acc_v_set * float(np.clip(ratio, 0.0, 1.0))
Basically, v_target only change small scale when vehicle moving with high speed, but v_target change 2m/s when lidar:d_lead(self.lidarditance) change by 1.
This lead to even worse on control part of acc:
   speed_err = v_target - v_ego
        if speed_err > 0:
            throttle = float(np.clip(self.acc_kp_throttle * speed_err, 0.0, self.acc_max_throttle))
            brake = 0.0
        else:
            throttle = 0.0
            brake = float(np.clip(-self.acc_kp_brake * speed_err, 0.0, 1.0))
        return throttle, brake

Which means the once v_target>v_ego, brake will immediatly change to throttle. So this is super confused for my model when v_ego is low with super sensitive lidar_distance.  
4. Acc control the forward direction, which mean Policy only getting physical reward from steering, which lead a worse cold start. Only lane_keeping and yaw_heading is providing reward. 
5. low_speed steer penalty teach model not to steer when car stop by over sensitive lidar

Update direction that i come up:
1. make sure physic reward vehicle_yaw is use with lateral_norm, right now if car is heading the direction of waypoint, it will get reward. But what we actually want is the steering is pointing the direction of center based on later_norm(positive/negatuve). I check the latest method: we should reward both speed and direction: speed*cos_h.
2.I checked with openpoilt, I notice lateral_acceletration is more commonly used in observation, because a car park on the side of the lane still provide an non-negative lateral error.
3. The bottleneck is still low_speed_steer penalty and week lidar, i want replace lidar with semantic lidar that provide by Carla.sensor.lidar.ray_cast_semantic. This will retrive the "Unsigned int containing the semantic tag of the object it." which i can check if this is vehicle. Index: 14,15,16 will be used this case based on Unreal/CarlaUE4/Content/Static
 
Trainning Note:
Model converge quicker on land following task compare to previous. But in early steps,because 
        forward = 0.5 * self.speed_norm * self.cos_h
        lane_keep = (1.0 - abs(self.lateral_norm)) * 0.4
The model is satify with driving forward along the path direction, even on the right lane, they already get reward and can run for a while as long as no building the side or NPC come from other side. Model Already can drive stright in 20K-30K global step. Which is more quick compare to V1.
Potential weakness: lidar_distance was update based on frequnecy, raw lidar 

5.20 Finally back from Final:

New noticed: 

Action Repeat is benefit for most case, but my action repeat is 50ms/time.sleep(0.05),this lead to 2m/s per second without any control can be apply to vehicle if we going full speed, which make super hard to turn at high speed.
Another thing i found and fixed that model stucked in local min due to lack of offroad/wrong way observation, which make the penalty looks super "BlackBox" form model perspective.
我觉得action repeat在carla simulator中的车速是有一个最优的方法论的，别的rl的思路有用但是hyperparametr应该不一样，需要根据车速来设计

### V4 stage 4(remove action_repeat)

5.20 Trainning
Something that i notice for Town02 is all this turn is 90 degree turn, making hard for model to learn in the beginning. I will apply next time in Better Town.
小范围实验成功，300K Step can follow lane, bad in trunning. I am going to test higher tranning self.lane_keep_coef = 0.5->0.8


## 阶段目标达成！
Svaed to .\models\stage_acc_hybrid_v4_20260520_040630_final.zip
完成ACC+Brake control+ PPO steering的目标，现在转移到更复杂的Town03中增强model robust.

### Demo



[V2  ACC+PPO for Steering — YouTube](https://www.youtube.com/watch?v=XdELDMxeIsc)
![V2 Demo](./carla_ppo/videos/IMG_3450.gif)

5.23
to do read
Neuro-Cognitive Reward Modeling for Human-Centered Autonomous Vehicle Control (CVPR 2026)
AEGIS: Human Attention-based Explainable Guidance for Intelligent Vehicle Systems 
(CHI 2025) 










## License

See [`LICENSE`](LICENSE) in the repository root.
