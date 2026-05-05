# Carla-Simulator
Autonomous Driving On Carla Simular Based on RL/MPC/PID, the goal of this project is to achieve fully automous driving in Carla.

# Tranning Version:

## V1:
I set the simple task of lane keep with out breke contro. So i define the 'action space' with Low[0,-1] High[1,1] with action[0]='Throttle'&Action[1]='steer' in *carla_env.py*. Train will *train_ppo.py*. Using simple reward 'method=velocity * 0.3-lateral_norm * 0.3 if collision=True:-50.0'. I used PPO from stable_baseline3 with training 100K steps(aka decision step). **Note: for V1**,Tranning Step(Decision Step)=Carla Step(action steps). 
**Result**: Weak Obs+Customize Reward with Fully Rl PPO in **Town02** with **0 NPC**, achieve smooth steer control with some support steer/smooth reward(sharp steer penalty). Actived an smooth **lane following** in Carla.



