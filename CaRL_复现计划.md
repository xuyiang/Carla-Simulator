# CaRL 本地复现计划

> 论文：CaRL: Learning Scalable Planning Policies with Simple Rewards (CoRL 2025, arXiv:2504.17838)
> 官方代码：https://github.com/autonomousvision/CaRL （含 CARLA leaderboard 2.0 的完整 RL 代码）
> 本文档：面向本机（Win10 + i7-11700K 8C16T + 32GB RAM + RTX 3060 Ti 8GB）的分阶段复现计划

---

## 0. 目标设定（先对齐预期）

| 配置 | 论文原版 | 本机目标 |
|---|---|---|
| 硬件 | 8×A100 节点，上百 CPU 核 | 1×3060 Ti + 8 核 CPU |
| 并行环境数 | 256~1024 | 4~8 |
| 训练样本 | 300M | **10M（对齐论文 Table 4 tiny 配置）** |
| batch / mini-batch | 32768 / 8192 | 2048~4096 / 512~1024 |
| longest6 v2 预期 DS | 64（v1.0）/ 73（v1.1） | **≈31±7（论文 10M 消融的参考值）** |

复现分两层：
- **A 线（跑通官方）**：用官方 CaRL 仓库在本机跑通"评测预训练模型 → tiny 规模训练 → 评测自己训的模型"全流程。这是保底目标。
- **B 线（自研实现）**：按论文规格自己实现 观测渲染 + 神经网络 + 奖励 + PPO，接到官方的 custom_leaderboard 环境上训练。这是"设计神经网络/训练/评测"的部分，A 线代码作为对照和调试基准。

---

## 1. 现有资产

- CARLA 0.9.15 WindowsNoEditor：`e:\zhuomian\Carla\Carla-Simulator\carla_ppo\WindowsNoEditor`（版本与 CaRL 要求一致，无需重装）
- conda 环境 `rl-driving`：Python 3.9.23，torch 2.5.1+cu121（CUDA 可用），carla 0.9.15，gymnasium，stable-baselines3
- 旧代码 `carla_env.py` / `train_ppo.py` 等：SB3 版 PPO，仅作参考。CaRL 需要 Beta 分布动作头、非对称 Critic、大 batch CleanRL 风格 PPO，不复用 SB3。

---

## 2. 总体架构（官方代码的运行方式）

```
┌─────────────────────┐   RPC(2000+i)   ┌──────────────────────────────┐
│ CarlaUE4.exe ×N     │◄───────────────►│ custom_leaderboard            │
│ (-nullrhi 纯CPU模式, │                 │ leaderboard_evaluator.py ×N   │
│  不占GPU显存)        │                 │ + team_code/env_agent.py      │
└─────────────────────┘                 │ (渲染BEV观测/算奖励/执行动作)   │
                                        └──────────────┬───────────────┘
                                                       │ ZMQ 消息 (gym_port 5555+i)
                                        ┌──────────────▼───────────────┐
                                        │ team_code/dd_ppo.py (GPU)     │
                                        │ PPO 训练主循环 (CleanRL风格)   │
                                        └──────────────────────────────┘
```

- 每个并行环境 = 1 个 CARLA server 进程 + 1 个 leaderboard client 进程，client 通过消息把观测发给训练进程、拿回动作。
- `train_parallel.py` 负责拉起并监控所有进程（CARLA 会偶发崩溃，它自动重启）。
- 训练时仿真 10 Hz；评测时按 leaderboard 标准 20 Hz、动作重复 2 帧。

---

## 3. 阶段计划

### 阶段 0：环境搭建（0.5~1 天）

1. Clone 官方仓库到 `e:\zhuomian\Carla\Carla-Simulator\CaRL`（仓库已内置 custom_leaderboard、original_leaderboard、scenario_runner，不用单独 clone leaderboard）。
2. 环境：先尝试直接用 `rl-driving`，按官方 `environment.yml` 补装缺的包（zmq、opencv、tensorboard、wandb 等）。若冲突再新建 `carl` 环境。
3. 设置环境变量（PowerShell 版）：
   ```powershell
   $env:CARLA_ROOT="e:\zhuomian\Carla\Carla-Simulator\carla_ppo\WindowsNoEditor"
   $env:WORK_DIR="e:\zhuomian\Carla\Carla-Simulator\CaRL\CARLA"
   $env:SCENARIO_RUNNER_ROOT="$env:WORK_DIR\original_leaderboard\scenario_runner"
   $env:LEADERBOARD_ROOT="$env:WORK_DIR\original_leaderboard\leaderboard"
   $env:PYTHONPATH="$env:CARLA_ROOT\PythonAPI\carla;$env:SCENARIO_RUNNER_ROOT;$env:LEADERBOARD_ROOT"
   ```
4. 验证：启动 `CarlaUE4.exe -nullrhi -carla-rpc-port=2000 -nosound`，用 Python `import carla; carla.Client('localhost',2000).get_server_version()` 连通。
5. **Windows 兼容排查**（本阶段最大风险点，官方只在 Linux 测过）：
   - 官方 `.sh` 脚本需改写成 `.ps1` 或直接手敲命令；
   - `torch.distributed.run`：单卡单进程，backend 用 `gloo`（Windows 无 NCCL）；
   - `train_parallel.py` 的进程管理若有 Linux 专属调用（信号、`os.setsid` 等）需小改。
   - **兜底方案**：训练/客户端代码放 WSL2 里跑，CARLA server 用 `-nullrhi` 留在 Windows 侧，二者走 localhost 端口互通。若 Windows 原生适配卡住超过 1~2 天，直接切 WSL2。

### 阶段 1：先跑通评测（1~2 天）—— 不训练，先验证评测管线

1. 下载官方预训练权重（`CARLA/results/CaRL_PY_00` 等，含 `config.json` + `model_final.pth`）。
2. 单路线冒烟：用 `original_leaderboard` 的 `leaderboard_evaluator.py`，`--agent team_code/eval_agent.py`，`--routes` 用 debug 路线，设 `SAMPLE_TYPE=mean`。开 `DEBUG_ENV_AGENT=1` + `SAVE_PATH` 保存可视化，肉眼确认车在正常开。
3. 跑 longest6 v2 完整 36 条路线（串行即可，`-nullrhi` + CaRL 推理很快；官方脚本是 SLURM 并行的，本机写个 PowerShell/Python 循环逐条跑）。
4. 用 `tools/result_parser.py` 聚合出 DS/RC/各类违规。**验收标准：预训练模型 DS 在 60~75 区间**（官方 v1.0=64，v1.1=73），说明评测管线正确。这个数也是后面自己模型的天花板参照。

### 阶段 2：跑通训练管线（2~3 天）

1. 训练路线数据：直接用仓库自带的 `custom_leaderboard/leaderboard/data/1000_meters_old_scenarios_01`（官方生成好的 ~1km 训练路线 + longest6 六类场景，Town01-06，每个环境一个文件）。不需要自己跑 `generate_long_routes_with_scenarios.py`。
2. 单环境冒烟：按 README "Training Debugging" 手动起 1 个 CARLA server + 1 个 leaderboard client + `dd_ppo.py`（`--num_envs_per_gpu 1 --total_batch_size 512 --total_minibatch_size 128 --reward_type simple_reward`），确认能采样、能更新、TensorBoard 有曲线。
3. 并行度压测：逐步加到 4/6/8 个 server（每个约 2~3GB 内存，32GB 内存上限约 8 个；CPU 8 核是瓶颈），server 启动参数加 `-RPCThreads=2 -StreamingThreads=2 -SecondaryThreads=2 -nothreading` 限线程。记录总采样 FPS，据此估算 10M 样本需要的天数。
4. 用 `train_parallel.py`（或自己写的 Windows 版启动脚本）实现崩溃自动重启，保证多天训练能无人值守。

### 阶段 3：神经网络设计（B 线核心，2~4 天实现）

按论文附录 B 的规格（最终模型约 **200 万参数**）：

**观测（输入）**
- BEV 语义分割图 `256×256×~10` 通道：道路、A* 路线掩码（仅路口内渲染条件引导）、车道线、车辆、行人、红绿灯、停车标志、限速标志、静态物体、路肩。
- 视野范围：前 78m / 后 50m / 左右各 64m，分辨率 2 像素/米。
- 关键细节：车辆/行人不用二值掩码，**把速度值编码进 bounding box 亮度**（替代历史帧，省显存）；车辆通道额外渲染开门、匀速外推预测线、转向灯/刹车灯/双闪。
- 标量测量向量：上一帧 steer/throttle/brake、档位、速度、速度向量、当前限速。

**网络结构**
- 图像分支：Roach 风格 CNN（6 层卷积）+ 为 256 输入多加 1 层 stride-2 Conv2D，保持输出特征数不变 → flatten → Linear。
- 测量分支：小 MLP。
- 融合：concat → 全连接 → 分出 Actor / Critic 两个头。
- **Actor 头（Beta 分布）**：动作 2 维——转向 ∈[-1,1]，油门/刹车合并为一维 ∈[-1,1]（互斥）。每维输出 α、β 两个参数，激活用 `Softplus(x)+1` 保证 α,β≥1（单峰、不退化）。**推理时取分布均值**（不是众数）作为确定性动作。
- **Critic 头（非对称）**：额外输入策略看不到的信息——距超时的剩余时间、距 blocked 判罚的时间、剩余路线长度、TTC 罚项剩余帧数、舒适度罚项剩余帧数（均归一化）。仅训练用，不影响推理。

### 阶段 4：奖励设计（与阶段 3 并行）

核心公式（论文式 1）：

```
r_t = RC_t × ∏ p_t − T
```

- `RC_t`：本仿真步完成的路线百分比（唯一的正奖励来源）。
- **软惩罚 `p_t ∈ [0,1)`**（乘性折减，不终止）：超速、舒适度（加速度/jerk 越界）、TTC 过小、压线/偏离车道、停止标志未停等。早期不可避免的项（如舒适度）必须 >0，否则学不到东西。可对持续状态连续多帧施加。
- **硬惩罚（直接终止 episode）**：碰撞、闯红灯、偏离路线过远、blocked（长时间不动）、超时。
- 终端罚 `T`：碰撞和闯红灯 T=1，其余 T=0。
- 完整数值以官方 `team_code/rl_config.py` 中 `simple_reward` 的默认参数为准，自研实现直接对照抄参数。
- （可选消融）同时实现 Roach 复杂奖励 `reward_type=roach`，用于复现论文图 1 的"简单奖励在大 mini-batch 下反超"的对比实验。

### 阶段 5：训练设计（PPO，本机缩放版）

超参数以论文 Table 9 的 CaRL 列为基准，按本机算力缩放：

| 超参数 | 论文 CaRL | 本机建议 |
|---|---|---|
| 学习率 | 2.5e-4，线性衰减到 0 | 同 |
| 并行环境 num_envs | 512/1024 | 4~8 |
| batch size | 16384/32768 | 2048~4096 |
| mini-batch size | 4096/8192 | 512~1024 |
| epochs | 3 | 3 |
| γ / GAE λ | 0.99 / 0.95 | 同 |
| entropy coef / vf coef | 0.01 / 0.5 | 同 |
| max grad norm | 0.5 | 同 |
| value loss clipping | 开 | 同 |
| 动作分布 | Beta（Softplus+1） | 同 |
| 总样本 | 300M | **10M**（约数天；若吞吐不够先做 5M） |
| 仿真频率 | 训练 10 Hz | 同 |

训练监控（TensorBoard）：episode return、平均 RC、各违规类型计数、KL 散度、clip fraction、value loss、entropy。
预期学习曲线：先学会起步直行（RC 上升）→ 碰撞率下降 → 路口/场景通过率缓慢爬升。若 10 万步后 return 仍为 0 附近，优先查奖励管线和观测渲染（保存 BEV 图肉眼检查）。

里程碑训练序列：
1. **T0 冒烟**：1 环境 ×10 万步，确认 return 上升趋势。
2. **T1 小跑**：4~8 环境 ×1~2M 样本，模型应能在简单直路上稳定行驶（评测 DS>10）。
3. **T2 正式**：8 环境 ×10M 样本，多天无人值守（依赖阶段 2 的自动重启），目标 DS≈25~35。
4. （可选）**T3 消融**：mini-batch 256 vs 1024 对比，复现论文核心论点。

### 阶段 6：评测设计

- **基准**：longest6 v2（36 条路线，Town01-06，每条 1~2km、5~21 个安全关键场景，背景车最高 80 km/h），路线文件在 `custom_leaderboard/leaderboard/data/longest6.xml`（按 split 逐条跑）。
- **协议**：original_leaderboard（未修改的官方 leaderboard 2.0 代码）、20 Hz、动作重复 2、Beta 均值确定性动作。
- **指标**：Driving Score（主指标）、Route Completion、每公里各类违规（行人碰撞 Ped、车辆碰撞 Veh、闯红灯、min-speed MS 等），用 `result_parser.py` 聚合成 CSV。
- **日常小评测**：抽 6 条路线做 dev 子集，每训练 1~2M 样本评一次，省时间；最终模型跑全部 36 条。
- **降方差**：CARLA 评测方差大，论文是 5 个训练种子 × 各评 3 次。本机至少做最终模型评 3 次取均值；种子数量按时间预算（1~3 个）。
- **对照表**：PDM-Lite 73 / CaRL 300M 64~73 / 论文 10M tiny 31±7 / Roach 22 / Think2Drive 7。本机 10M 模型落在 25~35 即复现成功。

### 阶段 7：总结与可选扩展

- 写实验报告：学习曲线、最终 DS/RC/违规表、与论文 Table 4/5 对照、失败案例可视化（`RECORD=1` 录违规片段）。
- 可选：简单奖励 vs Roach 奖励消融；mini-batch 缩放消融；继续训到 20~30M 样本看 DS 增长。

---

## 4. 风险清单

| 风险 | 影响 | 对策 |
|---|---|---|
| 官方代码 Windows 不兼容（bash 脚本、distributed、进程管理） | 阶段 0~2 卡住 | 优先小改；卡 >2 天切 WSL2（训练侧在 WSL2，CARLA server 留 Windows `-nullrhi`） |
| CPU 8 核限制并行环境数 → 采样吞吐低 | 10M 样本训练时间拉长（可能 5~10 天） | 限制 server 线程数；训练期间不用机器干别的；必要时降到 5M 样本 |
| CARLA 0.9.15 偶发崩溃 | 长训中断 | train_parallel 自动重启 + PPO checkpoint 定期落盘续训 |
| 32GB 内存上限 | server 数量 ≤8 | 监控内存，每 server ~2-3GB |
| 评测方差大 | 结论不可靠 | 最终评测 ×3 次取均值；固定评测协议 |
| 自研实现（B 线）有 bug 难定位 | 训练不收敛 | 每个组件与官方代码对拍：BEV 渲染图逐像素比、奖励逐帧比、同权重下动作输出比 |

## 5. 时间线（兼职节奏，约 4~6 周）

- 第 1 周：阶段 0 + 阶段 1（环境 + 预训练模型评测跑通）
- 第 2 周：阶段 2（训练管线冒烟 + 并行压测）+ 阶段 3/4 开工
- 第 3~4 周：阶段 5（T1 小跑 → T2 10M 正式训练，训练期间完善评测脚本）
- 第 5 周：阶段 6（完整评测 + 复评降方差)
- 第 6 周：阶段 7（报告 + 可选消融）
