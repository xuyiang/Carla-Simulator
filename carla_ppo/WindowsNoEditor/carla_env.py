import math
import random
import time

import carla
import cv2
import gymnasium as gym
import numpy as np

class CarlaEnv(gym.Env):

    def __init__(
        self,
        spectator_follow: bool = True,
        spectator_distance: float = 12.0,
        spectator_height: float = 4.0,
        spectator_pitch: float = -15.0,
        debug_lidar: bool = False,
        lidar_vis_fps: float = 20.0,           # 和 LiDAR rotation_frequency 对齐，最丝滑
        lidar_vis_max_points: int = 30000,     # 600k pts/s ÷ 20Hz = 30k/帧，全显示
        num_npcs: int = 12,
        map_name: str = "Town02",
    ):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(20.0)
        # 训练默认 Town02，eval 时可以换成 Town03 / Town05 等更大、布局更不同的地图，测泛化
        self.map_name = map_name
        self.world = self.client.load_world(map_name)
        # Traffic Manager 用来给 NPC 跑 autopilot；端口默认 8000
        self.tm = self.client.get_trafficmanager()
        self.tm.set_global_distance_to_leading_vehicle(2.5)
        self.tm.global_percentage_speed_difference(20.0)  # NPC 比限速慢 20%，更易于 ego 学避让
        self.num_npcs = int(num_npcs)
        self.npc_vehicles = []
        self.spectator_follow = spectator_follow
        self._spectator_distance = spectator_distance
        self._spectator_height = spectator_height
        self._spectator_pitch = spectator_pitch
        # 路径 A 混合控制：RL 只输出 steer ∈ [-1, 1]
        # throttle / brake 由内部经典 ACC 控制器 (_compute_acc_control) 计算，不交给网络
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([ 1.0], dtype=np.float32),
            dtype=np.float32
        )
        # obs 6 维：speed_norm, lateral_norm, sin_h, cos_h, lidar_d/30, lateral_accel/6
        # 加了 lateral_accel 让模型能区分「停在路边」和「行驶中跑偏」——前者 a_y=0 后者 a_y 大
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        self.vehicle=None
        self.camera=None
        self.collision_sensor = None
        self.collision_happened = False
        self.lidar=None
        self.lidar_distance=30.0
        self.max_speed=15.0
        self.max_steps = 1000
        self.step_count = 0
        #调参优化reward
        self.action_repeat = 4
        # 上一拍 RL 下发的转角，用于平滑惩罚（reset 后初值为 0）
        self._prev_steer = 0.0
        # 相邻两次 RL 决策的转角变化惩罚，抑制左右抽搐
        self.steer_delta_coef = 0.08

        # --- Reward 权重 / 终止阈值（路径 A v3 修补：堵 sidewalk / 逆行 / 倒车漏洞）---
        # forward 用 signed_speed_norm（沿车头方向的速度），倒车自动负分
        self.forward_coef = 0.7
        # lane_keep 从「正奖励」改成「penalty」：偏离即扣，越偏越扣
        self.lane_keep_penalty_coef = 1.0
        # 在非 Driving 车道（人行道/草地等）每个 inner step 持续扣
        self.off_road_penalty = 1.0
        # 逆行（倒车 or 跨黄线到对向车道）惩罚比 off_road 更狠
        self.wrong_way_penalty = 2.0
        # 持续越界 N 个 inner step 直接终止 episode（按 0.05s/step 算时间）
        self.off_road_terminate_steps = 30      # 1.5 秒
        self.wrong_way_terminate_steps = 20     # 1.0 秒

        # --- 经典 ACC 控制器参数（内部用，不暴露给 RL）---
        # Constant-Time-Headway: d_safe = v_ego * tau + d_min
        # 前车足够远 → 巡航 v_set；d <= d_min → 强刹至 0；中间段线性插值
        self.acc_v_set = 10.0          # 目标巡航速度 m/s（约 36 km/h，比 NPC 略快）
        self.acc_tau = 1.5             # 时间间距 s
        self.acc_d_min = 5.0           # 最小安全距离 m
        self.acc_kp_throttle = 0.4     # 加速 P 增益
        self.acc_kp_brake = 0.5        # 减速 P 增益
        self.acc_max_throttle = 0.7
        self.acc_lidar_far = 28.0      # lidar_distance >= 此值视为前方空旷
        # v2 反抖动：speed_err 死区 + v_target 一阶低通
        self.acc_deadband = 0.5        # |v_target - v_ego| < 这个值就 coast，不踩油门也不刹车
        self.acc_target_alpha = 0.3    # 越小越平滑，但响应越慢；0.3 = 大约 3 拍后跟上目标
        self._v_target_filt = self.acc_v_set

        # --- LiDAR OpenCV 俯视可视化（仅调试；训练请 debug_lidar=False）---
        # 原因：CARLA 主窗口不渲染点云；用 OpenCV 把传感器坐标系下的 (x前,y左) 投到 2D 图像上即可实时看障碍物分布。
        self.debug_lidar = debug_lidar
        self._lidar_vis_dt = 1.0 / max(lidar_vis_fps, 1e-3)
        self._lidar_vis_last_t = 0.0
        self._lidar_vis_max_points = int(lidar_vis_max_points)
        self._lidar_bev_w = 600
        self._lidar_bev_h = 600
        # 像素/米：6 px/m × 600 px ≈ ±50m 视野，正好覆盖 lidar range
        # 之前 14 px/m 只能看到 ±17m，远处车都被裁掉了
        self._lidar_bev_scale = 6.0
        self._lidar_cv_window_ready = False
        # 仅缓存最新点云的 (N, 2) xy，由 LiDAR 回调线程写、主线程读 → 主线程做 imshow，
        # 避免 Windows 上 OpenCV 窗口跑在 sensor 线程导致「无响应」。
        self._lidar_latest_pts = None
        self._lidar_latest_tags = None
        # CARLA 语义标签里被认为是「ACC 目标」的：14=Car / 15=Truck / 16=Bus / 18=Motorcycle / 19=Bicycle
        # 行人 / Rider 没加进来；如果想让 ACC 也对行人刹车，把 12, 13 加上去
        self._lidar_dyn_tags = np.array([14, 15, 16, 18, 19], dtype=np.uint32)

        # --- 运行时状态的占位默认值（实际值由 _update_state 在每个 step 前刷新）---
        # 不写也能跑，但写上去 IDE / 静态检查 / 直接调试时更稳
        self.speed = 0.0
        self.speed_norm = 0.0
        self.signed_speed = 0.0       # 沿车头方向的速度，倒车为负
        self.signed_speed_norm = 0.0  # 归一化到 [-1, 1]，给 obs / forward reward 用
        self.lateral_norm = 0.0
        self.sin_h = 0.0
        self.cos_h = 1.0
        self.yaw_rate = 0.0
        self.lateral_accel = 0.0
        self.vehicle_location = None
        self.waypoint = None
        # 越界 / 逆行检测的瞬时标志 + 持续步数（用于提前终止）
        self.off_road = False
        self.actual_lane_type = None
        self.wrong_way = False
        self.off_road_steps = 0
        self.wrong_way_steps = 0
        # wrong_way Type B 检测的「参考」lane_id / road_id：
        # 进入新 road 时如果状态合法会被重新锚定，避免跨路口后误判
        self._ref_lane_id = None
        self._ref_road_id = None

    def reset(self, seed=None, options=None):
        if self.vehicle is not None:
            if self.camera is not None:
                self.camera.stop()
                self.camera.destroy()
            if self.collision_sensor is not None:
                self.collision_sensor.stop()
                self.collision_sensor.destroy()
            if self.lidar is not None:
                self.lidar.stop()
                self.lidar.destroy()
            self.vehicle.destroy()
        # 上一回合的 NPC 全部清掉，避免残留堆积
        self._destroy_npcs()

        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find("vehicle.tesla.model3")
        spawn_points = self.world.get_map().get_spawn_points()
        ego_spawn = random.choice(spawn_points)
        self.vehicle = self.world.spawn_actor(vehicle_bp, ego_spawn)
        # ego 之后再放 NPC：用其余 spawn points + 距离 ego 足够远，避免开局穿模/卡死
        self._spawn_npcs(ego_spawn, spawn_points)

        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "800")
        camera_bp.set_attribute("image_size_y", "600")
        camera_bp.set_attribute("fov", "90")

        camera_transform = carla.Transform(
            carla.Location(x=2, z=1)
        )
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)

        self.latest_image = None
        self.camera.listen(lambda img: setattr(self, 'latest_image', img))

        # 升级到 ray_cast_semantic：每个点除 (x,y,z) 外还带 (cos_inc_angle, obj_idx, obj_tag)
        # 这样可以在回调里按 semantic tag 过滤，只把「车辆」当作 ACC 跟车目标，
        # 路灯杆、墙、栅栏等就不会误触发刹车（这是单通道 LiDAR 又无 tag 的最大痛点）。
        lidar_bp = blueprint_library.find("sensor.lidar.ray_cast_semantic")
        # 参数选自 Velodyne VLP-32 + CARLA 社区惯例（carla-roach / TransFuser）
        # range 50m：ACC 用 30m 内距离，再多 20m 给前瞻预判和可视化
        lidar_bp.set_attribute("range", "50")
        # 32 通道是性能/精度甜点；64 不显著提升下游 RL 但 CPU 翻倍
        lidar_bp.set_attribute("channels", "32")
        # FOV +10° / -30°：和真实 Velodyne 一致，下视广覆盖近地，上视看高大障碍
        lidar_bp.set_attribute("upper_fov", "10")
        lidar_bp.set_attribute("lower_fov", "-30")
        # 600k pts/s @ 20Hz = 30k 点 / 圈 / 32 通道 ≈ 0.4° 角分辨率（之前 2.3° 太稀）
        # 30m 处相邻点间距 ≈ 21 cm，一辆车能稳定打到 8~10 个点
        lidar_bp.set_attribute("points_per_second", "600000")
        # 20Hz 比真车快（真车 10Hz），仿真里方便 ACC 快速反应
        lidar_bp.set_attribute("rotation_frequency", "20")

        lidar_transformation = carla.Transform(
            carla.Location(z=2.5)
        )
        self.lidar = self.world.spawn_actor(
            lidar_bp,
            lidar_transformation,
            attach_to=self.vehicle
        )
        self.lidar_distance = 30.0
        self.lidar.listen(self._on_lidar)


        collision_bp = blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),#xyz000,yawpitchroll000
            attach_to=self.vehicle 
        )
        self.collision_happened = False 
        self.collision_sensor.listen(lambda event: setattr(self,'collision_happened', True))

        time.sleep(1.0)
        self.step_count = 0
        self._prev_steer = 0.0
        # ACC 低通滤波器在每个 episode 开头重置回巡航速度，避免上 episode 的余值灌进来
        self._v_target_filt = self.acc_v_set
        # 越界 / 逆行计数器在 episode 开头清零
        self.off_road_steps = 0
        self.wrong_way_steps = 0
        # 先把 _ref_lane_id 置 None，让 _update_state 第一次跑时跳过 wrong_way 比较
        self._ref_lane_id = None
        self._ref_road_id = None

        self._update_state()
        # 用第一帧的 waypoint 作为「合法初始车道」锚点（spawn point 一定在 Driving 上）
        self._ref_lane_id = self.waypoint.lane_id
        self._ref_road_id = self.waypoint.road_id

        self._update_spectator()
        obs = self._get_observation()
        return obs, {}

    def _update_spectator(self):
        """Move default UE spectator behind vehicle so training view tracks resets."""
        if not self.spectator_follow or self.vehicle is None:
            return
        spectator = self.world.get_spectator()
        v_tf = self.vehicle.get_transform()
        fwd = v_tf.get_forward_vector()
        d = self._spectator_distance
        z_up = self._spectator_height
        loc = carla.Location(
            v_tf.location.x - fwd.x * d,
            v_tf.location.y - fwd.y * d,
            v_tf.location.z - fwd.z * d + z_up,
        )
        rot = carla.Rotation(
            pitch=self._spectator_pitch,
            yaw=v_tf.rotation.yaw,
            roll=0.0,
        )
        spectator.set_transform(carla.Transform(loc, rot))

    def step(self, action):
        # 路径 A：RL 只控制 steer，throttle / brake 每个 inner-step 由 ACC 实时计算
        steer = float(action[0])

        total_reward = 0.0
        terminated = False
        truncated = False
        last_throttle = 0.0
        last_brake = 0.0
        for _ in range(self.action_repeat):
            # 用最新观测重算 ACC（前车突然减速也能在 50ms 内反应）
            throttle, brake = self._compute_acc_control()
            last_throttle, last_brake = throttle, brake
            control = carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=brake,
            )
            self.vehicle.apply_control(control)

            time.sleep(0.05)
            self.step_count += 1
            self._update_state()
            self._update_spectator()
            self._pump_lidar_vis()

            # 越界 / 逆行的「持续步数」累加（一旦回到合法状态立即清零）
            self.off_road_steps = self.off_road_steps + 1 if self.off_road else 0
            self.wrong_way_steps = self.wrong_way_steps + 1 if self.wrong_way else 0

            total_reward += self._physics_reward()

            if self.collision_happened:
                terminated = True
                break
            if self.off_road_steps >= self.off_road_terminate_steps:
                # 持续越界过久 → 提前终止 + 一次性大惩罚
                # 这样 PPO 不会让 agent 在人行道上「跑完整个 episode 慢慢回血」
                terminated = True
                total_reward -= 50.0
                break
            if self.wrong_way_steps >= self.wrong_way_terminate_steps:
                terminated = True
                total_reward -= 50.0
                break
            if self.step_count >= self.max_steps:
                truncated = True
                break

        obs = self._get_observation()
        # 路径 A：reward 只关心车道保持；steer shaping 仍保留以抑制方向盘抽搐
        reward = total_reward + self._steer_shaping_penalty(steer)
        self._prev_steer = steer
        info = {
            "speed": float(self.speed),
            "signed_speed": float(self.signed_speed),
            "lidar_d": float(self.lidar_distance),
            "throttle": last_throttle,
            "brake": last_brake,
            "off_road": bool(self.off_road),
            "wrong_way": bool(self.wrong_way),
            "lane_type": str(self.actual_lane_type) if self.actual_lane_type else "None",
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        if self.camera is not None:
            self.camera.stop()
            self.camera.destroy()
            self.camera = None

        if self.collision_sensor is not None:
            self.collision_sensor.stop()
            self.collision_sensor.destroy()
            self.collision_sensor=None
        if self.lidar is not None:
            self.lidar.stop()
            self.lidar.destroy()
            self.lidar=None
        if self.debug_lidar:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
            self._lidar_cv_window_ready = False
        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None
        self._destroy_npcs()


    def _update_state(self):
        #这个function是我根据get_observation优化过来的，因为reward计算也需要调用这些waypoint api所有就在step里面更新全局变量了
        # 占位，之后填充真实的数值状态
        #一共五个observation数值
        #速度
        vel = self.vehicle.get_velocity()
        self.speed = (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5
        self.speed_norm = np.clip(self.speed / self.max_speed, 0.0, 1.0)

        self.vehicle_location = self.vehicle.get_transform().location
        self.waypoint = self.world.get_map().get_waypoint(self.vehicle_location)

        #计算横向偏差,对比于waypoint center,+right.-left
        dx = self.vehicle_location.x - self.waypoint.transform.location.x
        dy = self.vehicle_location.y - self.waypoint.transform.location.y
        #因为carla.transform.rotation return degree, example 45.0 degreee, 45 is only a float number. math.radious transfer degree float into actual degree
        wp_yaw = math.radians(self.waypoint.transform.rotation.yaw)
        lateral = math.cos(wp_yaw) * dy - math.sin(wp_yaw) * dx
        self.lateral_norm = np.clip(lateral / (self.waypoint.lane_width / 2.0), -1.0, 1.0)
        #车头朝哪
        vehicle_yaw = self.vehicle.get_transform().rotation.yaw
        waypoint_yaw = self.waypoint.transform.rotation.yaw
        heading_error = math.radians(vehicle_yaw - waypoint_yaw)
        self.sin_h = math.sin(heading_error)
        self.cos_h = math.cos(heading_error)

        # 横向加速度 a_y = v * yaw_rate（CARLA angular_velocity 单位为 deg/s，转 rad/s）
        # 用途：(1) reward 里加舒适性惩罚 → 抑制「高速猛打方向」
        #       (2) 加进 obs → 模型能感知自己产生的离心力
        # 注意：和 lateral_norm 不重合 —— 停在路边的车 lateral_norm 大但 a_y=0；
        #              开 50 km/h 急转弯的车 lateral_norm 可能很小但 a_y 爆表。
        ang = self.vehicle.get_angular_velocity()
        self.yaw_rate = math.radians(ang.z)
        self.lateral_accel = float(self.speed * self.yaw_rate)

        # === 沿车头方向的速度（egocentric forward velocity）===
        # 关键：不依赖任何车道信息，纯靠 vehicle 自己的 transform。
        # 倒车时这个值是负的 → forward reward 自动负分，不需要额外检测。
        fwd = self.vehicle.get_transform().get_forward_vector()
        self.signed_speed = float(vel.x * fwd.x + vel.y * fwd.y)
        self.signed_speed_norm = float(np.clip(self.signed_speed / self.max_speed, -1.0, 1.0))

        # === Off-road 检测：是不是在 Driving 车道上 ===
        # project_to_road=False 让这次查询不投影 → 真实反映「车下面是什么 lane」
        # lane_type=Any 让 sidewalk / shoulder / parking 这些都能被检测到
        actual_wpt = self.world.get_map().get_waypoint(
            self.vehicle_location,
            project_to_road=False,
            lane_type=carla.LaneType.Any,
        )
        if actual_wpt is None:
            # 完全脱离 OpenDRIVE 元素（草地、地图边界外等）
            self.off_road = True
            self.actual_lane_type = None
        else:
            self.off_road = (actual_wpt.lane_type != carla.LaneType.Driving)
            self.actual_lane_type = actual_wpt.lane_type

        # === Wrong-way 检测：两类逆行 ===
        # Type A: 速度方向 vs 当前最近车道 forward 反向（倒车 / 跑反方向）
        v_horiz = math.hypot(vel.x, vel.y)
        if v_horiz > 0.5:
            lane_fwd = self.waypoint.transform.get_forward_vector()
            cos_v_lane = (vel.x * lane_fwd.x + vel.y * lane_fwd.y) / v_horiz
            type_a_wrong = (cos_v_lane < -0.3)
        else:
            type_a_wrong = False

        # Type B: 同 road 内 lane_id 符号反转（跨过中线到对向车道）
        # 注：lane_id 符号约定是「相对 road 参考线左右」，所以同 road 内符号反 = 跨中线
        # 不同 road 时不能直接比，需要在合法过渡时重新锚定 _ref_*
        if self._ref_lane_id is None:
            type_b_wrong = False
        elif self.waypoint.road_id == self._ref_road_id:
            type_b_wrong = (
                np.sign(self.waypoint.lane_id) != np.sign(self._ref_lane_id)
            )
        else:
            # 进入新 road：如果当前看着合法（没逆行也没越界），把这条新路当作新参考
            type_b_wrong = False
            if not type_a_wrong and not self.off_road:
                self._ref_road_id = self.waypoint.road_id
                self._ref_lane_id = self.waypoint.lane_id

        self.wrong_way = type_a_wrong or type_b_wrong

    def _get_observation(self):
        # 6 m/s² 是「能感受到」的横向加速度量级；超过的极端值被 clip 成 ±1
        lat_acc_norm = float(np.clip(self.lateral_accel / 6.0, -1.0, 1.0))
        # 第 0 维：signed_speed_norm（不是 speed_norm）→ 倒车时为负，模型直接看到方向
        return np.array([
            self.signed_speed_norm,
            self.lateral_norm,
            self.sin_h,
            self.cos_h,
            self.lidar_distance / 30.0,
            lat_acc_norm,
        ], dtype=np.float32)
    def _physics_reward(self):
        """
        路径 A v3 的 reward：堵 v2 的三个漏洞（人行道拿分 / 逆向拿分 / 倒车拿分）。

        v2 漏洞：
        (1) lane_keep 是正奖励 + lateral_norm clip 到 ±1 → 人行道也能拿 0；
        (2) cos_h 比的是「最近车道 forward」，对向车道按其 forward 同向开 → cos_h=+1；
        (3) forward 用 speed_norm * cos_h，speed_norm 永远 ≥ 0 → 倒车 cos_h 也可能 ≈ +1。

        v3 修复：
        (1) lane_keep 改成 penalty：-|lateral_norm| * coef，越偏越扣；
        (2) 加 off_road penalty（_update_state 里用 project_to_road=False 检测）；
        (3) 加 wrong_way penalty（Type A: 速度反向；Type B: 跨黄线 lane_id 符号反转）；
        (4) forward 改用 signed_speed_norm（沿车头方向的速度）→ 倒车自动负分。
        """
        # 主奖励：沿车头方向开多快。倒车 → 负，正常前进 → 正。最大 ±forward_coef。
        forward = self.forward_coef * self.signed_speed_norm
        # 车道居中：永远 ≤ 0；贴中心 0，贴车道边 -coef
        lane_keep = -self.lane_keep_penalty_coef * abs(self.lateral_norm)
        # 越界：脚下不是 Driving lane（人行道 / 草地 / 应急道等）
        off_road_p = -self.off_road_penalty if self.off_road else 0.0
        # 逆行：倒车 / 跨黄线到对向
        wrong_way_p = -self.wrong_way_penalty if self.wrong_way else 0.0
        # 舒适性：横向加速度二次惩罚（a_y=2 → -0.1；a_y=6 → -0.9）
        lat_acc_n = self.lateral_accel / 2.0
        comfort = -0.1 * (lat_acc_n ** 2)

        reward = forward + lane_keep + off_road_p + wrong_way_p + comfort
        if self.collision_happened:
            reward -= 100.0
        return float(reward)

    def _steer_shaping_penalty(self, steer_cmd: float) -> float:
        """
        v1 里有 low_speed_steer 项（车慢就罚打方向）—— ACC 上线后这项变成 bug：
        如果 ACC 因为 lidar 误检而把车刹停，模型反而被惩罚去打方向修正，
        于是「停下不动 + 不打方向」成了局部最优。这里直接删掉。
        现在只剩抖方向盘惩罚（jerk 代理），抑制左右抽搐。
        横向加速度的「猛打方向」惩罚已经移到 _physics_reward 里的 comfort 项。
        """
        smooth = -self.steer_delta_coef * abs(steer_cmd - self._prev_steer)
        return float(smooth)

    def _compute_acc_control(self):
        """
        ACC 控制器 v2：解决 v1 的 3 个老问题：
        (1) 平滑区间宽度跟 v_ego 挂钩 → 低速时 1m 距离变化 = 2 m/s v_target 跳变
            修复：直接用固定的 [d_min, lidar_far] = [5, 28] 这段 23m 做线性插值，
                  1m 距离变化只引起 0.43 m/s v_target 变化。
        (2) 油门/刹车硬切换（speed_err 过零就翻转）→ 零附近反复抖
            修复：deadband = 0.5 m/s，|speed_err| 小于这个就既不油门也不刹车（coast）。
        (3) lidar 自身瞬时抖动直接灌进 v_target → 控制曲线锯齿
            修复：v_target 一阶低通滤波，平滑系数 alpha=0.3。
        """
        v_ego = float(self.speed)
        d_lead = float(self.lidar_distance)

        if d_lead >= self.acc_lidar_far:
            v_target_raw = self.acc_v_set
        elif d_lead <= self.acc_d_min:
            v_target_raw = 0.0
        else:
            denom = max(self.acc_lidar_far - self.acc_d_min, 1e-3)
            ratio = (d_lead - self.acc_d_min) / denom
            v_target_raw = self.acc_v_set * float(np.clip(ratio, 0.0, 1.0))

        # 一阶低通：吃掉 lidar 抖动；reset 时 _v_target_filt 会被置回 acc_v_set
        self._v_target_filt = (
            (1.0 - self.acc_target_alpha) * self._v_target_filt
            + self.acc_target_alpha * v_target_raw
        )
        v_target = self._v_target_filt

        speed_err = v_target - v_ego
        if speed_err > self.acc_deadband:
            throttle = float(np.clip(self.acc_kp_throttle * speed_err, 0.0, self.acc_max_throttle))
            brake = 0.0
        elif speed_err < -self.acc_deadband:
            throttle = 0.0
            brake = float(np.clip(-self.acc_kp_brake * speed_err, 0.0, 1.0))
        else:
            # deadband 区：滑行，避免油门刹车反复切换
            throttle = 0.0
            brake = 0.0
        return throttle, brake


    def _spawn_npcs(self, ego_spawn, spawn_points):
        """在 ego 之外的出生点随机生成 autopilot NPC，离 ego 至少 25 米，避免开局碰撞。"""
        self.npc_vehicles = []
        if self.num_npcs <= 0:
            return
        blueprint_library = self.world.get_blueprint_library()
        all_bps = blueprint_library.filter("vehicle.*")
        # 只要四轮车，避免摩托/自行车 autopilot 行为不稳
        car_bps = [
            bp for bp in all_bps
            if bp.has_attribute("number_of_wheels")
            and int(bp.get_attribute("number_of_wheels")) == 4
        ]
        if not car_bps:
            car_bps = list(all_bps)

        candidates = [sp for sp in spawn_points if sp.location != ego_spawn.location]
        random.shuffle(candidates)

        ex, ey = ego_spawn.location.x, ego_spawn.location.y
        spawned = 0
        for sp in candidates:
            if spawned >= self.num_npcs:
                break
            dx = sp.location.x - ex
            dy = sp.location.y - ey
            if dx * dx + dy * dy < 25.0 * 25.0:
                continue
            bp = random.choice(car_bps)
            if bp.has_attribute("color"):
                color = random.choice(bp.get_attribute("color").recommended_values)
                bp.set_attribute("color", color)
            npc = self.world.try_spawn_actor(bp, sp)
            if npc is None:
                continue
            try:
                npc.set_autopilot(True, self.tm.get_port())
            except Exception:
                npc.destroy()
                continue
            self.npc_vehicles.append(npc)
            spawned += 1

    def _destroy_npcs(self):
        """
        正确的销毁顺序：
        1) 先 set_autopilot(False)，让 Traffic Manager 释放对这些 NPC 的控制；
        2) 用 client.apply_batch_sync 一次性下发 DestroyActor 命令，server 端原子处理并等待返回。
        逐个 actor.destroy() 在 async 模式下是 fire-and-forget，进程退出时会丢命令，留下幽灵车。
        """
        if not self.npc_vehicles:
            return
        for npc in self.npc_vehicles:
            try:
                npc.set_autopilot(False)
            except Exception:
                pass
        try:
            self.client.apply_batch_sync(
                [carla.command.DestroyActor(npc.id) for npc in self.npc_vehicles],
                False,  # async server 模式下不能 do_tick
            )
        except Exception:
            # 兜底：批量失败再退化为单个 destroy
            for npc in self.npc_vehicles:
                try:
                    npc.destroy()
                except Exception:
                    pass
        self.npc_vehicles = []

    def _visualize_lidar_bev(self, points_xy: np.ndarray) -> None:
        """
        传感器坐标：x 前、y 左（CARLA LiDAR 默认）。投到鸟瞰图：车体附近在图像下方，前方朝上。
        降频刷新：避免每次 LiDAR 回调都 imshow，减轻 CPU/GIL 对 CARLA 的影响。
        """
        now = time.monotonic()
        if now - self._lidar_vis_last_t < self._lidar_vis_dt:
            return
        self._lidar_vis_last_t = now

        h, w = self._lidar_bev_h, self._lidar_bev_w
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cx, cy = w // 2, h - 40
        scl = self._lidar_bev_scale

        n = points_xy.shape[0]
        if n > self._lidar_vis_max_points:
            idx = np.random.choice(n, self._lidar_vis_max_points, replace=False)
            pts = points_xy[idx]
        else:
            pts = points_xy

        x = pts[:, 0].astype(np.float64)
        y = pts[:, 1].astype(np.float64)
        u = (cx + y * scl).astype(np.int32)
        v = (cy - x * scl).astype(np.int32)
        m = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        img[v[m], u[m]] = (0, 255, 128)

        cv2.line(img, (cx, cy), (cx, cy - int(25 * scl)), (200, 200, 200), 1)
        cv2.putText(
            img,
            "Forward",
            (cx - 28, cy - int(26 * scl)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        if not self._lidar_cv_window_ready:
            cv2.namedWindow("Carla LiDAR BEV", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Carla LiDAR BEV", w, h)
            self._lidar_cv_window_ready = True
        cv2.imshow("Carla LiDAR BEV", img)
        cv2.waitKey(1)

    def _on_lidar(self, data):
        # 此回调跑在 CARLA sensor 工作线程里。绝对不要在这里调 cv2.imshow / waitKey，
        # 否则窗口的 Win32 消息泵不在主线程会被系统判为「无响应」。
        #
        # ray_cast_semantic 每个点 = (x, y, z, cos_inc_angle, obj_idx, obj_tag)
        # 4 × float32 + 2 × uint32 = 24 字节；用结构化 dtype 一次性解析。
        # 为什么要 tag 过滤：原来 ray_cast 单通道扫到任何障碍（路灯/墙/栅栏）都
        # 会缩小 lidar_distance → ACC 误刹车。现在只对 DYNAMIC_TAGS 里的物体
        # 计算「跟车距离」，路边静态物完全不影响 ACC。
        dt = np.dtype([
            ("x", np.float32), ("y", np.float32), ("z", np.float32),
            ("cos_inc", np.float32),
            ("idx", np.uint32), ("tag", np.uint32),
        ])
        pts = np.frombuffer(data.raw_data, dtype=dt)

        if self.debug_lidar and pts.shape[0] > 0:
            self._lidar_latest_pts = np.stack([pts["x"], pts["y"]], axis=1).astype(np.float32)
            self._lidar_latest_tags = pts["tag"].copy()

        # CARLA 0.9.12+ 语义标签：14=Car, 15=Truck, 16=Bus, 18=Motorcycle, 19=Bicycle
        # 12=Pedestrian / 13=Rider 也可以加进来当 ACC 目标（更安全），按需打开
        x = pts["x"]; y = pts["y"]; tag = pts["tag"]
        # 前向 60° 锥（|y| < x * tan(30°) ≈ 0.577*x），剔除地面附近 z < -1.5（车顶高度约 z=0）
        cone = (x > 0.5) & (np.abs(y) < x * 0.577) & (pts["z"] > -1.5)
        is_vehicle = np.isin(tag, self._lidar_dyn_tags)
        front = cone & is_vehicle
        if np.any(front):
            d = np.sqrt(x[front] ** 2 + y[front] ** 2)
            self.lidar_distance = float(np.min(d))
        else:
            self.lidar_distance = 30.0

    def _pump_lidar_vis(self):
        """在主线程调用：把回调线程缓存下来的点云画出来。"""
        if not self.debug_lidar:
            return
        pts = self._lidar_latest_pts
        if pts is None or pts.shape[0] == 0:
            return
        self._visualize_lidar_bev(pts)