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
        lidar_vis_fps: float = 12.0,
        lidar_vis_max_points: int = 8000,
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
        # 方案 A：第 0 维 long ∈ [-1, 1]，正数→油门，负数→刹车（互斥，物理天然合理）
        # 第 1 维 steer ∈ [-1, 1]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([ 1.0,  1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
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
        # 低速时惩罚大方向盘：系数 × |steer| × (1 - speed_norm)，越慢权重越大
        self.low_speed_steer_coef = 0.25
        # 相邻两次 RL 决策的转角变化惩罚，抑制左右抽搐
        self.steer_delta_coef = 0.08

        # --- 避障 / 刹车 shaping ---
        # 距离前方障碍 < safe_d 时开始介入；< danger_d 时全权重
        self.brake_safe_dist = 12.0
        self.brake_danger_dist = 5.0
        # 近障碍仍踩油门 → 大惩罚；近障碍踩刹车 → 小奖励
        self.throttle_near_obstacle_coef = 1.5
        self.brake_near_obstacle_coef = 0.5

        # --- LiDAR OpenCV 俯视可视化（仅调试；训练请 debug_lidar=False）---
        # 原因：CARLA 主窗口不渲染点云；用 OpenCV 把传感器坐标系下的 (x前,y左) 投到 2D 图像上即可实时看障碍物分布。
        self.debug_lidar = debug_lidar
        self._lidar_vis_dt = 1.0 / max(lidar_vis_fps, 1e-3)
        self._lidar_vis_last_t = 0.0
        self._lidar_vis_max_points = int(lidar_vis_max_points)
        self._lidar_bev_w = 480
        self._lidar_bev_h = 480
        self._lidar_bev_scale = 14.0  # 像素/米，越大视野越「_zoom in」
        self._lidar_cv_window_ready = False
        # 仅缓存最新点云的 (N, 2) xy，由 LiDAR 回调线程写、主线程读 → 主线程做 imshow，
        # 避免 Windows 上 OpenCV 窗口跑在 sensor 线程导致「无响应」。
        self._lidar_latest_pts = None

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

        lidar_bp = blueprint_library.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("range", "30")            # 探测距离 30 米
        lidar_bp.set_attribute("points_per_second", "10000")  # 每秒点数
        lidar_bp.set_attribute("channels", "1")          # 只用一层（水平扫描），省资源
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

        self._update_state()
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

    def step(self,action):
        # 方案 A：long ∈ [-1, 1]，正→油门，负→刹车，互斥
        long_cmd = float(action[0])
        steer    = float(action[1])
        if long_cmd >= 0.0:
            throttle = long_cmd
            brake = 0.0
        else:
            throttle = 0.0
            brake = -long_cmd

        control = carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
        )

        total_reward = 0.0
        terminated = False
        truncated = False
        for _ in range (self.action_repeat):

            self.vehicle.apply_control(control)

            time.sleep(0.05)
            #observation前先计数
            self.step_count+=1
            #计算调用一遍
            self._update_state()
            #确保我能看到车跟随视角
            self._update_spectator()
            # 主线程刷 LiDAR BEV 窗口，内部 12fps 限频；debug_lidar=False 时直接 return
            self._pump_lidar_vis()

            total_reward += self._physics_reward()

            if self.collision_happened:
                terminated = True
                break
            if self.step_count>=self.max_steps:
                truncated  = True
                break

        obs = self._get_observation()
        # 转角 / 刹车相关惩罚：macro-step 只算一次（repeat 内动作不变）
        reward = (
            total_reward
            + self._steer_shaping_penalty(steer)
            + self._brake_shaping(long_cmd)
        )
        self._prev_steer = steer
        info = {}

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

    def _get_observation(self):
        return np.array([self.speed_norm,self.lateral_norm,self.sin_h,self.cos_h,self.lidar_distance/30.0], dtype=np.float32)
    def _physics_reward(self):
        """每个仿真小步累计：速度 / 横向 / 碰撞。"""
        reward = self.speed_norm * 0.5
        reward -= abs(self.lateral_norm) * 0.5
        if self.collision_happened:
            reward -= 50.0
        return float(reward)

    def _steer_shaping_penalty(self, steer_cmd: float) -> float:
        """
        每条 RL 决策一次：低速打方向惩罚 + 与上一拍转角的平滑惩罚。
        返回加到 total_reward 上的增量（通常为负）。
        """
        low_speed = -self.low_speed_steer_coef * abs(steer_cmd) * (1.0 - self.speed_norm)
        smooth = -self.steer_delta_coef * abs(steer_cmd - self._prev_steer)
        return float(low_speed + smooth)

    def _brake_shaping(self, long_cmd: float) -> float:
        """
        前方近障碍时：踩油门 → 强惩罚；踩刹车 → 弱奖励。
        距离 >= safe_d 时本项为 0，不影响纯巡航场景的 reward 分布。
        """
        d = self.lidar_distance
        if d >= self.brake_safe_dist:
            return 0.0
        # proximity ∈ [0, 1]，越近越接近 1
        denom = max(self.brake_safe_dist - self.brake_danger_dist, 1e-3)
        proximity = float(np.clip((self.brake_safe_dist - d) / denom, 0.0, 1.0))
        if long_cmd > 0.0:
            return -self.throttle_near_obstacle_coef * proximity * long_cmd
        else:
            # long_cmd <= 0 → 刹车强度 = -long_cmd
            return self.brake_near_obstacle_coef * proximity * (-long_cmd)

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
        points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
        if self.debug_lidar and points.shape[0] > 0:
            self._lidar_latest_pts = points[:, :2].copy()

        front = points[
            (points[:, 0] > 0) & (np.abs(points[:, 1]) < points[:, 0] * 0.577)
        ]
        if len(front) > 0:
            distance = np.sqrt(front[:, 0] ** 2 + front[:, 1] ** 2)
            self.lidar_distance = float(np.min(distance))
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