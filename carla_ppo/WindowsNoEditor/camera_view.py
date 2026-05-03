import carla
import time
import random
import numpy as np
import cv2


#用于处理我遇到的线程冲突的问题定义的全局变量
latest_image = None
#用于处理camera传回数据的function
def process_image(image):
    global latest_image   # 这里我漏掉了导致mian里面一直以为latest_image==None
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    latest_image = array[:, :, :3]

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    world = client.load_world("Town02")
    print("Loaded Town02")

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find("vehicle.tesla.model3")
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print("Spawned:", vehicle)

    vehicle.set_autopilot(True)
    print("自动驾驶已开启")

    #这里是传感器模块，attach到了车上
    camera_bp=blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x","800")
    camera_bp.set_attribute("image_size_y","600")
    camera_bp.set_attribute("fov","90")

    camera_transform = carla.Transform(
        carla.Location(x=2,z=1)

    )
    camera=world.spawn_actor(camera_bp,camera_transform,attach_to=vehicle)

    camera.listen(process_image)


    try:
        for _ in range(600):
            if latest_image is not None:
                cv2.imshow("Camera View",latest_image)
                cv2.waitKey(1)
            time.sleep(0.05)
    finally:
        camera.stop()
        camera.destroy()
        vehicle.set_autopilot(False)
        vehicle.destroy()
        cv2.destroyAllWindows()
        



if __name__ == "__main__":
    main()