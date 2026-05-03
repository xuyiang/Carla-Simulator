import time
import carla

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.load_world("Town02")

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find("vehicle.tesla.model3")

    spawn_points=world.get_map().get_spawn_points()
    spawn_point=spawn_points[0]

    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print("Spawned:", vehicle)

    # Move the spectator camera to look at the spawned vehicle
    spectator = world.get_spectator()
    spectator_transform = carla.Transform(
        spawn_point.location + carla.Location(z=30),   # 30m above the car
        carla.Rotation(pitch=-90)                       # look straight down
    )
    spectator.set_transform(spectator_transform)

    try:
        time.sleep(15)

    finally:
        vehicle.destroy()
        print("Destroyed Vehicle")


if __name__ == "__main__":
    main()

