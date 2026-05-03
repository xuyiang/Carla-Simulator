import time
import carla
import random

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    world = client.load_world("Town02")
    print("Loaded Town02")
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find("vehicle.tesla.model3")

    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)   # randomly pick one spawn point

    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print("Spawned:", vehicle)

    # Enable built-in autopilot (CARLA Traffic Manager controls the car)
    vehicle.set_autopilot(True)
    print("Autopilot ON — watch the car drive!")

    spectator = world.get_spectator()

    try:
        # Follow the car for 60 seconds, updating camera every 0.05s
        for _ in range(1200):
            transform = vehicle.get_transform()

            # Place spectator 8m behind and 4m above the car, in the car's local frame
            fwd = transform.get_forward_vector()
            offset = carla.Location(x=-8 * fwd.x, y=-8 * fwd.y, z=4)
            spectator_transform = carla.Transform(
                transform.location + offset,
                carla.Rotation(pitch=-15, yaw=transform.rotation.yaw)
            )
            spectator.set_transform(spectator_transform)

            # Print speed every 40 frames (~2 seconds)
            if _ % 40 == 0:
                vel = vehicle.get_velocity()
                speed_kmh = (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5 * 3.6
                loc = transform.location
                print(f"  t={_*0.05:.1f}s  speed={speed_kmh:.1f} km/h  "
                      f"pos=({loc.x:.1f}, {loc.y:.1f})")

            time.sleep(0.05)

    finally:
        vehicle.set_autopilot(False)
        vehicle.destroy()
        print("Destroyed vehicle")

if __name__ == "__main__":
    main()
