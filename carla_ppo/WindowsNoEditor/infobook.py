import carla
import time
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)


world = client.get_world()
print("Map name:", world.get_map().name)
print("Number of actors:", len(world.get_actors()))
print("Available vehicles:", [bp.id for bp in world.get_blueprint_library().filter("vehicle.*")])
print("Spawn points:", len(world.get_map().get_spawn_points()))
print("Weather:", world.get_weather())