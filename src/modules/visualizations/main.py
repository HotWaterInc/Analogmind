from manim import *
from pyglet.window import key
from manim.opengl import *
import logging
import manim
from src.modules.data_handlers.ai_data_handle import read_data_from_file, read_other_data_from_file, CollectedDataType
import threading
from src.autoencoder.autoencoder import Autoencoder, lee_direction_step

RENDERER = manim.RendererType.OPENGL

class IntroScene(Scene):
    def on_key_press(self, symbol, modifiers):
        """Called each time a key is pressed."""
        super().on_key_press(symbol, modifiers)


def find_connections(datapoint_name, connections_data):
    all_cons = []
    for connection in connections_data:
        start = connection["start"]
        end = connection["end"]
        distance = connection["distance"]
        direction = connection["direction"]
        if start == datapoint_name:
            all_cons.append((start, end, distance, direction))
        # if end == datapoint_name:
        #     direction[0] = -direction[0]
        #     direction[1] = -direction[1]
        #     all_cons.append((end, start, distance, direction))

    return all_cons

def add_connections(scene, connections_data, mapped_data, distance_scale):
    for connection in connections_data:
        start = connection["start"]
        end = connection["end"]
        distance = connection["distance"]
        direction = connection["direction"]
        x_start, y_start = mapped_data[start]
        x_end, y_end = mapped_data[end]
        x_dir, y_dir = direction
        line = Line(start=x_start*distance_scale*RIGHT + y_start*distance_scale*UP, end=x_end*distance_scale*RIGHT + y_end*distance_scale*UP, color=WHITE)
        scene.add(line)

def add_vectors(scene, mapped_data, distance_scale, target_name):

    pass

def build_scene():
    scene = IntroScene()

    json_data = read_data_from_file(CollectedDataType.Data8x8)
    connections_data = read_other_data_from_file("data8x8_connections.json")

    # print(json_data)
    # print(connections_data)
    mapped_data = {}
    queue = []
    queue.append((json_data[0]["name"], 0, 0))

    while not len(queue) == 0:
        current = queue.pop(0)
        if current[0] in mapped_data:
            continue
        name = current[0]
        x = current[1]
        y = current[2]
        mapped_data[name] = (x, y)
        print("iterating", name, x, y)

        connections = find_connections(name, connections_data)

        for connection in connections:
            end_point = connection[1]
            if end_point in mapped_data:
                # check if coordinates match, extra safety check
                x_end, y_end = mapped_data[end_point]
                distance = connection[2]
                x_dir, y_dir = connection[3]
                x_start, y_start = x, y
                if x_start + x_dir != x_end or y_start + y_dir != y_end:
                    print(f"Error: {name} to {end_point} is not correct")
            else:
                x_dir, y_dir = connection[3]
                end_name = connection[1]
                queue.append((end_name, x + x_dir, y + y_dir))


    distance_scale = 1
    for key in mapped_data:
        x, y = mapped_data[key]
        print(key, x, y)
        circ = Circle(radius=0.2, color=WHITE)
        circ.move_to(x*distance_scale*RIGHT + y*distance_scale*UP)
        scene.add(circ)

    add_connections(scene, connections_data, mapped_data, distance_scale)
    add_vectors(scene, mapped_data, distance_scale, "2_2")



    # circ = Circle()
    # scene.play(FadeIn(circ))
    scene.interactive_embed()
    return scene

def run_opengl_configs():
    config.renderer = RENDERER
    print(f"{config.renderer = }")

    # config.disable_caching = True
    config.preview = True
    config.write_to_movie = False

    config.input_file = "main.py"

    # mutes manim logger
    logger.setLevel(logging.WARNING)

def run_opengl_scene(scene):
    scene.render()

if __name__ == "__main__":
    run_opengl_configs()
    scene = build_scene()
    # run_opengl_scene(scene)