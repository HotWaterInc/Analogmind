from manim import *
import logging
import manim
from src.modules.data_handlers.ai_data_handle import read_data_from_file, read_other_data_from_file, CollectedDataType
from src.ai.models.autoencoder import Autoencoder, lee_improved_direction_step, lee_direction_step_second_degree
from src.modules.data_handlers.ai_models_handle import load_latest_ai, load_manually_saved_ai, AIType
from src.ai.runtime_data_storage.storage import Storage

RENDERER = manim.RendererType.OPENGL
storage = None


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
        line = Line(start=x_start * distance_scale * RIGHT + y_start * distance_scale * UP,
                    end=x_end * distance_scale * RIGHT + y_end * distance_scale * UP, color=WHITE)
        scene.add(line)


def add_vectors_lee(scene, mapped_data, distance_scale, target_name, autoencoder: Autoencoder, json_data,
                    connections_data):
    print("Adding vectors to target", target_name)
    for key in mapped_data:
        if key == target_name:
            continue

        closest_point = lee_direction_step_second_degree(autoencoder, key, target_name, json_data, connections_data)

        x_current, y_current = mapped_data[key]
        x_target, y_target = mapped_data[closest_point]
        scene.add(Arrow(start=x_current * distance_scale * RIGHT + y_current * distance_scale * UP,
                        end=x_target * distance_scale * RIGHT + y_target * distance_scale * UP, color=RED,
                        stroke_width=2, max_tip_length_to_length_ratio=0.1))


def add_vectors_lee_improved(scene, mapped_data, distance_scale, target_name, autoencoder: Autoencoder, json_data,
                             connections_data):
    print("Adding vectors to target", target_name)
    for key in mapped_data:
        if key == target_name:
            continue

        closest_points_names = lee_improved_direction_step(autoencoder, key, target_name, json_data, connections_data)

        for point in closest_points_names:
            closest_point = point
            x_current, y_current = mapped_data[key]
            x_target, y_target = mapped_data[closest_point]
            scene.add(Arrow(start=x_current * distance_scale * RIGHT + y_current * distance_scale * UP,
                            end=x_target * distance_scale * RIGHT + y_target * distance_scale * UP, color=RED,
                            stroke_width=2, max_tip_length_to_length_ratio=0.1))


def build_scene():
    scene = IntroScene()
    typeai = AIType.Autoencoder

    global storage
    storage = Storage()
    storage.load_raw_data(CollectedDataType.Data8x8)
    storage.normalize_all_data()

    # autoencoder = load_manually_saved_ai("autoenc_8x8_old_training.pth")
    autoencoder = load_latest_ai(typeai)

    json_data = storage.get_raw_env_data()
    connections_data = storage.get_raw_connections_data()

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
        # print("iterating", name, x, y)

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
        # print(key, x, y)
        circ = Circle(radius=0.2, color=WHITE)
        circ.move_to(x * distance_scale * RIGHT + y * distance_scale * UP)
        # adds a text with the name of the post in the circle
        text = Text(key, font_size=8)
        text.move_to(circ.get_center())

        scene.add(text)
        scene.add(circ)

    # add_connections(scene, connections_data, mapped_data, distance_scale)
    # add_vectors_lee_improved(scene, mapped_data, distance_scale, "0_0", autoencoder, json_data, connections_data)
    add_vectors_lee(scene, mapped_data, distance_scale, "0_0", autoencoder, json_data, connections_data)

    scene.interactive_embed()
    return scene


def run_opengl_configs():
    config.renderer = RENDERER
    print(f"{config.renderer = }")

    config.disable_caching = True
    config.preview = True
    config.write_to_movie = False

    config.input_file = "entry.py"

    # mutes manim logger
    logger.setLevel(logging.WARNING)


def run_opengl_scene(scene):
    scene.render()


def change_scene(scene):
    scene.play(Create(Square()))
    scene.interactive_embed()


def run_visualization():
    run_opengl_configs()
    scene = build_scene()


def myfunc1():
    print("Hello from a thread")


if __name__ == "__main__":
    run_visualization()
