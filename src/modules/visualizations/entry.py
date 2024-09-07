import numpy as np
from manim import *
import torch
import logging
import manim
from pyglet.resource import add_font
from scipy.interpolate import griddata

from src.ai.variants.exploration.data_augmentation import load_storage_with_base_data
from src.ai.variants.exploration.data_filtering import data_filtering_redundant_datapoints
from src.ai.variants.exploration.exploration_autonomous_policy import augment_data_raw_heuristic, \
    augment_data_cheating_heuristic
from src.ai.variants.exploration.inference_policy import calculate_positions_manifold_distance
from src.ai.variants.exploration.networks.abstract_base_autoencoder_model import BaseAutoencoderModel
from src.ai.variants.exploration.utils import find_frontier_all_datapoint_and_direction
from src.modules.save_load_handlers.data_handle import read_data_from_file, read_other_data_from_file, \
    CollectedDataType
from src.modules.save_load_handlers.ai_models_handle import load_latest_ai, load_manually_saved_ai, AIType
from src.ai.runtime_data_storage.storage import Storage, Coords
from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2
from src.ai.runtime_data_storage.storage import RawConnectionData, RawEnvironmentData
from src.utils import array_to_tensor

OPENGL_RENDERER = manim.RendererType.OPENGL
storage: Storage = Storage()
storage_superset2: StorageSuperset2 = StorageSuperset2()


class Scene3D(ThreeDScene):
    def on_key_press(self, symbol, modifiers):
        """Called each time a key is pressed."""
        super().on_key_press(symbol, modifiers)


class Scene2D(Scene):
    def on_key_press(self, symbol, modifiers):
        """Called each time a key is pressed."""
        super().on_key_press(symbol, modifiers)


def add_mobjects_found_adjacencies(scene: Scene, model: BaseAutoencoderModel, datapoints: List[RawEnvironmentData],
                                   mapped_data: Dict, distance_scale: float):
    """
    Adds lines between the points found as adjacent
    (similar to add_mobjects_connections, but it's based on running the model instead of using the raw data)
    """
    global storage
    length = len(datapoints)
    average_distance_true_adjacent = get_model_distance_degree1(model, storage, verbose=False)

    # check for each pair if adjacent (by network)
    for start in range(length):
        for end in range(start + 1, length):
            start_data = array_to_tensor(np.array(datapoints[start]["data"]))
            end_data = array_to_tensor(np.array(datapoints[end]["data"]))

            start_embedding = model.encoder_inference(start_data.unsqueeze(0))
            end_embedding = model.encoder_inference(end_data.unsqueeze(0))

            distance_from_embeddings = torch.norm((start_embedding - end_embedding), p=2).item()

            # gets coords + draws connections
            if distance_from_embeddings < average_distance_true_adjacent * 1.25:
                x_start, y_start = mapped_data[datapoints[start]["name"]]["x"], mapped_data[datapoints[start]["name"]][
                    "y"]
                x_end, y_end = mapped_data[datapoints[end]["name"]]["x"], mapped_data[datapoints[end]["name"]]["y"]
                line = Line(start=x_start * distance_scale * RIGHT + y_start * distance_scale * UP,
                            end=x_end * distance_scale * RIGHT + y_end * distance_scale * UP, color=WHITE)
                scene.add(line)


def add_frontier_connections(scene: Scene, connections_data: List[RawConnectionData], mapped_data: Dict,
                             distance_scale: float) -> None:
    """
    Adds lines between the points in the connections_data
    """
    global DEBUG_ARRAY
    for connection in connections_data:
        start = connection["start"]
        dx = connection["direction"][0]
        dy = connection["direction"][1]

        x_start, y_start = mapped_data[start]["x"], mapped_data[start]["y"]
        x_end, y_end = x_start + dx, y_start + dy

        index = storage_global.get_datapoint_index_by_name(start)
        if DEBUG_ARRAY != None and index not in DEBUG_ARRAY:
            continue

        line = Line(start=x_start * distance_scale * RIGHT + y_start * distance_scale * UP,
                    end=x_end * distance_scale * RIGHT + y_end * distance_scale * UP, color=BLUE, stroke_width=1)
        scene.add(line)


def add_null_connections(scene: Scene, connections_data: List[RawConnectionData], mapped_data: Dict,
                         distance_scale: float) -> None:
    """
    Adds lines between the points in the connections_data
    """
    global DEBUG_ARRAY, storage_global
    for connection in connections_data:
        start = connection["start"]
        key = connection["start"]
        dx = connection["direction"][0]
        dy = connection["direction"][1]

        x_start, y_start = mapped_data[start]["x"], mapped_data[start]["y"]
        x_end, y_end = x_start + dx, y_start + dy

        index = storage_global.get_datapoint_index_by_name(key)
        if DEBUG_ARRAY != None and index not in DEBUG_ARRAY:
            continue

        line = Line(start=x_start * distance_scale * RIGHT + y_start * distance_scale * UP,
                    end=x_end * distance_scale * RIGHT + y_end * distance_scale * UP, color=YELLOW, stroke_width=1)
        scene.add(line)


def add_mobjects_connections(scene: Scene, connections_data: List[RawConnectionData], mapped_data: Dict,
                             distance_scale: float) -> None:
    """
    Adds lines between the points in the connections_data
    """
    for connection in connections_data:
        start = connection["start"]
        end = connection["end"]
        if start not in mapped_data or end not in mapped_data:
            continue

        index = storage_global.get_datapoint_index_by_name(start)
        if DEBUG_ARRAY != None and index not in DEBUG_ARRAY:
            continue

        x_start, y_start = mapped_data[start]["x"], mapped_data[start]["y"]
        x_end, y_end = mapped_data[end]["x"], mapped_data[end]["y"]

        line = Line(start=x_start * distance_scale * RIGHT + y_start * distance_scale * UP,
                    end=x_end * distance_scale * RIGHT + y_end * distance_scale * UP, color=WHITE, stroke_width=1)
        scene.add(line)


def add_mobjects_datapoints(scene, mapped_data: Dict, distance_scale: float,
                            radius: float, constrained: bool = False) -> None:
    """
    Adds circles to the scene representing the data points, based on the built coordinates map
    """
    global DEBUG_ARRAY, storage_global
    for key in mapped_data:
        x, y = mapped_data[key]["x"], mapped_data[key]["y"]
        circ = Circle(radius=radius, color=WHITE)
        circ.move_to(x * distance_scale * RIGHT + y * distance_scale * UP)

        # adds a text with the name of the post in the circle
        index = storage_global.get_datapoint_index_by_name(key)
        text_string = f"{index}"
        text = Text(text_string, font_size=8)
        text.move_to(circ.get_center())

        index = storage_global.get_datapoint_index_by_name(key)
        if DEBUG_ARRAY != None and index not in DEBUG_ARRAY and constrained == True:
            continue

        scene.add(text)
        scene.add(circ)


def build_datapoints_topology(scene, storage: StorageSuperset2):
    storage.build_sparse_datapoints_coordinates_map_based_on_xy(percent=1)
    storage.recenter_datapoints_coordinates_map()
    datapoints_coordinates_map = storage.get_datapoints_coordinates_map()

    DISTANCE_SCALE = 1
    RADIUS = 0.2
    global DEBUG_ARRAY
    global storage_global

    # DEBUG_ARRAY = [8, 0]
    storage_global = storage

    # datapoints and normal connections
    add_mobjects_datapoints(scene, datapoints_coordinates_map, DISTANCE_SCALE, RADIUS, constrained=True)
    connections = storage.get_all_connections_only_datapoints()
    add_mobjects_connections(scene, connections, datapoints_coordinates_map,
                             DISTANCE_SCALE)

    # add null connections
    null_connections = storage.get_all_connections_null_data()
    add_null_connections(scene, null_connections, datapoints_coordinates_map, DISTANCE_SCALE)

    # add possible directions
    possible_directions_connections = find_frontier_all_datapoint_and_direction(
        storage=storage,
        return_first=False,
        starting_point=None
    )
    add_frontier_connections(scene, possible_directions_connections, datapoints_coordinates_map, DISTANCE_SCALE)

    return scene


def build_scene_autoencoded_permuted():
    scene = Scene2D()

    global storage_superset2
    permutor = load_manually_saved_ai("permutor_deshift_working.pth")
    storage_superset2.set_transformation(permutor)
    storage_superset2.load_raw_data_from_others("data8x8_rotated20.json")
    storage_superset2.load_raw_data_connections_from_others("data8x8_connections.json")
    storage_superset2.normalize_all_data_super()
    storage_superset2.tanh_all_data()

    storage_superset2.build_permuted_data_raw_with_thetas()
    storage_superset2.build_permuted_data_random_rotations()

    autoencoder = load_manually_saved_ai("autoencodPerm10k_(7).pth")
    storage_superset2.build_datapoints_coordinates_map()
    # quality of life, centered coords at 0,0
    storage_superset2.recenter_datapoints_coordinates_map()
    datapoints_coordinates_map = storage_superset2.get_datapoints_coordinates_map()

    distance_scale = 1
    radius = 0.2

    add_mobjects_datapoints(scene, datapoints_coordinates_map, distance_scale, radius)
    add_mobjects_vectors_pathfinding_super_ab(scene, autoencoder, storage_superset2, datapoints_coordinates_map,
                                              distance_scale,
                                              "0_0", 1,
                                              1)
    return scene


def manim_configs_png():
    config.renderer = manim.RendererType.CAIRO

    config.disable_caching = True
    config.preview = True
    config.input_file = "entry.py"

    # mutes manim logger
    logger.setLevel(logging.WARNING)


def manim_configs_opengl():
    config.renderer = OPENGL_RENDERER
    print(f"{config.renderer = }")

    config.disable_caching = True
    config.preview = True
    config.write_to_movie = False

    config.input_file = "entry.py"

    # mutes manim logger
    logger.setLevel(logging.WARNING)


def run_opengl_scene(scene):
    scene.interactive_embed()


def build_3d_mse(scene):
    scene.set_camera_orientation(phi=45 * DEGREES, theta=-45 * DEGREES)

    # manifold_network: BaseAutoencoderModel = load_manually_saved_ai("manifold_network_2048_1_0.03_0.03.pth")
    manifold_network: BaseAutoencoderModel = load_manually_saved_ai("manifold_network_normal.pth")
    storage = StorageSuperset2()
    load_storage_with_base_data(
        storage=storage,
        datapoints_filename="(1)_datapoints_autonomous_walk.json",
        connections_filename="(1)_connections_autonomous_walk_augmented_filled.json"
    )

    target_x, target_y = -1, 2
    target_name = storage.get_closest_datapoint_to_xy(target_x, target_y)

    def calculate_coords(name):
        coords = storage.get_datapoint_metadata_coords(name)
        x, y = coords[0], coords[1]
        z = 0
        z = calculate_positions_manifold_distance(
            current_name=name,
            target_name=target_name,
            manifold_network=manifold_network,
            storage=storage
        )
        z = z * 5

        return x, y, z

    datapoints_names = storage.get_all_datapoints()
    print("zipping")
    x, y, z = zip(*[calculate_coords(name) for name in datapoints_names])

    # Add axes
    axes = ThreeDAxes(
        x_length=6,
        y_length=6,
        z_length=6
    )

    z_norm = (np.array(z) - min(z)) / (max(z) - min(z))
    scatter = VGroup(
        *[Dot3D(point=[x[i], y[i], z[i]], radius=0.05,
                color=color_gradient([BLUE, GREEN, YELLOW, RED], 101)[int(z_norm[i] * 100)]
                ) for i in
          range(len(x))])

    print("scatter done")

    lines = VGroup(
        *[Line(start=[x[i], y[i], 0], end=[x[i], y[i], z[i] - 0.05], color=WHITE, stroke_width=1, stroke_opacity=0.5)
          for i in range(len(x))]
    )

    print("lines done")

    scene.add(axes, scatter, lines)

    # Add everything to the scene

    # Rotate the camera
    # scene.begin_ambient_camera_rotation(rate=0.1)
    # scene.wait(10)


def build_test_scene():
    # manim_configs_opengl()
    manim_configs_png()
    scene = Scene3D()
    build_3d_mse(scene)
    scene.render()
    # run_opengl_scene(scene)


def visualization_collected_data_photo(storage: StorageSuperset2):
    manim_configs_png()
    scene = Scene2D()
    scene = build_datapoints_topology(scene, storage)
    scene.render()
    pass


DEBUG_ARRAY = None
storage_global: StorageSuperset2 = None
