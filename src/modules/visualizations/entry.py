from manim import *
import torch
import logging
import manim
from pyglet.resource import add_font

from src.ai.variants.exploration.data_filtering import data_filtering_redundant_datapoints
from src.ai.variants.exploration.exploration_autonomous_policy import augment_data_raw_heuristic, \
    augment_data_cheating_heuristic
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


class IntroScene(Scene):
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
    datapoints_coordinates_map = storage.get_datapoints_coordinates_map()

    DISTANCE_SCALE = 1
    RADIUS = 0.2
    global DEBUG_ARRAY
    global storage_global

    # DEBUG_ARRAY = [8, 0]
    storage_global = storage

    # index1 = 5
    # index2 = 15
    # dp1 = storage.get_datapoint_by_index(index1)["name"]
    # dp2 = storage.get_datapoint_by_index(index2)["name"]
    # print(dp1, dp2)
    # dist = storage.get_datapoints_real_distance(dp1, dp2)
    # conns = storage.get_datapoint_adjacent_connections(dp1)
    # conns = [storage.get_datapoint_index_by_name(x["end"]) for x in conns]
    # print(dist)
    # print(conns)

    # storage_raw: StorageSuperset2 = StorageSuperset2()
    # random_walk_datapoints = storage.get_raw_environment_data()
    # random_walk_datapoints_names = storage.get_all_datapoints()
    #
    # storage_raw.incorporate_new_data(random_walk_datapoints, [])
    #
    # new_connnections = augment_data_cheating_heuristic(storage_raw, random_walk_datapoints_names)
    # total_connections_found = []
    # total_connections_found.extend(new_connnections)

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
    scene = IntroScene()

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


def build_test_scene():
    scene = IntroScene()
    circle = Circle()
    scene.add(circle)
    return scene


def visualization_collected_data_photo(storage: StorageSuperset2):
    manim_configs_png()
    scene = IntroScene()
    scene = build_datapoints_topology(scene, storage)
    scene.render()

    pass


DEBUG_ARRAY = None
storage_global: StorageSuperset2 = None
