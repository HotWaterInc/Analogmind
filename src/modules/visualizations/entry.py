from manim import *
import torch
import logging
import manim
from src.modules.save_load_handlers.data_handle import read_data_from_file, read_other_data_from_file, \
    CollectedDataType
from src.ai.models.autoencoder import Autoencoder
from src.modules.save_load_handlers.ai_models_handle import load_latest_ai, load_manually_saved_ai, AIType
from src.ai.runtime_data_storage.storage import Storage, Coords
from src.ai.runtime_data_storage.storage_superset2 import StorageSuperset2
from src.ai.evaluation.pathfinding_known import pathfinding_step, pathfinding_step_super, pathfinding_step_super_ab
from src.ai.models.base_autoencoder_model import BaseAutoencoderModel
from src.ai.runtime_data_storage.storage import RawConnectionData, RawEnvironmentData
from src.utils import array_to_tensor
from src.ai.evaluation.evaluation import evaluate_adjacency_properties, evaluate_reconstruction_error, \
    evaluate_distances_between_pairs
from src.ai.evaluation.other_operations import get_model_distance_degree1

RENDERER = manim.RendererType.OPENGL
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


def add_mobjects_connections(scene: Scene, connections_data: List[RawConnectionData], mapped_data: Dict,
                             distance_scale: float) -> None:
    """
    Adds lines between the points in the connections_data
    """
    for connection in connections_data:
        start = connection["start"]
        end = connection["end"]

        x_start, y_start = mapped_data[start]["x"], mapped_data[start]["y"]
        x_end, y_end = mapped_data[end]["x"], mapped_data[end]["y"]

        line = Line(start=x_start * distance_scale * RIGHT + y_start * distance_scale * UP,
                    end=x_end * distance_scale * RIGHT + y_end * distance_scale * UP, color=WHITE)
        scene.add(line)


def add_mobjects_vectors_pathfinding(scene: Scene, model: BaseAutoencoderModel, storage: Storage,
                                     datapoints_coordinate_map: Dict[str, Coords],
                                     distance_scale: float, target_name: str, first_n_closest: int,
                                     max_search_distance: int) -> None:
    """
    Runs pathfinding algorithm step for each datapoint and adds arrow direction corresponding to the walking direction
    """

    for current_position in datapoints_coordinate_map:
        if current_position == target_name:
            continue

        closest_point = \
            pathfinding_step(model, storage, current_position, target_name, first_n_closest, max_search_distance)[0]

        x_current = datapoints_coordinate_map[current_position]["x"]
        y_current = datapoints_coordinate_map[current_position]["y"]

        x_target = datapoints_coordinate_map[closest_point]["x"]
        y_target = datapoints_coordinate_map[closest_point]["y"]

        scene.add(Arrow(start=x_current * distance_scale * RIGHT + y_current * distance_scale * UP,
                        end=x_target * distance_scale * RIGHT + y_target * distance_scale * UP, color=RED,
                        stroke_width=2, max_tip_length_to_length_ratio=0.1))


def add_mobjects_vectors_pathfinding_super_ab(scene: Scene, model: BaseAutoencoderModel, storage: Storage,
                                              datapoints_coordinate_map: Dict[str, Coords],
                                              distance_scale: float, target_name: str, first_n_closest: int,
                                              max_search_distance: int) -> None:
    """
    Runs pathfinding algorithm step for each datapoint and adds arrow direction corresponding to the walking direction
    """

    for current_position in datapoints_coordinate_map:
        if current_position == target_name:
            continue

        closest_point = \
            pathfinding_step_super_ab(model, storage, current_position, target_name, first_n_closest,
                                      max_search_distance)[
                0]

        x_current = datapoints_coordinate_map[current_position]["x"]
        y_current = datapoints_coordinate_map[current_position]["y"]

        x_target = datapoints_coordinate_map[closest_point]["x"]
        y_target = datapoints_coordinate_map[closest_point]["y"]

        scene.add(Arrow(start=x_current * distance_scale * RIGHT + y_current * distance_scale * UP,
                        end=x_target * distance_scale * RIGHT + y_target * distance_scale * UP, color=RED,
                        stroke_width=2, max_tip_length_to_length_ratio=0.1))


def add_mobjects_vectors_pathfinding_super(scene: Scene, model: BaseAutoencoderModel, storage: Storage,
                                           datapoints_coordinate_map: Dict[str, Coords],
                                           distance_scale: float, target_name: str, first_n_closest: int,
                                           max_search_distance: int) -> None:
    """
    Runs pathfinding algorithm step for each datapoint and adds arrow direction corresponding to the walking direction
    """

    for current_position in datapoints_coordinate_map:
        if current_position == target_name:
            continue

        closest_point = \
            pathfinding_step_super(model, storage, current_position, target_name, first_n_closest, max_search_distance)[
                0]

        x_current = datapoints_coordinate_map[current_position]["x"]
        y_current = datapoints_coordinate_map[current_position]["y"]

        x_target = datapoints_coordinate_map[closest_point]["x"]
        y_target = datapoints_coordinate_map[closest_point]["y"]

        scene.add(Arrow(start=x_current * distance_scale * RIGHT + y_current * distance_scale * UP,
                        end=x_target * distance_scale * RIGHT + y_target * distance_scale * UP, color=RED,
                        stroke_width=2, max_tip_length_to_length_ratio=0.1))


def add_mobjects_datapoints(scene: Scene, mapped_data: Dict, distance_scale: float, radius: float) -> None:
    """
    Adds circles to the scene representing the data points, based on the built coordinates map
    """
    for key in mapped_data:
        x, y = mapped_data[key]["x"], mapped_data[key]["y"]
        circ = Circle(radius=radius, color=WHITE)
        circ.move_to(x * distance_scale * RIGHT + y * distance_scale * UP)

        # adds a text with the name of the post in the circle
        text = Text(key, font_size=8)
        text.move_to(circ.get_center())

        scene.add(text)
        scene.add(circ)


def build_scene_normal():
    scene = IntroScene()

    global storage
    storage.load_raw_data(CollectedDataType.Data8x8)
    storage.normalize_all_data()

    autoencoder = load_manually_saved_ai("autoencod_permutated1mil.pth")
    # autoencoder = load_latest_ai(AIType.Autoencoder)

    storage.build_datapoints_coordinates_map()
    # quality of life, centered coords at 0,0
    storage.recenter_datapoints_coordinates_map()
    datapoints_coordinates_map = storage.get_datapoints_coordinates_map()

    raw_environment_data = storage.get_raw_environment_data()
    connections_data = storage.get_raw_connections_data()

    distance_scale = 1
    radius = 0.2

    add_mobjects_datapoints(scene, datapoints_coordinates_map, distance_scale, radius)
    add_mobjects_vectors_pathfinding(scene, autoencoder, storage, datapoints_coordinates_map, distance_scale, "0_0", 1,
                                     1)

    # add_mobjects_connections(scene, connections_data, datapoints_coordinates_map, distance_scale)
    # add_mobjects_found_adjacencies(scene, autoencoder, raw_environment_data, datapoints_coordinates_map,
    #                                distance_scale)

    return scene


def build_scene_datapoints_topology():
    scene = IntroScene()

    global storage_superset2
    storage_superset2.load_raw_data_from_others("datapoints_random_walks_50.json")
    storage_superset2.load_raw_data_connections_from_others("datapoints_connections_randon_walks_50.json")

    storage_superset2.build_datapoints_coordinates_map()
    # storage_superset2.recenter_datapoints_coordinates_map()
    datapoints_coordinates_map = storage_superset2.get_datapoints_coordinates_map()

    distance_scale = 1
    radius = 0.2

    add_mobjects_datapoints(scene, datapoints_coordinates_map, distance_scale, radius)
    return scene


def build_scene_autoencoded_permuted():
    scene = IntroScene()

    global storage_superset2
    permutor = load_manually_saved_ai("permutor_deshift_working.pth")
    storage_superset2.set_permutor(permutor)
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
    scene.interactive_embed()


def run_visualization():
    run_opengl_configs()
    scene = build_scene_datapoints_topology()
    run_opengl_scene(scene)


def basic_viz():
    pass


def myfunc1():
    print("Hello from a thread")


if __name__ == "__main__":
    run_visualization()
