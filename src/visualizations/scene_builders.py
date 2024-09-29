from src.navigation_core.utils import frontier_find_all_datapoints_and_directions
from src.runtime_storages.storage_struct import StorageStruct
from src.runtime_storages.types import ConnectionNullData
from src.save_load_handlers.data_handle import read_other_data_from_file
from manim import *
from src import runtime_storages as storage
from src.visualizations.configs_loader import load_config_ini_visualization, ConfigParameters
from src.visualizations.visualization_storage.types import NodesMapping, MobjectsParams
from src.visualizations.visualization_storage.visualization_struct import VisualizationDataStruct
from . import visualization_storage


def _add_nodes_connections(scene: Scene, visualization_struct: VisualizationDataStruct,
                           storage_struct: StorageStruct) -> None:
    all_connections = storage.connections_all_get(storage_struct)
    mapped_data = visualization_storage.get_nodes_coordinates_map(visualization_struct)
    distance_scale = visualization_storage.get_mobjects_params(visualization_struct)["distance_scale"]

    for connection in all_connections:
        start = connection["start"]
        end = connection["end"]

        if start not in mapped_data or end not in mapped_data:
            continue

        x_start, y_start = mapped_data[start]["x"], mapped_data[start]["y"]
        x_end, y_end = mapped_data[end]["x"], mapped_data[end]["y"]

        line = Line(start=x_start * distance_scale * RIGHT + y_start * distance_scale * UP,
                    end=x_end * distance_scale * RIGHT + y_end * distance_scale * UP, color=WHITE, stroke_width=1)
        scene.add(line)


def _add_connections_frontier(scene: Scene, visualization_struct: VisualizationDataStruct,
                              storage_struct: StorageStruct,
                              ) -> None:
    params = visualization_storage.get_mobjects_params(visualization_struct)
    distance_scale = params["distance_scale"]
    connections_frontier = frontier_find_all_datapoints_and_directions(storage_struct)
    mapped_data = visualization_storage.get_nodes_coordinates_map(visualization_struct)

    for connection in connections_frontier:
        start = connection["start"]
        dx = connection["direction"][0]
        dy = connection["direction"][1]

        x_start, y_start = mapped_data[start]["x"], mapped_data[start]["y"]
        x_end, y_end = x_start + dx, y_start + dy

        line = Line(start=x_start * distance_scale * RIGHT + y_start * distance_scale * UP,
                    end=x_end * distance_scale * RIGHT + y_end * distance_scale * UP, color=BLUE, stroke_width=1)
        scene.add(line)


def _add_connections_null(scene: Scene, visualization_struct: VisualizationDataStruct,
                          storage_struct: StorageStruct) -> None:
    """
    Adds lines between the points in the connections_data
    """
    connections_null: List[ConnectionNullData] = storage.connections_null_get(storage_struct)
    mapped_data: NodesMapping = visualization_storage.get_nodes_coordinates_map(visualization_struct)
    distance_scale = visualization_storage.get_mobjects_params(visualization_struct)["distance_scale"]

    for connection in connections_null:
        start = connection["start"]
        key = connection["start"]
        dx = connection["direction"][0]
        dy = connection["direction"][1]

        x_start, y_start = mapped_data[start]["x"], mapped_data[start]["y"]
        x_end, y_end = x_start + dx, y_start + dy

        line = Line(start=x_start * distance_scale * RIGHT + y_start * distance_scale * UP,
                    end=x_end * distance_scale * RIGHT + y_end * distance_scale * UP, color=YELLOW, stroke_width=1)
        scene.add(line)


def _add_mobjects_datapoints(scene: Scene, visualization_struct: VisualizationDataStruct,
                             storage_struct: StorageStruct) -> None:
    """
    Adds circles to the scene representing the data points, based on the built coordinates map
    """
    mapped_data: NodesMapping = visualization_storage.get_nodes_coordinates_map(visualization_struct)
    params = visualization_storage.get_mobjects_params(visualization_struct)
    distance_scale = params["distance_scale"]
    radius = params["radius"]

    for key in mapped_data:
        x, y = mapped_data[key]["x"], mapped_data[key]["y"]
        circ = Circle(radius=radius, color=WHITE)
        circ.move_to(x * distance_scale * RIGHT + y * distance_scale * UP)

        if load_config_ini_visualization(ConfigParameters.SHOW_NODE_INDEXES):
            # adds a text with the name of the post in the circle
            index = storage.node_get_index_by_name(storage_struct, key)
            text_string = f"{index}"
            text = Text(text_string, font_size=8)
            text.move_to(circ.get_center())
            scene.add(text)

        scene.add(circ)


def build_datapoints_topology(scene: Scene, visualization_struct: VisualizationDataStruct,
                              storage_struct: StorageStruct, show_null: bool = True, show_frontier: bool = True,
                              show_connections=True):
    visualization_storage.build_nodes_coordinates_map(
        visualization_struct=visualization_struct,
        storage_struct=storage_struct,
        percent=1
    )
    visualization_storage.recenter_datapoints_coordinates_map(visualization_struct=visualization_struct)

    params = MobjectsParams(
        radius=0.2,
        distance_scale=1
    )
    visualization_storage.set_mobjects_params(visualization_struct, params)

    # datapoints and normal connections
    _add_mobjects_datapoints(scene=scene, visualization_struct=visualization_struct, storage_struct=storage_struct)
    if show_connections:
        _add_nodes_connections(scene=scene, visualization_struct=visualization_struct, storage_struct=storage_struct)
    if show_frontier:
        _add_connections_frontier(scene=scene, visualization_struct=visualization_struct, storage_struct=storage_struct)
    if show_null:
        _add_connections_null(scene=scene, visualization_struct=visualization_struct, storage_struct=storage_struct)

    return scene


def build_inference_navigation(scene: Scene) -> Scene:
    DISTANCE_SCALE = 1
    RADIUS = 0.2
    inference_policy_data = read_other_data_from_file("inference_policy_results_no_noise.json")
    STEPS_TO_WIN = inference_policy_data["STEPS_TO_WIN"]
    DISTANCE_FROM_TRUE_TARGET = inference_policy_data["DISTANCE_FROM_TRUE_TARGET"]
    WINS = inference_policy_data["WINS"]
    RECORDED_STEPS = inference_policy_data["RECORDED_STEPS"]
    ORIGINAL_TARGETS = inference_policy_data["ORIGINAL_TARGETS"]
    ACTUAL_POSITIONS = inference_policy_data["ACTUAL_POSITIONS"]

    WALK_INDEX = 0
    count = 1

    for idx, steps in enumerate(RECORDED_STEPS):
        if len(steps) < 25 and len(steps) > 20:
            if count == 0:
                count += 1
                continue
            WALK_INDEX = idx
            break

    # Create a group to hold all elements
    all_elements = VGroup()

    # add target circle
    target_circle = Circle(radius=RADIUS, color=RED)
    x_target = ORIGINAL_TARGETS[WALK_INDEX][0]
    y_target = ORIGINAL_TARGETS[WALK_INDEX][1]
    target_circle.move_to(x_target * DISTANCE_SCALE * RIGHT + y_target * DISTANCE_SCALE * UP)
    all_elements.add(target_circle)

    # add starting circle
    starting_circle = Circle(radius=RADIUS, color=BLUE)
    x_start = ACTUAL_POSITIONS[WALK_INDEX][0][0]
    y_start = ACTUAL_POSITIONS[WALK_INDEX][0][1]
    starting_circle.move_to(x_start * DISTANCE_SCALE * RIGHT + y_start * DISTANCE_SCALE * UP)
    all_elements.add(starting_circle)

    # add end path circle
    end_circle = Circle(radius=RADIUS, color=GREEN)
    x_end = ACTUAL_POSITIONS[WALK_INDEX][-1][0]
    y_end = ACTUAL_POSITIONS[WALK_INDEX][-1][1]
    end_circle.move_to(x_end * DISTANCE_SCALE * RIGHT + y_end * DISTANCE_SCALE * UP)
    all_elements.add(end_circle)

    # add path
    current_x, current_y = x_start, y_start
    for step in RECORDED_STEPS[WALK_INDEX]:
        dx, dy = step[0], step[1]
        line = Arrow(
            start=current_x * DISTANCE_SCALE * RIGHT + current_y * DISTANCE_SCALE * UP,
            end=(current_x + dx) * DISTANCE_SCALE * RIGHT + (current_y + dy) * DISTANCE_SCALE * UP,
            color=WHITE
        )
        all_elements.add(line)
        current_x += dx
        current_y += dy

    # Scale the entire group to fit the screen
    screen_width = config.frame_width
    screen_height = config.frame_height
    group_width = all_elements.width
    group_height = all_elements.height

    scale_factor = min(
        (screen_width - 1) / group_width,
        (screen_height - 1) / group_height
    )

    all_elements.scale(scale_factor)

    # Center the scaled group
    all_elements.move_to(ORIGIN)

    # Add the scaled and centered group to the scene
    scene.add(all_elements)

    return scene
