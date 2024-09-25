import matplotlib.pyplot as plt
from manim import *
import torch
import logging
import manim

from src.save_load_handlers.ai_models_handle import load_manually_saved_ai

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
    storage.build_nodes_coordinates_sparse(percent=1)
    storage.recenter_nodes_coordinates()
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
    null_connections = storage.connection_null_get_all()
    add_null_connections(scene, null_connections, datapoints_coordinates_map, DISTANCE_SCALE)

    # add possible directions
    possible_directions_connections = find_frontier_all_datapoint_and_direction(
        storage=storage,
        return_first=False,
        starting_point=None
    )
    add_frontier_connections(scene, possible_directions_connections, datapoints_coordinates_map, DISTANCE_SCALE)

    return scene


def plot_histogram(data, bins=30, title='Histogram', xlabel='Value', ylabel='Frequency',
                   color='skyblue', edgecolor='black', alpha=0.7, figsize=(10, 6),
                   show_mean=True, show_median=True, show_stats=True,
                   xlim=None, ylim=None, xticks=None, yticks=None):
    """
    Create and display a histogram using Matplotlib.

    ... [previous docstring content] ...

    xlim (tuple): Tuple of (min, max) for x-axis limits
    ylim (tuple): Tuple of (min, max) for y-axis limits
    xticks (list): List of tick locations for x-axis
    yticks (list): List of tick locations for y-axis

    Returns:
    matplotlib.figure.Figure: The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the histogram
    n, bins, patches = ax.hist(data, bins=bins, color=color, edgecolor=edgecolor, alpha=alpha)

    # Customize the plot
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set axis limits if provided
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    # Set axis ticks if provided
    if xticks:
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(tick) for tick in xticks])
    if yticks:
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(tick) for tick in yticks])

    # Calculate statistics
    mean = np.mean(data)
    median = np.median(data)
    std = np.std(data)

    # Add mean line
    if show_mean:
        ax.axvline(mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean ({mean:.2f})')

    # Add median line
    if show_median:
        ax.axvline(median, color='green', linestyle='dashed', linewidth=1, label=f'Median ({median:.2f})')

    # Add legend if mean or median is shown
    if show_mean or show_median:
        ax.legend()

    # Add text with statistics
    if show_stats:
        stats_text = f'Mean: {mean:.2f}\nMedian: {median:.2f}\nStd Dev: {std:.2f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.tight_layout()
    return fig


def build_inference_navigation(scene):
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


def build_scene_autoencoded_permuted():
    scene = Scene2D()

    global storage_superset2
    permutor = load_manually_saved_ai("permutor_deshift_working.pth")
    storage_superset2.transformation_set(permutor)
    storage_superset2.load_raw_data_from_others("data8x8_rotated20.json")
    storage_superset2.load_raw_data_connections_from_others("data8x8_connections.json")
    storage_superset2.normalize_all_data_super()
    storage_superset2.tanh_all_data()

    storage_superset2.build_permuted_data_raw_with_thetas()
    storage_superset2.build_permuted_data_random_rotations()

    autoencoder = load_manually_saved_ai("autoencodPerm10k_(7).pth")
    storage_superset2.build_datapoints_coordinates_map()
    # quality of life, centered coords at 0,0
    storage_superset2.recenter_nodes_coordinates()
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
    target_name = storage.node_get_closest_to_xy(target_x, target_y)

    def calculate_coords(name):
        coords = storage.node_get_coords_metadata(name)
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


def visualization_3d_target_surface():
    # manim_configs_opengl()
    manim_configs_png()
    scene = Scene3D()
    build_3d_mse(scene)
    scene.render()
    # run_opengl_scene(scene)


def visualization_topological_graph(storage: StorageSuperset2):
    manim_configs_png()
    scene = Scene2D()
    scene = build_datapoints_topology(scene, storage)
    scene.render()


def visualization_inference_navigation():
    # manim_configs_opengl()
    manim_configs_png()
    scene = Scene2D()
    scene = build_inference_navigation(scene)
    scene.render()
    # run_opengl_scene(scene)


DEBUG_ARRAY = None
storage_global: StorageSuperset2 = None
