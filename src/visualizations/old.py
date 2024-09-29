from manim import *
import matplotlib.pyplot as plt
import torch
import logging
import manim
from src.navigation_core.networks.metric_generator.metric_network_abstract import MetricNetworkAbstract
from src.runtime_storages.storage_struct import StorageStruct
from src.save_load_handlers.ai_models_handle import load_manually_saved_ai
from src.visualizations.configs import manim_configs_png, manim_configs_opengl
from src.visualizations.decorators import run_as_png, run_as_interactive_opengl
from src.visualizations.run_functions import manim_run_opengl_scene
from src.visualizations.scenes import Scene2D


def add_mobjects_found_adjacencies(scene: Scene, model: MetricNetworkAbstract, datapoints: List[RawEnvironmentData],
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
    manim_configs_opengl()
    # manim_configs_png()
    scene = Scene3D()
    build_3d_mse(scene)
    # scene.render()
    run_opengl_scene(scene)


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
