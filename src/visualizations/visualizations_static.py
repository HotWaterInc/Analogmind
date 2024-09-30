from manim import *
import torch
import manim
from src.navigation_core.networks.metric_generator.metric_network_abstract import MetricNetworkAbstract
from src.runtime_storages.storage_struct import StorageStruct
from src.save_load_handlers.ai_models_handle import load_manually_saved_ai
from src.visualizations.configs import manim_configs_png, manim_configs_opengl
from src.visualizations.decorators import run_as_png, run_as_interactive_opengl
from src.visualizations.run_functions import manim_run_opengl_scene
from src.visualizations.scene_builders import build_navigation_path, build_nodes_topology, build_metric_space_surface
from src.visualizations.scenes import Scene2D, Scene3D
from src.visualizations import visualization_storage as visualization_storage


@run_as_interactive_opengl("entry_dev.py")
def visualization_3d_target_surface(storage_struct: StorageStruct, metric_network: any) -> None:
    # TODO: Implement with the networks actually trained
    scene = Scene3D()
    visualization_struct = visualization_storage.create_visualization_struct()
    build_metric_space_surface(
        scene=scene,
        visualization_struct=visualization_struct,
        storage_struct=storage_struct,
        target_x=0,
        target_y=0,
        metric_network=None
    )


@run_as_png("entry_dev.py")
def visualization_topological_graph(storage_struct: StorageStruct) -> Scene:
    scene = Scene2D()
    visualization_struct = visualization_storage.create_visualization_struct()
    build_nodes_topology(
        scene=scene,
        storage_struct=storage_struct,
        visualization_struct=visualization_struct,
    )
    return scene


@run_as_png("entry_dev.py")
def visualization_navigation_paths() -> Scene:
    # TODO: Implement again
    scene = Scene2D()
    build_navigation_path(scene)
    return scene


storage_struct: StorageStruct
