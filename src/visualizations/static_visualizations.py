from manim import *
import torch
import manim
from src.navigation_core.networks.metric_generator.metric_network_abstract import MetricNetworkAbstract
from src.runtime_storages.storage_struct import StorageStruct
from src.save_load_handlers.ai_models_handle import load_manually_saved_ai
from src.visualizations.configs import manim_configs_png, manim_configs_opengl
from src.visualizations.decorators import run_as_png_decorator, run_as_interactive_opengl
from src.visualizations.run_functions import manim_run_opengl_scene
from src.visualizations.scene_builders import build_inference_navigation, build_datapoints_topology
from src.visualizations.scenes import Scene2D
from src.visualizations import visualization_storage as visualization_storage


@run_as_interactive_opengl("entry_dev.py")
def visualization_3d_target_surface():
    pass
    # manim_configs_opengl()
    # # manim_configs_png()
    # scene = Scene3D()
    # build_3d_mse(scene)
    # # scene.render()
    # run_opengl_scene(scene)


@run_as_png_decorator("entry_dev.py")
def visualization_topological_graph(storage: StorageStruct) -> Scene:
    scene = Scene2D()
    visualization_struct = visualization_storage.create_visualization_struct()
    scene = build_datapoints_topology(
        scene=scene,
        storage_struct=storage,
        visualization_struct=visualization_struct,
    )
    return scene


@run_as_png_decorator("entry_dev.py")
def visualization_navigation_paths() -> Scene:
    scene = Scene2D()
    scene = build_inference_navigation(scene)
    return scene


storage_struct: StorageStruct
