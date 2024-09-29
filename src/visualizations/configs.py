from manim import *
import torch
import logging
import manim

from src.navigation_core.networks.metric_generator.metric_network_abstract import MetricNetworkAbstract
from src.runtime_storages.storage_struct import StorageStruct
from src.save_load_handlers.ai_models_handle import load_manually_saved_ai


def manim_configs_png(filename: str):
    config.renderer = manim.RendererType.CAIRO

    config.disable_caching = True
    config.preview = True
    config.input_file = filename
    config.quality = "high_quality"

    # mutes manim logger
    logger.setLevel(logging.WARNING)


def manim_configs_opengl(filename: str):
    config.renderer = manim.RendererType.OPENGL
    print(f"{config.renderer = }")

    config.disable_caching = True
    config.preview = True
    config.write_to_movie = False

    config.input_file = filename
    # mutes manim logger
    logger.setLevel(logging.WARNING)
