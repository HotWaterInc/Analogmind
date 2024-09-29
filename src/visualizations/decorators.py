from src.visualizations.configs import manim_configs_png, manim_configs_opengl
from src.visualizations.run_functions import manim_run_png_scene, manim_run_opengl_scene


def run_as_png(filename: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            manim_configs_png(filename)
            scene = func(*args, **kwargs)
            manim_run_png_scene(scene)

        return wrapper

    return decorator


def run_as_interactive_opengl(filename: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            manim_configs_opengl(filename)
            scene = func(*args, **kwargs)
            manim_run_opengl_scene(scene)

        return wrapper

    return decorator
