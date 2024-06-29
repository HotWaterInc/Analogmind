from manim import *
from pyglet.window import key
from manim.opengl import *
import logging
import manim
from src.modules.data_handlers.ai_data_handle import read_data_from_file, read_other_data_from_file, CollectedDataType

RENDERER = manim.RendererType.OPENGL

class IntroScene(Scene):
    def construct(self):
        square = Square(color=RED).shift(LEFT * 2)
        circle = Circle(color=BLUE).shift(RIGHT * 2)

        self.play(Write(square), Write(circle))

        self.interactive_embed()


    def on_key_press(self, symbol, modifiers):
        """Called each time a key is pressed."""

        # so we can still use the default controls
        super().on_key_press(symbol, modifiers)

def build_scene():
    scene = IntroScene()

    json_data = read_data_from_file(CollectedDataType.Data8x8)
    print(json_data)
    connections_data = read_other_data_from_file("connections.json")


    return scene

def run_scene(scene):
    config.renderer = RENDERER
    print(f"{config.renderer = }")
    # config.video_dir = "videos"
    # config.quality = "high_quality"

    config.disable_caching = True
    config.preview = True
    config.write_to_movie = False
    config.input_file = "main.py"

    # mutes manim logger
    logger.setLevel(logging.WARNING)
    scene = IntroScene()
    scene.render()

if __name__ == "__main__":
    scene = build_scene()
    # run_scene(scene)