from manim import *


class Scene3D(ThreeDScene):
    def on_key_press(self, symbol, modifiers):
        """Called each time a key is pressed."""
        super().on_key_press(symbol, modifiers)


class Scene2D(Scene):
    def on_key_press(self, symbol, modifiers):
        """Called each time a key is pressed."""
        super().on_key_press(symbol, modifiers)
