from manim import *
from pyglet.window import key
from manim.opengl import *
import logging
import manim
import threading
from time import sleep

class IntroScene(Scene):
    def func(self):
        print("Hello from IntroScene")
    pass

def change_scene(scene: Scene):
    # print("Embedded")
    sleep(1)
    scene.add(Circle())
    # print("here 2")
    # scene.renderer.render(scene, -1, [])
    # scene.render()
    # scene.renderer.clear_screen()
    # print("here 4")

def build_scene():
    scene = IntroScene()

    square = Square()
    scene.add(square)
    # scene.interactive_embed()
    print("here 0")
    thread = threading.Thread(target=change_scene, args=(scene,), daemon=True)
    thread.start()
    scene.embed()

def run_opengl_configs():
    config.renderer = manim.RendererType.OPENGL
    config.disable_caching = True
    config.preview = True
    config.write_to_movie = False
    config.input_file = "test.py"
    # mutes manim logger
    logger.setLevel(logging.WARNING)


def myfunc1():
    print("Hello from a thread")

if __name__ == "__main__":
    run_opengl_configs()
    scene = build_scene()
