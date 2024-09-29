from manim import *
import logging
import manim
import threading
from time import sleep


class IntroScene(Scene):
    def func(self):
        print("Hello from IntroScene")


def change_scene(scene: Scene):
    print("in change scen")
    sleep(1)
    print("after sleep")
    # scene.add(Circle())
    set_event()


def render_scene_cont():
    global scene_global
    while True:
        wait_event()
        print("rerendering scene")
        # scene_global.embed_update()
        circ = Circle()
        scene_global.play(Create(circ))
        scene_global.wait(1)

        scene_global.play(circ.animate.shift(UP))
        scene_global.wait(1)
        clear_event()


event1 = threading.Event()


def set_event():
    event1.set()


def clear_event():
    event1.clear()


def wait_event():
    event1.wait()


def build_scene():
    scene = IntroScene()
    square = Square()
    scene.add(square)
    return scene


def run_opengl_configs():
    config.renderer = manim.RendererType.OPENGL
    config.disable_caching = True
    config.preview = True
    config.write_to_movie = False
    config.input_file = "visualizations_real_time.py"
    # mutes manim logger
    logger.setLevel(logging.WARNING)


global scene_global

if __name__ == "__main__":
    # Only a small demo
    # TODO: Needs to be implemented at a basic level on the new infrastructure
    run_opengl_configs()
    scene = build_scene()
    scene_global = scene

    # renderer needs to use main thread to render
    thread = threading.Thread(target=change_scene, args=(scene_global,))
    thread.start()

    scene.embed_test()
    render_scene_cont()

    thread.join()
