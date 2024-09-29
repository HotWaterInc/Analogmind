from manim import Scene


def manim_run_opengl_scene(scene: Scene):
    scene.interactive_embed()


def manim_run_png_scene(scene: Scene):
    scene.render()
