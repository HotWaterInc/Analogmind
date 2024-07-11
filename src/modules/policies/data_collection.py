from typing import Dict, TypedDict

from src.action_robot_controller import detach_robot_sample, detach_robot_teleport_relative, \
    detach_robot_rotate_absolute, detach_robot_rotate_relative, detach_robot_teleport_absolute


def grid_data_collection(width: float, height: float, grid_size: int, center_x: float, center_y: float,
                         rotations: int) -> None:
    """
    Collects data from the environment in a grid pattern with height and width as the boundaries.
    The height and width are split into grid_size x grid_size grid.

    rotations tells in how many parts are the 360 degrees split into.
    """

    # Calculate the step size for the grid
    step_size = width / grid_size

    # Calculate the starting x and y positions
    start_x = center_x - width / 2
    start_y = center_y - height / 2

    # Calculate the rotation step
    rotation_step = 360 / rotations

    # Loop through the grid
    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate the x and y positions
            x = start_x + i * step_size
            y = start_y + j * step_size

            detach_robot_teleport_absolute(x, y)
            # Loop through the rotations
            for k in range(rotations):
                # Calculate the angle
                angle = k * rotation_step

                # Send the data
                detach_robot_rotate_absolute(angle)
                detach_robot_sample()
