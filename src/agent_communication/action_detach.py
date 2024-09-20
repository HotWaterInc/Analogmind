from src.modules.agent_communication.action_decorators import wait_agent_response
from src.modules.agent_communication.communication_controller import agent_send_data
from src.modules.agent_communication.action_types import *


@wait_agent_response
def detach_robot_sample_distance():
    json_data: SampleDistanceType = {
        "action_type": SAMPLE_DISTANCE_STRING
    }
    agent_send_data(json_data)


@wait_agent_response
def detach_robot_sample_image():
    json_data: SampleImageType = {
        "action_type": SAMPLE_IMAGE_STRING
    }
    agent_send_data(json_data)


@wait_agent_response
def detach_robot_teleport_relative(dx: float, dy: float):
    json_data: TeleportRelativeType = {
        "action_type": TELEPORT_RELATIVE_STRING,
        "dx": dx,
        "dy": dy
    }
    agent_send_data(json_data)


@wait_agent_response
def detach_robot_teleport_absolute(x: float, y: float):
    json_data: TeleportAbsoluteType = {
        "action_type": TELEPORT_ABSOLUTE_STRING,
        "x": x,
        "y": y
    }
    agent_send_data(json_data)


@wait_agent_response
def detach_robot_rotate_absolute(angle: float):
    json_data: RotateAbsoluteType = {
        "action_type": ROTATE_ABSOLUTE_STRING,
        "angle": angle
    }
    agent_send_data(json_data)


@wait_agent_response
def detach_robot_rotate_relative(dangle: float):
    json_data: RotateRelativeType = {
        "action_type": ROTATE_RELATIVE_STRING,
        "dangle": dangle
    }
    agent_send_data(json_data)
