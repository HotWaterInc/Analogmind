from enum import Enum
from src.modules.external_communication.communication_interface import send_data
from src.modules.external_communication.communication_interface import TeleportRelativeAction, TeleportAbsoluteAction, \
    RotateAbsoluteAction, RotateRelativeAction, SampleAction, RotateAbsoluteContinuousAction, ForwardContinuousAction
from src.modules.external_communication.communication_interface import action_types, ActionTypeTeleportAbsolute, \
    ActionTypeTeleportRelative, ActionTypeRotateAbsolute, ActionTypeRotateRelative, ActionTypeSample, \
    ActionTypeContRotateAbsolute, ActionTypeContForward

from typing import Dict, TypedDict


def detach_robot_sample():
    json_data: SampleAction = {
        "action_type": ActionTypeSample
    }
    send_data(json_data)


def detach_robot_teleport_relative(dx: float, dy: float):
    json_data: TeleportRelativeAction = {
        "action_type": ActionTypeTeleportRelative,
        "dx": dx,
        "dy": dy
    }
    send_data(json_data)


def detach_robot_teleport_absolute(x: float, y: float):
    json_data: TeleportAbsoluteAction = {
        "action_type": ActionTypeTeleportAbsolute,
        "x": x,
        "y": y
    }
    send_data(json_data)


def detach_robot_rotate_continuous_absolute(angle: float):
    json_data: RotateAbsoluteContinuousAction = {
        "action_type": ActionTypeContRotateAbsolute,
        "angle": angle
    }
    send_data(json_data)


def detach_robot_forward_continuous(distance: float):
    json_data: ForwardContinuousAction = {
        "action_type": ActionTypeContForward,
        "distance": distance
    }
    send_data(json_data)


def detach_robot_rotate_absolute(angle: float):
    json_data: RotateAbsoluteAction = {
        "action_type": ActionTypeRotateAbsolute,
        "angle": angle
    }
    send_data(json_data)


def detach_robot_rotate_relative(dangle: float):
    json_data: RotateRelativeAction = {
        "action_type": ActionTypeRotateRelative,
        "dangle": dangle
    }
    send_data(json_data)
