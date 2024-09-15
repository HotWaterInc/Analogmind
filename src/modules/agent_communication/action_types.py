from typing import Callable, Dict, TypedDict, Union, List


class TeleportAbsoluteType(TypedDict):
    action_type: str
    x: float
    y: float


class TeleportRelativeType(TypedDict):
    action_type: str
    dx: float
    dy: float


class RotateAbsoluteType(TypedDict):
    action_type: str
    angle: float


class RotateRelativeType(TypedDict):
    action_type: str
    dangle: float


class SampleDistanceType(TypedDict):
    action_type: str
    pass


class SampleImageType(TypedDict):
    action_type: str
    pass


class OtherType(TypedDict):
    action_type: str
    pass


TELEPORT_ABSOLUTE_STRING = "teleport_absolute"
TELEPORT_RELATIVE_STRING = "teleport_relative"
ROTATE_ABSOLUTE_STRING = "rotate_absolute"
ROTATE_RELATIVE_STRING = "rotate_relative"

SAMPLE_DISTANCE_STRING = "sample_distance"
SAMPLE_IMAGE_STRING = "sample_image"

action_types: List[str] = [
    TELEPORT_ABSOLUTE_STRING,
    TELEPORT_RELATIVE_STRING,
    ROTATE_ABSOLUTE_STRING,
    ROTATE_RELATIVE_STRING,
    SAMPLE_DISTANCE_STRING,
    SAMPLE_IMAGE_STRING,
]

JsonDataAction = Union[
    TeleportAbsoluteType,
    TeleportRelativeType,
    RotateAbsoluteType,
    RotateRelativeType,
    SampleDistanceType,
    SampleImageType,
    OtherType
]
