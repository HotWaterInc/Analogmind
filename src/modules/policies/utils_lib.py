import math


def calculate_angle(x_vector, y_vector):
    dot_product = x_vector[0] * y_vector[0] + x_vector[1] * y_vector[1]
    determinant = x_vector[0] * y_vector[1] - x_vector[1] * y_vector[0]
    angle = math.atan2(determinant, dot_product)
    return angle


def degrees_to_radians(degrees: float) -> float:
    return degrees * math.pi / 180


def radians_to_degrees(radians: float) -> float:
    return radians * 180 / math.pi


def webots_radians_to_normal(x: float) -> float:
    if x < 0:
        x += 2 * math.pi
    return x


def normal_radians_to_webots(x: float) -> float:
    if x > math.pi:
        x -= 2 * math.pi
    return x
