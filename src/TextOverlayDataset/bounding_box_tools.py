import math
from typing import Tuple

import numpy


def aabb_to_bbox(aabb: numpy.generic) -> Tuple[float, float, float, float]:
    """Convert an axis-aligned bounding box to a grid-normal bounding box.
    ASSUMES DOWN IS +Y!  0 is the top!
    Also assumes +x is right, which one would hope is uncontroversial, but stranger things have happened.
    The winding order of this array is important, as it matches with what PIL's Quad transform expects.
    We start in the upper left and go counter-clockwise.
    :return float, float, float, float: left, top, right, bottom
    """
    left, top = aabb[:, :2].min(axis=0)
    right, bottom = aabb[:, :2].max(axis=0)
    return left, top, right, bottom


def bbox_to_aabb(left: float, top: float, right: float, bottom: float) -> numpy.ndarray:
    """Return a 4x3 augmented matrix."""
    return numpy.asarray([
        [left, top, 1.0],
        [right, top, 1.0],
        [left, bottom, 1.0],
        [right, bottom, 1.0],
    ])


def make_rotation_matrix_2d(angle: float) -> numpy.ndarray:
    # Remember: x' = xcos(t) - ysin(t),  y' = xsin(t) + ycos(t)
    rotation_matrix_2d = numpy.eye(3)
    rotation_matrix_2d[0, 0] = math.cos(angle)
    rotation_matrix_2d[0, 1] = -math.sin(angle)
    rotation_matrix_2d[1, 0] = math.sin(angle)
    rotation_matrix_2d[1, 1] = math.cos(angle)
    return rotation_matrix_2d


def rotate_around_point(points: numpy.generic, angle: float, x: float, y: float) -> numpy.ndarray:
    """Rotate each point (row) in the given points by angle radians around the point x, y."""
    p = numpy.asarray([x, y, 1.0])
    points -= p
    points = points @ make_rotation_matrix_2d(angle)
    points += p
    return points
