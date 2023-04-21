import math
from itertools import permutations
from typing import List, Optional, Tuple

import numpy


def aabb_to_bbox(aabb: numpy.generic) -> Tuple[float, float, float, float]:
    """Convert an axis-aligned bounding box to a grid-normal bounding box.
    ASSUMES DOWN IS +Y!  0 is the top!
    Also assumes +x is right, which one would hope is uncontroversial, but stranger things have happened.
    :return float, float, float, float: left, top, right, bottom
    """
    left = numpy.min(aabb[:, 0])
    top = numpy.min(aabb[:, 1])
    right = numpy.max(aabb[:, 0])
    bottom = numpy.max(aabb[:, 1])
    return left, top, right, bottom


def bbox_to_aabb(left: float, top: float, right: float, bottom: float) -> numpy.ndarray:
    """Return a 4x3 augmented matrix starting in the bottom left and going counter-clockwise."""
    return numpy.asarray([
        [left, top, 1.0],
        [left, bottom, 1.0],
        [right, bottom, 1.0],
        [right, top, 1.0],
    ])


def sign(a):
    if a < 0:
        return -1
    return 1


def points_to_theta(x1, y1, x2, y2):
    dx = x2-x1
    dy = y2-y1
    return math.atan2(dy, dx)


def line_circle_intersection(x1: float, y1: float, x2: float, y2: float, cx: float, cy: float, radius: float) -> List[Tuple[float, float]]:
    # Recenter both points.  We will operate about the origin.
    dx = x2 - x1
    dy = y2 - y1
    x1 -= cx
    y1 -= cy
    x2 -= cx
    y2 -= cy
    dmag = math.sqrt(dx*dx + dy*dy)
    det = x1*y2 - x2*y1
    discriminant = (radius*radius * dmag*dmag) - det*det
    if discriminant < 0:
        return []
    # We have one or more points.
    x_intercept_1 = (det*dy + sign(dy)*dx*math.sqrt(discriminant))/(dmag*dmag)
    x_intercept_2 = (det*dy - sign(dy)*dx*math.sqrt(discriminant))/(dmag*dmag)
    y_intercept_1 = (-det*dx + abs(dy)*math.sqrt(discriminant))/(dmag*dmag)
    y_intercept_2 = (-det*dx - abs(dy)*math.sqrt(discriminant))/(dmag*dmag)
    return [[x_intercept_1+cx, y_intercept_1+cy], [x_intercept_2+cx, y_intercept_2+cy]]
    

def compute_max_rect_in_rect_rotation(
        pivot: Tuple[float, float], 
        inner_rectangle: Tuple[float, float, float, float], 
        outer_rectangle: Tuple[float, float, float, float]
) -> List[Tuple[float, float]]:
    """Compute the valid rotation ranges for the given inner triangle and pivot.
    :param float float pivot: The x,y coordinates of the point around which the rectangle will be rotated.
    :param float float float float inner_rectangle: The left, top, right, bottom of the inner rectangle.
    :param float float float float outer_rectnalge: The left, top, right, and bottom of the outer rectangle.
    :return List[Tuple[float, float]]: [] if the inner rectangle cannot fit inside the outer rectangle.  [[0, 2*math.pi]] if the inner is unconstrained.
    """
    ranges = [(0, 2.0*math.pi)]
    inner_left, inner_up, inner_right, inner_bottom = inner_rectangle
    outer_left, outer_up, outer_right, outer_bottom = outer_rectangle
    inner_points = [
        [inner_left, inner_up],
        [inner_right, inner_up],
        [inner_right, inner_bottom],
        [inner_left, inner_bottom],
    ]
    inner_lines = [
        [inner_points[0], inner_points[1]],
        [inner_points[1], inner_points[2]],
        [inner_points[2], inner_points[3]],
        [inner_points[3], inner_points[0]],
    ]
    outer_points = [
        [outer_left, outer_up],
        [outer_right, outer_up],
        [outer_right, outer_bottom],
        [outer_left, outer_bottom],
    ]
    outer_lines = [
        [outer_points[0], outer_points[1]],
        [outer_points[1], outer_points[2]],
        [outer_points[2], outer_points[3]],
        [outer_points[3], outer_points[0]],
    ]
    for p in inner_points:
        pass
    # Find all spans where the edges intersect and return valid spans.
    raise NotImplementedError()
    return [[0, 2*math.pi]]
    # Ignore unreachable.  Moving code here for reference.  Will get pulled.
    """
    Compute the maximum rotation and translation that we can apply to this block of text, assuming the pivot point
    is the center of the text bounding box.

    Explanation of this function:
    There's a little basic trig in here.  I think there's a more optimal way to determine this value, but it
    requires a better mind.

    Start by moving the image_width to the left so that the rotation axis coincides with the center of whatever
    space is left over.

    ```
    +------ Width_img / 2 ----------+
    |                               |
    +-- Width_txt/2 ----(A)       Height_img/2
    |                    |          |
    | Theta_txt    Height_txt/2     |
    +--------------------+----------+
    ```

    The point A is the top-right corner of the rectangle defined by the text.
    We want to ensure that A is below height_img/2 and to the left of width_img/2.
    Because the rectangle is balanced and symmetric, we ensure that the other corners are inside the rect, too.
    Theta_txt refers to the rotation of the whole text box and starts at zero.
    Theta_a (not labeled) is the angle of A when Theta_txt is zero.  (A.k.a., the starting angle of point A.)
    Theta_a = atan(height_txt/2 / width_txt/2), which simplifies to atan(height_txt/width_txt).
    A moves to the right and reaches max when Theta_txt hits -Theta_A.
    A moves up and reaches a max when Theta_A is pi/2 (90), so a minx/maxy at Theta_txt + Theta_a = pi/2.
    We have two cases to check for A_x and A_y.  If A_x is less than or equal to Width_img/2 at max and A_y is less
    than or equal to Height_img/2, we have no constraint on the rotation.
    :param text_bbox: The [left, top, right, bottom] of the text. Assumed to be centered at the box's center.
    :param image_width: The full width of the enclosing image.
    :param image_height: The full height of the enclosing image.
    :return Tuple[float, float]: The minimum and maximum angles or (0, 2*pi) if there is no limit.
    """
    text_left, text_top, text_right, text_bottom = text_bbox
    text_halfwidth = abs(text_right - text_left) * 0.5
    text_halfheight = abs(text_top - text_bottom) * 0.5
    text_radius = math.sqrt(text_halfwidth**2 + text_halfheight**2)
    angle_pa = math.atan2(text_halfheight, text_halfwidth)  # The 0.5 cancels in atan(dy*0.5 / dx*0.5).
    min_angle = 0.0
    max_angle = 2.0*math.pi
    cos_min_angle = image_width*0.5 / text_radius
    sin_max_angle = image_height*0.5 / text_radius
    if -1 < cos_min_angle < 1:
        # We are constrained.  :(
        min_angle = math.acos(cos_min_angle) - angle_pa
    if -1 < sin_max_angle < 1:
        # Constrained by ceiling.
        max_angle = math.asin(sin_max_angle) - angle_pa
    return min_angle, max_angle


def find_lowest_cost_assignment(f: numpy.ndarray, t: numpy.ndarray) -> List[int]:
    """Find the assignment from each vertex in 'from' to a vertex in 'to' that minimizes the total length of the edges.
    Returns a mapping from the 'from' array to the 'to' array, zero-indexed.

    Assumes that f and t are matrices with 4 rows and matched columns.
    
    Example: 

    ```
        t0-----------t1
        |  f3---f2   |
        |  f0---f1   |
        t2-----------t3
    Would return {0:2, 1:3, 2:1, 3:0} because we're mapping f[0] to t[2].
    (In reality, the return value is a list and is just indexed by 'from' in order.)
    ```

    This function is a little inefficient with compute for the sake of simplicity.
    There are 4! possible assignments for a total of 24 checks.  Big whoop.
    """
    smallest_distance = 1e1000
    smallest_permutation = []
    for p in permutations([0, 1, 2, 3], 4):
        # Compute the length of each pair.
        deltas = t[p,:] - f[:,:]
        dist = (deltas * deltas).sum()
        if dist < smallest_distance:
            smallest_distance = dist
            smallest_permutation = p
    return smallest_permutation


def make_rotation_matrix_2d(angle: float) -> numpy.ndarray:
    """Generate a 3x3 rotation around the z axis.
    Since we're multiplying on the _RIGHT_, we're transposing the matrix.
    This means that this will keep rotation consistent with cos(t), sin(t).  Phrased differently, if you multiply a
    point [1, 0, 1] @ this matrix, you would get what you would expect from doing (cos(t), sin(t)).
    This matches with a counter-clockwise rotation in a right-handed space (y-up) or a counter-clockwise space in a left
     handed space (y-down).
    :param angle:
    :return:
    """
    # Remember: x' = xcos(t) - ysin(t),  y' = xsin(t) + ycos(t)
    rotation_matrix_2d = numpy.eye(3)
    rotation_matrix_2d[0, 0] = math.cos(angle)
    rotation_matrix_2d[0, 1] = math.sin(angle)
    rotation_matrix_2d[1, 0] = -math.sin(angle)
    rotation_matrix_2d[1, 1] = math.cos(angle)
    return rotation_matrix_2d


def rotate_around_point(points: numpy.generic, angle: float, x: float, y: float) -> numpy.ndarray:
    """Rotate each point (row) in the given points by angle radians around the point x, y in a counter-clockwise direct-
    ion, assuming y-down. Remember that PIL uses a left-handed y-down chirality.  Increasing angle rotates counter-
    clockwise.
    """
    p = numpy.asarray([x, y, 1.0])
    points -= p
    points = points @ make_rotation_matrix_2d(angle)
    points += p
    return points
