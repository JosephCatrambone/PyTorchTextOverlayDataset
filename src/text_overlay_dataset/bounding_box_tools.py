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


def line_circle_intersection(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        cx: float,
        cy: float,
        radius: float
) -> List[Tuple[float, float]]:
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
    return [(x_intercept_1+cx, y_intercept_1+cy), (x_intercept_2+cx, y_intercept_2+cy)]
    

def compute_max_rect_in_rect_rotation(
        pivot: Tuple[float, float], 
        inner_rectangle: Tuple[float, float, float, float], 
        outer_rectangle: Tuple[float, float, float, float]
) -> List[Tuple[float, float]]:
    """Compute all values of theta such that the inner_rectangle, when rotated by theta about the pivot, will have no
    walls that intersect with the outer rectangle.

    :param float float pivot:
    The x,y coordinates of the point around which the rectangle will be rotated.

    :param float float float float inner_rectangle:
    The left, top, right, bottom of the inner rectangle.

    :param float float float float outer_rectangle:
    The left, top, right, and bottom of the outer rectangle.

    :return List[Tuple[float, float]]:
    [] if the inner rectangle cannot fit inside the outer rectangle.  [[0, 2*math.pi]] if the inner is unconstrained.
    """

    # A brief breakdown of this algorithm, as we've rewritten it so many times that it's hard to keep straight.
    # This is our setup, we have a pivot (P), an outer rectangle (characterized by ABCD) and an inner rectangle
    # (characterized by EFGH).  We are looking to find all valid ranges for theta.
    #
    # A----------------------D
    # |   / theta            |
    # |  P- - - - - - - - - -|
    # |      E-----------H   |
    # |      |           |   |
    # |      F-----------G   |
    # B----------------------C
    #
    # The annoying counter-clockwise rotation of these squares is for consistency with PIL.  :(
    #
    # We begin by translating all the points to the origin subtracting the pivot.
    # For each edge of the inner rect,
    #   For each edge of the outer rect,
    #     Find theta 1 where the first vertex of the inner rect's edge intersects the outer edge
    #     Find theta 2 where the second vertex of the inner rect's edge intersects the outer edge
    #     Find theta 3 where the midpoint of the inner rect's edge intersects with the outer edge
    #     If theta 3 does not exist, then the range [theta_1, theta_2] is an open one!
    #     If theta 3 does exist, then the range [theta_1, theta_2] is an invalid one.
    # Carry on building the big set of valid and invalid lists, then do the intersection of all of them.

    valid_spans = []
    invalid_spans = []

    inner_left, inner_up, inner_right, inner_bottom = inner_rectangle
    outer_left, outer_up, outer_right, outer_bottom = outer_rectangle
    inner_points = [
        [inner_left - pivot[0], inner_up - pivot[1]],
        [inner_right - pivot[0], inner_up - pivot[1]],
        [inner_right - pivot[0], inner_bottom - pivot[1]],
        [inner_left - pivot[0], inner_bottom - pivot[1]],
    ]
    outer_points = [
        [outer_left - pivot[0], outer_up - pivot[1]],
        [outer_right - pivot[0], outer_up - pivot[1]],
        [outer_right - pivot[0], outer_bottom - pivot[1]],
        [outer_left - pivot[0], outer_bottom - pivot[1]],
    ]
    for idx in range(0, 4):
        inner_a = inner_points[idx]
        inner_b = inner_points[(idx+1) % 4]
        inner_m = [(inner_a[0]+inner_b[0])*0.5, (inner_a[1]+inner_b[1])*0.5]
        inner_a_radius = math.sqrt(inner_a[0]**2 + inner_a[1]**2)
        inner_b_radius = math.sqrt(inner_b[0] ** 2 + inner_b[1] ** 2)
        inner_m_radius = math.sqrt(inner_m[0] ** 2 + inner_m[1] ** 2)
        for odx in range(0, 4):
            # Get it, like 'index' except outdex!  Ha!
            outer_a = outer_points[odx]
            outer_b = outer_points[(odx+1) % 4]
            # Does inner_a intersect this span at some theta?
            # We already subtracted the pivot, so all of these points are centered at the origin.
            # The circle is formed by inner_a around the origin.
            intersection_points_a = line_circle_intersection(
                outer_a[0],
                outer_a[1],
                outer_b[0],
                outer_b[1],
                0, 0, inner_a_radius
            )
            intersection_points_b = line_circle_intersection(
                outer_a[0],
                outer_a[1],
                outer_b[0],
                outer_b[1],
                0, 0, inner_b_radius
            )
            intersection_points_m = line_circle_intersection(
                outer_a[0],
                outer_a[1],
                outer_b[0],
                outer_b[1],
                0, 0, inner_m_radius
            )
            if len(intersection_points_a) == 0:
                # No collision.  Full range!
                valid_spans.append([0.0, 2.0*math.pi])
            else:
                # Perhaps it's only a tangent graze.
                theta_a_0 = points_to_theta(
                    0, 0,
                    intersection_points_a[0][0],
                    intersection_points_a[0][1],
                )
                theta_a_1 = points_to_theta(
                    0, 0,
                    intersection_points_a[1][0],
                    intersection_points_a[1][1],
                )
                angle_diff = abs(theta_a_0 - theta_a_1)
                if angle_diff < 1e-8:
                    # If we only contact at one point, it's a graze.  No rotation limit.
                    valid_spans.append([0.0, 2.0*math.pi])
                else:
                    # Dang it.  Collision.
                    # This means that we have a valid band.  We need to take the band from our other point.
                    theta_b_0 = points_to_theta(0, 0, intersection_points_b[0][0], intersection_points_b[0][1])
                    theta_b_1 = points_to_theta(0, 0, intersection_points_b[1][0], intersection_points_b[1][1])
                    # We don't know if this intersection happens to be the _invalid_ span where they both touch
                    # the edge or if it's wrapping around and this is the valid span.
                    # Take the smallest of these two and check
                    min_angle = min(theta_a_0, theta_a_1)  # Given our direction of rotation, we want the smallest.
                    max_angle = max(theta_b_0, theta_b_1)
                    theta_m_0 = points_to_theta(0, 0, intersection_points_m[0][0], intersection_points_m[0][1])
                    theta_m_1 = points_to_theta(0, 0, intersection_points_m[1][0], intersection_points_m[1][1])
                    if min_angle < theta_m_0 < max_angle or min_angle < theta_m_1 < max_angle:
                        # This part does collide with a wall.  The OUTER range is the one we want.
                        # Pick the BIGGEST angle that could be invalid.
                        invalid_spans.append([max(theta_a_0, theta_a_1), min(theta_b_0, theta_b_1)])
                    else:
                        valid_spans.append([min_angle, max_angle])
    # TODO: UNION of valid spans and subtraction of invalid spans.

    # Find all spans where the edges intersect and return valid spans.
    return [[0, 2*math.pi]]


def fast_conservative_theta_range(
        inner_box_points: numpy.ndarray,
        outer_box_width: float,
        outer_box_height: float
) -> Optional[Tuple[float, float]]:
    """Find the min/max value that theta can take.  If the inner box can't fit inside the outer box, return None.
    Assumes the inner box pivots around its center.  Assumes the outer box has the left corner at x=0 and top at y=0.

    (0,0)--outer_box_width--+
    |   A--------B          |
    |   |        |   outer_box_height
    |   D--------C         |
    +----------------------+

    Order ABCD does not matter, but the matrix should be 4x2 or 4x3.
    """
    centerpoint = inner_box_points.mean(axis=0)[:2]
    center_x, center_y = centerpoint[:]
    # EDGE CASE: If the pivot is outside of the box, return None.
    if center_x > outer_box_width or center_y > outer_box_height or center_x < 0 or center_y < 0:
        return None
    # Center ABCD at the origin so we can find the 'top right' or 'bottom right'.  The farthest point.
    inner_box_points = inner_box_points[:, :2] - centerpoint
    outer_box_width -= center_x
    outer_box_height -= center_y
    # Remember distance = sqrt(dx*dx + dy*dy).  We can do this in one step via numpy:
    magnitudes = (inner_box_points * inner_box_points).sum(axis=1) ** 0.5
    farthest_point_idx = numpy.argmax(magnitudes)
    far_point = inner_box_points[farthest_point_idx, :]  # 'H' in the equation below.
    radius = magnitudes[farthest_point_idx]
    # Let 'H' be our new point, some distance from the center, the farthest out of the inner points.
    # We want to find where H intersects with the right.
    # r * cos(t) = outer_box_width -> t = acos(outer_box_width / r)
    # That is, when we have a rod of length r and raise it by r radians, we hit the edge.  It's already at start_theta,
    # so we need to bring it back by start_theta-t
    # Same applies for r * sin(t) = outer_box_height, but we need to increase to the angle.

    if abs(far_point[0]) > 1e-8:
        start_theta = math.tan(abs(far_point[1]) / abs(far_point[0]))  # Enforce first quadrant.
    else:
        start_theta = math.pi/2

    if outer_box_width > radius:
        min_angle = 0
    else:
        min_angle = start_theta - math.acos(outer_box_width / radius)

    if outer_box_height > radius:
        max_angle = math.pi
    else:
        max_angle = math.asin(outer_box_height / radius) - start_theta
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
