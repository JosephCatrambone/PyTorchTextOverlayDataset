"""
test_bounding_box_tools.py
"""

import math
import unittest

import numpy

from src.text_overlay_dataset.bounding_box_tools import *


class TestBoundingBoxTools(unittest.TestCase):
    def test_aabb_round_trip(self):
        left, top, right, bottom = 1.0, 2.0, 3.0, 4.0
        aabb = bbox_to_aabb(left, top, right, bottom)
        bbox = aabb_to_bbox(aabb)
        self.assertEqual(bbox, (left, top, right, bottom))

    def test_aabb_to_bbox(self):
        wonky_quad = numpy.asarray([
            [-10, -5, 1],
            [4, 8, 1],
            [2, 11, 1],
            [-4, 3, 1],
        ])
        bbox = aabb_to_bbox(wonky_quad)
        left, top, right, bottom = bbox
        self.assertEqual(left, -10)
        self.assertEqual(top, -5)
        self.assertEqual(right, 4)
        self.assertEqual(bottom, 11)
        self.assertEqual(bbox, (-10, -5, 4, 11))
        #with self.assertRaises(TypeError):

    def test_generate_rotation_matrix(self):
        rotmat_0 = make_rotation_matrix_2d(0.0)
        self.assertTrue(numpy.allclose(numpy.eye(3), rotmat_0))

        rotmat_90 = make_rotation_matrix_2d(math.pi/2)
        # Recall: cos(90) = 0.  sin(90) = 1
        self.assertTrue(numpy.allclose(numpy.asarray([
            [0, 1, 0.0],
            [-1, 0, 0.0],
            [0, 0, 1.0],
        ]), rotmat_90))

    def test_rotation(self):
        # DEBUG: Just visualizing the rotation process.
        point_at_x1 = numpy.asarray([[1, 0, 1.0]])

        point_at_y1 = rotate_around_point(point_at_x1, math.pi/2.0, 0, 0)  # rotate about the origin.
        self.assertAlmostEqual(point_at_y1[0, 0], 0.0)
        self.assertAlmostEqual(point_at_y1[0, 1], 1.0)

        for i in range(0, 100):
            rotation = (2*math.pi / 100)*i
            rotated_point = rotate_around_point(point_at_x1, rotation, 0, 0)
            self.assertAlmostEqual(rotated_point[0, 0], math.cos(rotation))
            self.assertAlmostEqual(rotated_point[0, 1], math.sin(rotation))

    def test_compute_min_max_text_angle_unbounded(self):
        """Verify that a max and min angles of a free-spinning block in a rectangle are properly calculated."""
        # This text box can spin freely in the image rectangle.
        limits = fast_conservative_theta_range(
            inner_box_points=bbox_to_aabb(100, 100, 300, 200),  # 200 wide, 100 tall, centered at about 200, 150
            outer_box_width=500,
            outer_box_height=400
        )
        self.assertTrue(limits is not None)
        min_angle, max_angle = limits
        self.assertAlmostEqual(min_angle, 0.0, 6, "Min angle for a free-spinning text block should be near zero.")
        self.assertAlmostEqual(max_angle, math.pi, 6, "Max angle for a free-spinning text block should be PI/2.")

    def test_compute_min_max_text_angle_bounded(self):
        """Verify that if a block of text would not fit in a rectangle at certain rotations we demark them."""
        # We expect to bonk the top, so there's a max but no min.
        box_width = 200.0 / 2.0
        box_height = 0.0 / 2.0
        box_center = numpy.asarray([1000.0, 1.0])  # X, Y
        image_size = numpy.asarray([5000.0, 2.0])  # Super wide, but not tall.
        aabb = numpy.asarray([
            [box_center[0] - box_width, box_center[1] - box_height, 1],
            [box_center[0] - box_width, box_center[1] + box_height, 1],
            [box_center[0] + box_width, box_center[1] + box_height, 1],
            [box_center[0] + box_width, box_center[1] - box_height, 1],
        ])
        limits = fast_conservative_theta_range(aabb, image_size[0], image_size[1])
        self.assertAlmostEqual(limits[0], 0.0, 5, "Lower limit isn't calculated correctly.")
        self.assertAlmostEqual(limits[1], 0.010000166674167114, 5, "Upper limit isn't calculated correctly.")
        new_pts = rotate_around_point(aabb, limits[1], box_center[0], box_center[1])
        self.assertAlmostEqual(new_pts.max(axis=0)[1], image_size[1], 5, "After rotation about max, no ceiling hit.")


if __name__ == '__main__':
    unittest.main()