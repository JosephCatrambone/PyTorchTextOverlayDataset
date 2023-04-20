"""
test_bounding_box_tools.py
"""

import math
import unittest

import numpy

from src.TextOverlayDataset.bounding_box_tools import *


class TestBoundingBoxTools(unittest.TestCase):
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
        image_width = 100
        image_height = 50
        text_width = 40
        text_height = 20
        # This text box can spin freely in the image rectangle.

        min_angle, max_angle = compute_max_rect_in_rect_rotation(
            pivot=(0.0, 0.0),
            inner_rectangle=(-text_width//2, -text_height//2, text_width//2, text_height//2),
            outer_rectangle=(-image_width//2, -image_height//2, image_width//2, image_height//2)
        )
        self.assertLess(min_angle, 1e-6, "Min angle for a free-spinning text block should be near zero.")
        self.assertGreater(max_angle, 6.28, "Max angle for a free-spinning text block should be 2*PI.")

    def test_compute_min_max_text_angle_bounded(self):
        """Verify that if a block of text would not fit in a rectangle at certain rotations we demark them."""
        # Generate an image that's 10 units wide and almost 1 unit tall.
        # Generate a text block that's 2 units wide and 0 units tall.
        # It's basically a horizontal paddle that will sit upright, like a throttle mechanism.
        # The text block will hit when sin(theta) = 1, which should be 90.
        min_angle, max_angle = ds._compute_min_max_text_angle([-1, 0, 1, 0], 10.0, 1.0-0.000001)
        print(min_angle)
        print(max_angle)

        # Anecdotal case:
        image_rect_half_size = [200, 100]  # Enclosing image is twice this width and twice this height.
        text_rect_half_size = [199, 99]  # Almost touching the edge.
        min_angle, max_angle = ds._compute_min_max_text_angle(
            (
                image_rect_half_size[0] - text_rect_half_size[0],  # Left
                image_rect_half_size[1] - text_rect_half_size[1],  # Top
                image_rect_half_size[0] + text_rect_half_size[0],  # Right
                image_rect_half_size[1] + text_rect_half_size[1],  # Bottom
            ),
            image_rect_half_size[0]*2,
            image_rect_half_size[1]*2
        )

        self.assertLess(abs(min_angle - -0.010205872253258474), 1e-6, "Min angle for constrained square is incorrect.")
        self.assertLess(abs(max_angle - 0.005031443897310306), 1e-6, "Max angle for a constrained square is incorrect.")


if __name__ == '__main__':
    unittest.main()