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


if __name__ == '__main__':
    unittest.main()