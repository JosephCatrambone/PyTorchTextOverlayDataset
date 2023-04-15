"""
test_bounding_box_tools.py
"""

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


if __name__ == '__main__':
    unittest.main()