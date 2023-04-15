"""
test_text_overlay_dataset.py

Usage:
python -m unittest tests/test_text_overlay_dataset.py
OR
python -m unittest test.test_text_overlay_dataset.py
OR
python -m unittest
"""

import random
import unittest

from PIL import Image
from torchvision.datasets import FakeData

from src.TextOverlayDataset.text_image_dataset import TextOverlayDataset


class TestTextOverlayDataset(unittest.TestCase):
    def _make_empty_dataset(self):
        return TextOverlayDataset(image_dataset=[], text_dataset=[], font_directory="fonts")

    def _make_small_dataset(self):
        image_dataset = [
            Image.new("RGB", (512, 512)),
            Image.new("RGB", (512, 512)),
            Image.new("RGB", (512, 512)),
        ]
        text_dataset = [
            "Is this real life?",
            "Is this just fantasy?",
            "Caught in a landslide, \nno escape from reality."
        ]
        font_dir = "./fonts"

        return TextOverlayDataset(
            image_dataset=image_dataset,
            text_dataset=text_dataset,
            font_directory=font_dir,
        )

    def test_sanity_lists(self):
        """Verify we can run the happy path of instancing a dataset and pulling one example."""
        ds = self._make_small_dataset()
        img = ds[0]
        self.assertTrue(img is not None, "Dataset generator yielded 'None'")

    def test_compute_min_max_text_angle_unbounded(self):
        """Verify that a max and min angles of a free-spinning block in a rectangle are properly calculated."""
        image_width = 100
        image_height = 50
        text_width = 40
        text_height = 20
        # This text box can spin freely in the image rectangle.

        ds = self._make_empty_dataset()
        min_angle, max_angle = ds._compute_min_max_text_angle(
            (-text_width//2, -text_height//2, text_width//2, -text_height//2),
            image_width,
            image_height
        )
        self.assertLess(min_angle, 1e-6, "Min angle for a free-spinning text block should be near zero.")
        self.assertGreater(max_angle, 6.28, "Max angle for a free-spinning text block should be 2*PI.")

    def test_compute_min_max_text_angle_bounded(self):
        """Verify that if a block of text would not fit in a rectangle at certain rotations we demark them."""
        ds = self._make_empty_dataset()
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

    def test_fuzz_generate_text_raster_augmented_bounds_check(self):
        """Randomly hit generate_text_raster_augmented and verify that it is always within the image bounds."""
        ds = self._make_small_dataset()
        for _ in range(10000):
            width = random.randint(128, 512)
            height = random.randint(128, 512)
            success, _, bbox = ds._generate_text_raster_advanced("testing", width, height, 1.0, 1.0)
            if not success:
                continue
            for point_idx in range(bbox.shape[0]):
                assert 0 <= bbox[point_idx, 0] <= width and 0 <= bbox[point_idx, 1] <= height

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)


if __name__ == '__main__':
    unittest.main()