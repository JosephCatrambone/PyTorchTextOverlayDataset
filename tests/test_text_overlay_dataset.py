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

from src.text_overlay_dataset import TextOverlayDataset


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
            maximum_font_translation_percent=1.0,
            maximum_font_rotation_percent=1.0,
        )

    def test_sanity_lists(self):
        """Verify we can run the happy path of instancing a dataset and pulling one example."""
        ds = self._make_small_dataset()
        img = ds[0]
        self.assertTrue(img is not None, "Dataset generator yielded 'None'")

    def test_fuzz_generate_text_raster_augmented_bounds_check(self):
        """Randomly hit generate_text_raster_augmented and verify that it is always within the image bounds."""
        ds = self._make_small_dataset()
        for _ in range(1000):
            width = random.randint(128, 512)
            height = random.randint(128, 512)
            success, _, bbox = ds._generate_text_raster_advanced("testing", width, height)
            if not success:
                continue
            for point_idx in range(bbox.shape[0]):
                in_bounds = 0 <= bbox[point_idx, 0] <= width and 0 <= bbox[point_idx, 1] <= height
                if not in_bounds:
                    print(width, height)
                    print(bbox)
                assert in_bounds

    def test_raise_on_unplaceable_text(self):
        text_dataset = ["One really long string that couldn't possibly fit in a thing.",]
        image_dataset = [Image.new("RGB", (1, 1)), ]
        ds = TextOverlayDataset(
            image_dataset,
            text_dataset,
            font_directory="fonts",
            empty_string_on_truncation=False,
        )
        with self.assertRaises(ValueError):
            _ = ds[0]


if __name__ == '__main__':
    unittest.main()