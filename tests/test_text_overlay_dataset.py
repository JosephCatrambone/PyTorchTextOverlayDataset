"""
test_text_overlay_dataset.py

Usage:
python -m unittest tests/test_text_overlay_dataset.py
OR
python -m unittest test.test_text_overlay_dataset.py
OR
python -m unittest
"""

import unittest

from PIL import Image
from torchvision.datasets import FakeData

from src.TextOverlayDataset.text_image_dataset import TextOverlayDataset


class TestTextOverlayDataset(unittest.TestCase):
    def test_sanity_lists(self):
        """Verify we can run the happy path of instancing a dataset and pulling one example."""
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
        font_dir = "fonts"

        ds = TextOverlayDataset(
            image_dataset=image_dataset,
            text_dataset=text_dataset,
            font_directory=font_dir,
        )
        img = ds[0]
        self.assertTrue(img is not None, "Dataset generator yielded 'None'")

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)


if __name__ == '__main__':
    unittest.main()