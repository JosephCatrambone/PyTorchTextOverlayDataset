TextOverlayDataset
===

A meta-dataset builder to combine text datasets and image datasets.

### At-a-glance usage:
```python
# %pip install text-overlay-dataset
from text_overlay_dataset import TextOverlayDataset
from PIL import Image

ds = TextOverlayDataset(
    image_dataset = [Image.new("RGB", (256, 256)), ], 
    text_dataset = ["Hello", "World"], 
    fonts="<path to ttf dir>"
)

composite_image, text, text_raster, aabb = ds[0]
```

### Intermediate Usage:
```python
from text_overlay_dataset import TextOverlayDataset
from torchtext.datasets import IMDB  # A text dataset should be mappable.
from torchvision.datasets.fakedata import FakeData  # Any mappable image dataset is fine, or just a list of Images.

image_dataset = FakeData(size=100, image_size=(3, 256, 256),)

text_dataset_iter = IMDB(split='train')
text_dataset = [label_text[1] for label_text in text_dataset_iter] 

ds = TextOverlayDataset(
    image_dataset,
    text_dataset,
    font_directory="./fonts/",
    maximum_font_translation_percent=0.5,
    maximum_font_rotation_percent=0.25,
    maximum_font_blur=3.0
)
```

### TODO:
- Add toggle to prefer larger fonts first?
- Fix bounds checking on rotation so we don't put text off the edge of the image.
- Add automatic line-breaking to fix text inside image areas.