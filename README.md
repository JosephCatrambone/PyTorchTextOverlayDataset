TextOverlayDataset
===

A meta-dataset builder to combine text datasets and image datasets.

### Recipes:

```python
# Basic Minimal Usage:

# %pip install text-overlay-dataset
from text_overlay_dataset import TextOverlayDataset
from PIL import Image

ds = TextOverlayDataset(
    image_dataset = [Image.new("RGB", (256, 256)), ], 
    text_dataset = ["Hello", "World"], 
    font_directory="<path to ttf dir>"
)

composite_image, text, etc = ds[0]

# composite_image is the 0th image with a randomly selected text.
# text is the given text that was selected.
# etc is an object with axis-aligned bounding box, font name, and so on.

# If desired, one can specify `randomly_choose='image'` in the constructor
# and text will be accessed sequentially with random images instead.
```

```python
# Augmenting the text and making it harder to read by blurring, rotating, etc.

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

```python
# Any torchvision transform can be used as part of the preprocessing.
# Perhaps your model requires images to be cropped to 512x512.
from torchvision.transforms import CenterCrop
ds = TextOverlayDataset(
    image_dataset = fake_image_dataset,
    text_dataset = ["Hello", "World"],  # This can also be a PyTorch text dataset.
    font_directory = "fonts",
    maximum_font_translation_percent=0.4,
    maximum_font_rotation_percent=0.5,
    maximum_font_blur=3.0,
    prefer_larger_fonts=True,
    pre_composite_transforms=[CenterCrop([512,])],
    # post_composite_transforms are also possible.
)
```

```python
# It's possible to try and fill each image with text.
# Set prefer_larger_fonts to use the maximum font size.
ds = TextOverlayDataset(
    image_dataset = fake_image_dataset,
    text_dataset = ["Hello", "World"],  # This can also be a PyTorch text dataset.
    font_directory = "fonts",
    prefer_larger_fonts = True,
    # Or you can specify `font_sizes = [36, 48, ...]`
)
```

```python
# If your dataset has a lot of long strings with no line breaks, it might be worth considering setting 
# 'long_text_behavior' to 'truncate_then_shrink' to avoid lots of null texts. 
ds = TextOverlayDataset(
    image_dataset = fake_image_dataset,
    text_dataset = ["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA!!!!!!"],
    font_directory = "fonts",
    long_text_behavior = 'truncate_then_shrink',
)
```

### TODO:
- ~~Add toggle to prefer larger fonts first?~~
- ~~Fix bounds checking on rotation so we don't put text off the edge of the image.~~
- Add automatic line-breaking to fix long text inside image areas.
- Check for sampling biases in the random generations.
- Support streaming datasets.
- Verify RTL languages.
- Verify Unicode line breaks and non-English fonts.