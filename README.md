TextOverlayDataset
===

A meta-dataset builder to combine text datasets and image datasets.

tl;dr usage:

```python
# %pip install text-overlay-dataset
from TextOverlayDataset import TextOverlayDataset
from torchvision.datasets.fakedata import FakeData as ImgData  # Any image dataset is fine, or just a list of Images.

ds = TextOverlayDataset(image_dataset = ImgData(), text_dataset = ["Hello", "World"], fonts="<path to ttf dir>")

```