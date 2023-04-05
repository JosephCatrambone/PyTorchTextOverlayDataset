"""
text_image_dataset.py
"""
import os
import random
from glob import glob
from typing import List, Optional, Tuple, Union

import numpy
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor


class TextOverlayDataset(Dataset):
    """
    TextOverlayDataset combines an image dataset and a text dataset (or collection of strings).
    The image dataset should return images that can be converted to RGB.
    """
    def __init__(
            self,
            image_dataset: Dataset,
            text_dataset: Union[List[str], Dataset],
            font_directory: Union[os.PathLike, str],

            pre_composite_transforms: Optional[List] = None,
            post_composite_transforms: Optional[List] = None,
            text_raster_transforms: Optional[List] = None,

            randomly_choose: str = "text",
            empty_string_on_truncation: bool = True,
            font_sizes: Optional[List[int]] = None,

            only_english_support: bool = False,

            # TODO: Unused!
            maximum_font_outline_size: int = 5,
            maximum_font_blur: float = 2.0,
    ):
        """Generate a new TextOverlayMappingDataset from an image dataset, text, and set of fonts.

        TextOverlayDataset will attempt to dynamically generate and return triplets of composited text images, text,
        and text raster.

        Parameters
        ----------

        :param Dataset | List[Image] image_dataset:
        A mappable PyTorch dataset OR a list of images.

        :param Dataset | List[str] text_dataset:
        A mappable PyTorch text dataset or list of strings.

        :param os.PathLike | str font_directory:
        A user-readable directory containing true-type and open-type font files.
        font_directory should not have a trailing slash and should contain one or more TTF/OTFs.

        :param list pre_composite_transforms:
        The torchvision transforms that will be called (in order) on the sampled image.
        There are no guarantees about the Image format (numpy or pil or tensor).

        :param list post_composite_transforms:
        A convenience method.  They will be run on the output image before it is returned.
        A user could perform all of these transforms on the returned value.  Expect a PIL image as input.

        :param list text_raster_transforms:
        The text_raster_transforms_override are the data-augmentations that will be run on the rasterized text before
        it is composited onto the dataset image. WARNING NOTE: if any transform removes the alpha channel, this can
        seriously skew results.  Data augmentations are generated based on the other configuration parameters and are
        performed before the rasterized text is composited onto the image.

        :param str randomly_choose:
        If randomly_choose 'text' (default), this_dataset[i] will give the i-th item from the image dataset with a
        randomly selected item from the text dataset superimposed.  len(this) is then defined as the length of the image
         dataset.
        If randomly_choose 'image', then this_dataset[i] will give a random image with the i-th entry in the text
        dataset.  The length of this is will be defined as len(text_dataset).
        If randomly_chose is None, this_dataset[i] will return text[i] superimposed on image[i].  The length of the
        dataset will be min(image_dataset, text_dataset).

        :param bool empty_string_on_truncation:
        Defaults to true.  If an image is too small to fit the given text sample, return an empty string and do not
        perform any text compositing.  If 'false', will raise an exception on truncation.

        :param List[int] font_sizes:
        A list of valid font sizes from which we can select.  If 'None' (default), will use [8, 10, 12, 16, 20, 24, 36,
        48, 72, 144].

        :param bool only_english_support:
        If false (default), will use ImageFont.Layout.RAQM for performing font layout.  This is slightly lower but
        enables non-English sentences.  If your text is solely English (LTR, Ascii, etc), you may see slightly better
        performance by setting this to 'false'.
        """
        super(TextOverlayDataset, self).__init__()

        self.image_dataset = image_dataset
        self.text_dataset = text_dataset

        self.pre_composite_transforms = pre_composite_transforms
        self.pre_composite_transforms = post_composite_transforms
        self.text_raster_transforms = text_raster_transforms

        # PIL font loading behaves a little strangely when shared across threads.  We have to load our fonts after the
        # dataset is forked or we get sharing problems.
        self.font_choices = glob(str(font_directory) + os.pathsep + "*.*tf")  # *.*tf gives us TTF and OTF.
        self.loaded_fonts = None  # This will become a dict [str -> font].

        # For dataset iteration, do we need to randomly sample from one of the input datasets?
        self.randomize_text = False
        self.randomize_image = False
        if randomly_choose is None or randomly_choose == "none":
            pass
        elif randomly_choose == "text":
            self.randomize_text = True
        elif randomly_choose == "image":
            self.randomize_image = True
        else:
            raise ValueError(f"Unrecognized value for 'randomly choose': {randomly_choose}")

        self.empty_string_on_truncation = empty_string_on_truncation

        if font_sizes is None:
            self.font_sizes = [8, 10, 12, 16, 20, 24, 36, 48, 72, 144]
        else:
            self.font_sizes = sorted(font_sizes)

        self.layout = ImageFont.LAYOUT_RAQM
        if only_english_support:
            self.layout = ImageFont.LAYOUT_BASIC

    def __len__(self):
        if self.randomize_text:
            return len(self.image_dataset)
        elif self.randomize_image:
            return len(self.text_dataset)
        return min([len(self.image_dataset), len(self.text_dataset)])

    def __getitem__(self, idx):
        """Given an index, yields a tuple of the following:
         - image (as a numpy array)
         - the text overlaid upon the image (string)
         - the un-composited text (a text overlay as a numpy array)
        """
        pass

    def get_image(self, idx):
        """Return either the image at position idx or a random image, depending on the value of 'randomly_chose'."""
        if self.randomize_image:
            return self.image_dataset[random.randint(0, len(self.image_dataset))]
        return self.image_dataset[idx]

    def get_text(self, idx):
        """Return either the text at position idx or a random string, depending on the value of 'randomly_chose'."""
        if self.randomize_text:
            return self.text_dataset[random.randint(0, len(self.text_dataset))]
        return self.text_dataset[idx]

    def _generate_text_raster(self, text: str, width: int, height: int, max_retries: int = 10) -> Image.Image:
        """
        We will try to pick a font, layout  the text, and then fit it into the image up to X times.

        :param text:
        :param width:
        :param height:
        :return:
        """
        if self.loaded_fonts is None:
            self.loaded_fonts = dict()

        text_width = width + 1
        text_height = height + 1

        while max_retries > 0:  # We check again at the tail of this function and may break out there instead.
            max_retries -= 1
            alignment = random.choice(["left", "center", "right"])
            size_idx = random.randint(0, len(self.font_sizes))  # We pick a random starting size for our font.
            font_size = self.font_sizes[size_idx]
            font_choice = random.choice(self.font_choices)

            while (text_width > width or text_height > height) and size_idx >= 0:
                if font_choice not in self.loaded_fonts:
                    # A note on this:
                    # On Windows the TrueType system keeps fonts open until the TTF object goes out of scope.  This caps
                    # the number of open fonts to 512 and can lead to OSErrors on load.  We get around this by copying
                    # the font into memory first.
                    with open(font_choice, 'rb') as fin:
                        self.loaded_fonts[font_choice] = fin.read()
                font = ImageFont.truetype(
                    self.loaded_fonts[font_choice],
                    font_size,
                    layout_engine=self.layout
                )
                # Try and generate the text.
                canvas = Image.new('RGB', (width, height), color='black')
                draw = ImageDraw.Draw(canvas)
                text_bbox = draw.textbbox((width//2, height//2), text, font=font, anchor="mm", align=alignment)
                left, top, right, bottom = text_bbox
                text_width = right-left
                text_height = abs(top-bottom)
                # We may have to recenter the text.
                if text_width < width and text_height < height:
                    draw.text((width//2, height//2), text, font=font, anchor="mm", align=alignment)
                    return canvas
                size_idx -= 1
        # If we're here, we've run out of retries.
        # Cannot fit the given text in the image.
        # TODO: Raise exception OR return empty, depending on setting.

    def _generate_image_mask_pair(self, index: int) -> Tuple[Image.Image, str, Image.Image]:
        img = self.get_image(index)
        text = self.get_text(index)

        # Run the pre-composite transformations.
        if self.pre_composite_transforms is not None:
            for tf in self.pre_composite_transforms:
                img = tf(img)

        # It's possible the outputs of these transforms is not a PIL image:
        if isinstance(img, Image.Image):
            pass  # NOOP
        elif isinstance(img, (numpy.ndarray, numpy.generic, torch.tensor)):
            if isinstance(img, torch.Tensor):
                # Pull this instance to the CPU from the GPU, then treat it as though it were a numpy array.
                img = img.detach().numpy()
            img = Image.fromarray(img)

        # Generate some random text:
        text_image_mask, text, text_rotation = self.random_text_image()
        text_image_mask = text_image_mask.convert('L')

        # Glorious hack to make a red mask:
        # red_channel = img_pil[0].point(lambda i: i < 100 and 255)

        # Draw the text image on top of our sample image.
        # if (red * 0.299 + green * 0.587 + blue * 0.114) > 186 use  # 000000 else use #ffffff
        total_color = [0, 0, 0]
        total_pixels = 0
        for y in range(self.target_height):
            for x in range(self.target_width):
                mask = text_image_mask.getpixel((x, y))
                if mask > 128:
                    px = img_pil.getpixel((x, y))
                    total_color[0] += px[0]
                    total_color[1] += px[1]
                    total_color[2] += px[2]
                    total_pixels += 1

        # In the off chance we have a completely blank image, save some compute.
        if total_pixels > 0:
            avg_r = total_color[0] // total_pixels
            avg_g = total_color[1] // total_pixels
            avg_b = total_color[2] // total_pixels

            # Default to light color...
            text_color = [random.randint(150, 255), random.randint(150, 255), random.randint(150, 255)]
            if (
                    avg_r * 0.299 + avg_g * 0.587 + avg_b * 0.114) > 125:  # 186 comes from the algorithm but is a little hard to see.
                # Unless our image is bright, in which case use dark color.
                text_color = [random.randint(0, 75), random.randint(0, 75), random.randint(0, 75)]
            # Have to convert text color to a hex string.  :rolleges:
            text_color = f"#{text_color[0]:02X}{text_color[1]:02X}{text_color[2]:02X}"
            # Make a rectangle of this color and paste it in with an image mask.
            text_color_block = Image.new("RGB", (self.target_width, self.target_height), color=text_color)
            # Maybe add noise to the color block.
            if self.text_noise > 0:
                pass
            img_pil.paste(text_color_block, (0, 0), text_image_mask)

        # Now dilate the text_image_mask to simulate highlighting a block.
        if self.random_text_mask_dilation > 0:
            text_image_mask = text_image_mask.filter(ImageFilter.MaxFilter(self.random_text_mask_dilation))

        return img_pil, text_image_mask, text, text_rotation
