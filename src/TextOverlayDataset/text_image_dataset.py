"""
text_image_dataset.py
"""
import math
import os
import random
from glob import glob
from io import BytesIO
from typing import List, Optional, Tuple, Union

import numpy
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms.functional import to_tensor, to_pil_image

from .bounding_box_tools import aabb_to_bbox, bbox_to_aabb, rotate_around_point


class TextOverlayDataset(Dataset):
    """
    TextOverlayDataset combines an image dataset and a text dataset (or collection of strings).
    The image dataset should return images that can be converted to RGB.
    """
    def __init__(
            self,
            image_dataset: Union[List[Image.Image], Dataset],
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
        There are no guarantees about the Image format (numpy or pil or tensor). If your dataset returns non-Image data
        it might be necessary to add a transform to convert it into a PIL image.

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
        If true (default), when text cannot be written to an image in a way that is legible, either because the image
        itself is too small to support a given string or a string is too large to be fit on any image (e.g., trying to
        fit War and Peace on a 16x16 image), the generator will instead return an empty text and an empty image.
        If this is false, will raise an exception.

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
        self.post_composite_transforms = post_composite_transforms
        self.text_raster_transforms = text_raster_transforms

        # PIL font loading behaves a little strangely when shared across threads.  We have to load our fonts after the
        # dataset is forked or we get sharing problems.
        self.font_choices = glob(os.path.join(font_directory, "*.*tf"))  # *.*tf gives us TTF and OTF.
        assert self.font_choices, f"No fonts detected in font_directory: {font_directory}"
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
         - a numpy array with the four corners of the bounding box
        """
        composite, text, raster, bounding_box = self._generate_image_mask_pair(idx)
        return composite, text, raster, bounding_box

    def get_image(self, idx):
        """Return either the image at position idx or a random image, depending on the value of 'randomly_chose'."""
        if self.randomize_image:
            return self.image_dataset[random.randint(0, len(self.image_dataset)-1)]
        return self.image_dataset[idx]

    def get_text(self, idx):
        """Return either the text at position idx or a random string, depending on the value of 'randomly_chose'."""
        if self.randomize_text:
            return self.text_dataset[random.randint(0, len(self.text_dataset)-1)]
        return self.text_dataset[idx]

    def _generate_text_raster_basic(
            self,
            text: str,
            width: int,
            height: int,
            max_retries: int = 10
    ) -> Tuple[bool, Image.Image, Tuple[int, int, int, int]]:
        """
        We will try to pick a font, layout the text, and then fit it into the image up to X times.
        :param text:
        :param width:
        :param height:
        :return: A tuple of success/failure, a LUMA PIL image, and a bounding box of left, up, right, bottom.
        """
        if self.loaded_fonts is None:
            self.loaded_fonts = dict()

        text_width = width + 1
        text_height = height + 1

        while max_retries > 0:  # We check again at the tail of this function and may break out there instead.
            max_retries -= 1
            alignment = random.choice(["left", "center", "right"])
            size_idx = random.randint(0, len(self.font_sizes)-1)  # We pick a random starting size for our font.
            font_size = self.font_sizes[size_idx]
            font_choice = random.choice(self.font_choices)

            while (text_width > width or text_height > height) and size_idx >= 0:
                if font_choice not in self.loaded_fonts:
                    # A note on this:
                    # On Windows the TrueType system keeps fonts open until the TTF object goes out of scope.  This caps
                    # the number of open fonts to 512 and can lead to OSErrors on load.  We get around this by copying
                    # the font into memory first.
                    print(font_choice)
                    with open(font_choice, 'rb') as fin:
                        buffer = BytesIO()
                        buffer.write(fin.read())
                        self.loaded_fonts[font_choice] = buffer
                self.loaded_fonts[font_choice].seek(0)
                font = ImageFont.truetype(
                    self.loaded_fonts[font_choice],
                    font_size,
                    layout_engine=self.layout
                )
                # Try and generate the text.
                canvas = Image.new('L', (width, height), color=0)
                draw = ImageDraw.Draw(canvas)
                text_bbox = draw.textbbox((width//2, height//2), text, font=font, anchor="mm", align=alignment)
                left, top, right, bottom = text_bbox
                text_width = right-left
                text_height = abs(top-bottom)
                # We may have to recenter the text.
                if text_width < width and text_height < height:
                    draw.text((width//2, height//2), text, font=font, anchor="mm", align=alignment, fill=255)
                    return True, canvas, (left, top, right, bottom)
                size_idx -= 1
        # If we're here, we've run out of retries.
        # Cannot fit the given text in the image.
        # TODO: Raise exception OR return empty, depending on setting.
        return False, None, None

    def _generate_text_raster_advanced(
            self,
            text: str,
            width: int,
            height: int,
            max_rotation_percent: float = 0.0,
            max_translation_percent: float = 0.0,
    ) -> Tuple[bool, Image.Image, numpy.ndarray]:
        """Move the rasterized text around, keeping all corners inside the width/height.
        Returns success/failure, the image, and the points on the axis-aligned bounding box as a 4x3 array.
        If max_rotation_percent is zero and max_translation_percent is zero this is equivalent to the
        _generate_text_raster_basic method and will behave similarly (except for returning aabb instead of bbox).
        """
        success, text_image_mask, text_bbox = self._generate_text_raster_basic(text, width, height)
        if not success:
            # JC: It would be nice to keep this signature in sync with the _basic version.  Does it help to enforce it
            # here, or should we return False, None, None?
            return success, text_image_mask, text_bbox
        assert text_image_mask.mode == 'L'

        aabb = bbox_to_aabb(*text_bbox)

        # If there are no translation or rotation operations, save the compute and return early.
        if abs(max_rotation_percent) < 1e-6 and abs(max_translation_percent) < 1e-6:
            return success, text_image_mask, aabb

        if max_rotation_percent > 0.0:
            min_angle, max_angle = self._compute_min_max_text_angle(text_bbox, width, height)
            rotation = random.uniform(0, (max_angle-min_angle)*max_rotation_percent) + min_angle
            aabb = rotate_around_point(aabb, rotation, width*0.5, height*0.5)

        # In theory it would be possible to do the translation and rotation in one matmul, but we don't know the limits
        # of the translation before we do the rotation.
        previous_bbox = text_bbox
        text_bbox = aabb_to_bbox(aabb)
        # Some sanity checking.  Our compute method should have made sure that we can't rotate the quad past the edge.
        in_bounds = \
            0 <= text_bbox[0] <= width and \
            0 <= text_bbox[1] <= height and \
            0 <= text_bbox[2] <= width and \
            0 <= text_bbox[3] <= height
        if not in_bounds:
            print(previous_bbox)
            print(aabb)
            print(text_bbox)
            print(width)
            print(height)
        assert in_bounds

        # Prep translation:
        dx = 0
        dy = 0
        if max_translation_percent > 0:
            # Text bbox is, coincidentally, the amount we can jitter the text left/right and top/bottom.
            # JC: Variable name choice: Slop?  Play?  Tolerance?
            max_left_movement = -int(text_bbox[0])
            max_up_movement = -int(text_bbox[1])
            max_right_movement = width - int(text_bbox[2])
            max_down_movement = height - int(text_bbox[3])
            dx = random.randrange(
                int(max_translation_percent * max_left_movement),
                int(max_translation_percent * max_right_movement)
            )
            dy = random.randrange(
                int(max_translation_percent * max_up_movement),
                int(max_translation_percent * max_down_movement)
            )
        # Translate by this amount.
        aabb += numpy.asarray([dx, dy, 0])  # Lean on broadcasting.
        # PIL expects the _source_ quad to map to the full width, not the target quad.  We can just invert the op.
        inv_aabb = numpy.asarray([
            [0, 0, 1],
            [width, 0, 1],
            [width, height, 1],
            [0, height, 1],
        ]) - aabb
        # This one-line magic converts our 2D array of [[x, y], [x, y], ...] to a list of [x, y, x, y, ...]
        quad = inv_aabb[:, :2].reshape(1, -1)[0]
        #transform = ImageTransform.QuadTransform(quad)  # Perhaps outdated?
        text_image_mask = text_image_mask.transform(text_image_mask.size, Image.QUAD, quad)

        return True, text_image_mask, aabb

    def _compute_min_max_text_angle(
            self,
            text_bbox: Tuple[int, int, int, int],
            image_width: int,
            image_height: int
    ) -> Tuple[float, float]:
        """
        Compute the maximum rotation and translation that we can apply to this block of text.
        There's a little basic trig in here.  I think there's a more optimal way to determine this value, but it
        requires a better mind.

        ```
        +------ Width_img / 2 ----------+
        |                               |
        +-- Width_txt/2 ----(A)       Height_img/2
        |                    |          |
        | Theta_txt    Height_txt/2     |
        +--------------------+----------+
        ```

        The point A is the top-right corner of the rectangle defined by the text.
        We want to ensure that A is below height_img/2 and to the left of width_img/2.
        Because the rectangle is balanced and symmetric, we ensure that the other corners are inside the rect, too.
        Theta_txt refers to the rotation of the whole text box and starts at zero.
        Theta_a (not labeled) is the angle of A when Theta_txt is zero.  (A.k.a., the starting angle of point A.)
        Theta_a = atan(height_txt/2 / width_txt/2), which simplifies to atan(height_txt/width_txt).
        A moves to the right and reaches max when Theta_txt hits -Theta_A.
        A moves up and reaches a max when Theta_A is pi/2 (90), so a minx/maxy at Theta_txt + Theta_a = pi/2.
        We have two cases to check for A_x and A_y.  If A_x is less than or equal to Width_img/2 at max and A_y is less
        than or equal to Height_img/2, we have no constraint on the rotation.
        :param text_bbox: The [left, top, right, bottom] of the text. Assumed to be centered at the box's center.
        :param image_width: The full width of the enclosing image.
        :param image_height: The full height of the enclosing image.
        :return Tuple[float, float]: The minimum and maximum angles or (0, 2*pi) if there is no limit.
        """
        text_left, text_top, text_right, text_bottom = text_bbox
        text_halfwidth = abs(text_right - text_left) * 0.5
        text_halfheight = abs(text_top - text_bottom) * 0.5
        text_radius = math.sqrt(text_halfwidth**2 + text_halfheight**2)
        angle_pa = math.atan2(text_halfheight, text_halfwidth)  # The 0.5 cancels in atan(dy*0.5 / dx*0.5).
        min_angle = 0.0
        max_angle = 2.0*math.pi
        cos_min_angle = image_width*0.5 / text_radius
        sin_max_angle = image_height*0.5 / text_radius
        if -1 < cos_min_angle < 1:
            # We are constrained.  :(
            min_angle = math.acos(cos_min_angle) - angle_pa
        if -1 < sin_max_angle < 1:
            # Constrained by ceiling.
            max_angle = math.asin(sin_max_angle) - angle_pa
        return min_angle, max_angle

    def _generate_image_mask_pair(self, index: int) -> Tuple[Image.Image, str, Image.Image, numpy.ndarray]:
        """
        Generate a 4-tuple of composited image+text, the text, the text mask, and a numpy array of shape 4x2 with the
        bounding box corners.

        There are three sections to this function:
        - Image preloading and prep where we do all the pre-transforms to ensure our image is in the right format,
        - Text raster augmentation, where we take and transform the text raster.
        - Text raster composition, where we composite the rasterized text in a nice color over the image.

        :param index:
        :return Tuple[Image.Image, str, Image.Image, numpy.generic]:
        """
        img = self.get_image(index)
        text = self.get_text(index)

        assert img, f"img at index {index} is null!"

        # Run the pre-composite transformations.
        if self.pre_composite_transforms is not None:
            for tf in self.pre_composite_transforms:
                img = tf(img)

        # It's possible the outputs of these transforms is not a PIL image:
        assert img, "One or more pre_compose_transforms has yielded a null. Make sure your functions return values."
        img_pil = img.copy()  # Avoid messing with the original dataset if we happen to be getting refs.
        img_arr = numpy.array(img_pil)
        if isinstance(img, Image.Image):
            pass  # Noop.  Got things as expected.
        elif isinstance(img, (numpy.ndarray, numpy.generic, torch.Tensor)):
            img_pil = to_pil_image(img).convert("RGB")
            # Image.fromarray(img) won't convert the channels-first format.
            # We still have to sanity check this.
            # TODO: It's wasteful to convert to PIL and immediately back to a tensor, but it guarantees we get HWC.
            img_arr = numpy.array(img_pil)

        # Generate some random text:
        success, text_image_mask, bbox = self._generate_text_raster_advanced(text, img_pil.width, img_pil.height)
        # It's possible the text we tried to add could not be composited onto an image.
        if not success:
            if self.empty_string_on_truncation:
                return img_pil, "", Image.new('RGB')

        # Glorious hack to make a red mask:
        # red_channel = img_pil[0].point(lambda i: i < 100 and 255)

        # This next operation requires a quick iteration over the input image to determine a good color to use.
        # We would do well to pick a color outside of any used in the region, but that's a hard problem.
        # For now, use the simply bright/dark heuristic and figure out something more clever later.
        # TODO: Find a better system of determining font color.
        # if (red * 0.299 + green * 0.587 + blue * 0.114) > 186 use  # 000000 else use #ffffff
        total_pixels = (img_pil.width*img_pil.height)+1e-6
        avg_r, avg_g, avg_b = img_arr.sum(axis=0).sum(axis=0) / total_pixels

        # Default to light color...
        text_color = [random.randint(150, 255), random.randint(150, 255), random.randint(150, 255)]
        if (avg_r * 0.299 + avg_g * 0.587 + avg_b * 0.114) > 186:
            # Our image is bright.  Use a dark color.
            text_color = [random.randint(0, 75), random.randint(0, 75), random.randint(0, 75)]
        # Have to convert text color to a hex string for Image.new.
        text_color = f"#{text_color[0]:02X}{text_color[1]:02X}{text_color[2]:02X}"
        # Make a rectangle of this color and use the rasterized text as the alpha channel, then paste it onto pil_img.
        # TODO: Add noise to the color block?
        text_color_block = Image.new("RGB", (img_pil.width, img_pil.height), color=text_color)
        text_color_block.putalpha(text_image_mask)
        img_pil.paste(text_color_block, (0, 0), text_image_mask)

        # Maybe dilate the mask?
        #text_image_mask = text_image_mask.filter(ImageFilter.MaxFilter(self.random_text_mask_dilation))

        return img_pil, text, text_image_mask, bbox
