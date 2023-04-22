"""
text_overlay_dataset.py
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

from .bounding_box_tools import (
    aabb_to_bbox,
    bbox_to_aabb,
    fast_conservative_theta_range,
    find_lowest_cost_assignment,
    rotate_around_point,
)


class TextOverlayDataset(Dataset):
    """
    text_overlay_dataset combines an image dataset and a text dataset (or collection of strings).
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
            prefer_larger_fonts: bool = False,

            only_english_support: bool = False,

            maximum_font_translation_percent: float = 0.0,
            maximum_font_rotation_percent: float = 0.0,
            maximum_font_quad_distortion_percent: float = 0.0,
            maximum_font_blur: float = 0.0,
            # TODO: Unused!
            maximum_font_outline_size: int = 5,
    ):
        """Generate a new TextOverlayMappingDataset from an image dataset, text, and set of fonts.

        text_overlay_dataset will attempt to dynamically generate and return quadruplets of composited text images,
        text, text raster, and axis-aligned bounding box coordinates.  (Counter-clockwise wrapping.)

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

        :param bool prefer_larger_fonts:
        If false (default) a font will be selected at random from the list of valid font sizes.  If true, will try to
        use the largest font that fits inside an image.

        :param bool only_english_support:
        If false (default), will use ImageFont.Layout.RAQM for performing font layout.  This is slightly lower but
        enables non-English sentences.  If your text is solely English (LTR, Ascii, etc), you may see slightly better
        performance by setting this to 'false'.

        :param float maximum_font_translation_percent:
        A value between 0 and 1 which indicates how much the text will move inside the image on which it's composited.
        A value of 0.0 (default) means the text will always be displayed exactly centered.  A value of 1.0 means the
        text may be moved all the way to the outer edge of the image.

        :param float maximum_font_rotation_percent:
        A value between 0 and 1 which indicates how much the text can be rotated about its center before being
        blitted.  WARNING: nonzero values, while currently supported, may cause the font to be cut off at image bounds.
        A value of 0.0 (default) means the text will always be displayed exactly horizontally.
        A value of 1.0 means the text may be rotated upside down.

        :param float maximum_font_quad_distortion_percent:
        A value between 0 and 1 which signifies the maximum allowable affine distortion.  At 0.0 (default), there is no
        affine distortion.  At 1.0, the font could theoretically be stretched to extremely odd proportions.  The text
        box will never be contracted.

        :param float maximum_font_blur:
        A value for the maximum number of pixels to be used in blurring the font.  0 means blurring is disabled.  For
        performance reasons, an optimized box blur is used rather than a gaussian blur.
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
            # MAINTAINER NOTE: Can we ensure this stays sorted without sorting every time?
            self.font_sizes = sorted(font_sizes)

        self.prefer_larger_fonts = prefer_larger_fonts

        self.layout = ImageFont.LAYOUT_RAQM
        if only_english_support:
            self.layout = ImageFont.LAYOUT_BASIC

        self.maximum_font_translation_percent = maximum_font_translation_percent
        self.maximum_font_rotation_percent = maximum_font_rotation_percent
        self.maximum_font_quad_distortion_percent = maximum_font_quad_distortion_percent
        self.maximum_font_blur = maximum_font_blur

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
            size_idx = len(self.font_sizes)-1  # Start with the biggest.
            if not self.prefer_larger_fonts:
                size_idx = random.randint(0, len(self.font_sizes)-1)  # We pick a random starting size for our font.
            font_size = self.font_sizes[size_idx]
            font_choice = random.choice(self.font_choices)

            while (text_width > width or text_height > height) and size_idx >= 0:
                if font_choice not in self.loaded_fonts:
                    # A note on this:
                    # On Windows the TrueType system keeps fonts open until the TTF object goes out of scope.  This caps
                    # the number of open fonts to 512 and can lead to OSErrors on load.  We get around this by copying
                    # the font into memory first.
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
        if self.empty_string_on_truncation:
            return False, None, None
        raise ValueError(f"Text with length {len(text)} is too large to fit in the image with size {width}x{height}.")

    def _generate_text_raster_advanced(
            self,
            text: str,
            width: int,
            height: int,
    ) -> Tuple[bool, Image.Image, numpy.ndarray]:
        """Move the rasterized text around, keeping all corners inside the width/height.
        Returns success/failure, the image, and the points on the axis-aligned bounding box as a 4x3 array.
        If max_*_percent is zero this is equivalent to the _generate_text_raster_basic method 
        and will behave similarly (except for returning aabb instead of bbox).
        """
        success, text_image_mask, text_bbox = self._generate_text_raster_basic(text, width, height)
        if not success:
            # JC: It would be nice to keep this signature in sync with the _basic version.  Does it help to enforce it
            # here, or should we return False, None, None?
            return False, None, None
        assert text_image_mask.mode == 'L'

        aabb = bbox_to_aabb(*text_bbox)
        start_aabb = aabb.copy()

        # If there are no translation or rotation operations, save the compute and return early.
        if abs(self.maximum_font_translation_percent) < 1e-6 \
                and abs(self.maximum_font_rotation_percent) < 1e-6 \
                and abs(self.maximum_font_quad_distortion_percent) < 1e-6:
            return success, text_image_mask, aabb

        if self.maximum_font_rotation_percent > 0.0:
            # TODO: We need to call the method in bbox utils to find the actual max percentage.
            angle_limits = fast_conservative_theta_range(aabb, width, height)
            if angle_limits is not None:
                rotation = random.uniform(0, (angle_limits[1]-angle_limits[0])*self.maximum_font_rotation_percent) \
                           + angle_limits[0]
                aabb_midpoint = aabb.mean(axis=0)
                aabb = rotate_around_point(aabb, rotation, aabb_midpoint[0], aabb_midpoint[1])

        # In theory it would be possible to do the translation and rotation in one matmul, but we don't know the limits
        # of the translation before we do the rotation.

        if self.maximum_font_translation_percent > 0:
            left, top, right, bottom = aabb_to_bbox(aabb)
            # Text bbox is, coincidentally, the amount we can jitter the text left/right and top/bottom.
            # JC: Variable name choice: Slop?  Play?  Tolerance?
            left_movement = left * random.random()
            right_movement = (width - right) * random.random()
            up_movement = min(top, bottom) * random.random()
            down_movement = (height - max(top, bottom)) * random.random()
            dx = (right_movement - left_movement) * self.maximum_font_translation_percent
            dy = (down_movement - up_movement) * self.maximum_font_translation_percent
            # Translate by this amount.
            aabb += numpy.asarray([dx, dy, 0])  # Lean on broadcasting.

        # Quad distortion means taking each of the points around axis-aligned bounding box and moving it at most up to
        # the nearest corner.

        # We are computing this outside of the quad_distortion code because we'll reuse it in a bit.
        # Ordering of this depends on the PIL Quad Transform:
        image_bounding_box = numpy.asarray([
            [0, 0, 1],
            [0, height, 1],
            [width, height, 1],
            [width, 0, 1],
        ])

        if self.maximum_font_quad_distortion_percent > 0.0:
            # If we were to pass the AABB to the quad transform, PIL would transform the text to fill the image.
            # If we were to pass teh image_bounding_box to quad transform, the image would be unchanged.
            # We want the text somewhere between the current position and image-filling.
            internal_external_matching = find_lowest_cost_assignment(aabb, image_bounding_box)
            # For each edge in the aabb, move it as far as the respective corner.
            for from_point_index in range(0, 4):
                to_point_index = internal_external_matching[from_point_index]
                start_x = aabb[from_point_index, 0]
                end_x = image_bounding_box[to_point_index, 0]
                start_y = aabb[from_point_index, 1]
                end_y = image_bounding_box[to_point_index, 1]
                delta_x = random.random() * (end_x - start_x) * self.maximum_font_quad_distortion_percent + start_x
                delta_y = random.random() * (end_y - start_y) * self.maximum_font_quad_distortion_percent + start_y
                aabb[from_point_index, 0] += delta_x
                aabb[from_point_index, 1] += delta_y

        # Right now AABB describes the theoretical position of the text in the image.  We want to translate from our
        # nice centered, aligned text mask to the distorted quad in the image.  This requires us to compute the
        # transform from start_aabb to aabb, then apply it to image_bounding_box
        try:
            transform = numpy.linalg.lstsq(start_aabb, aabb, rcond=None)[0]
            inv_transform = numpy.linalg.pinv(transform)
            # This one-line magic converts our 2D array of [[x, y], [x, y], ...] to a list of [x, y, x, y, ...]
            quad = (image_bounding_box @ inv_transform)[:, :2].reshape(1, -1)[0]
            #transform = ImageTransform.QuadTransform(quad)  # Perhaps outdated?
            text_image_mask = text_image_mask.transform(text_image_mask.size, Image.QUAD, quad)
        except numpy.linalg.LinAlgError:
            # Drop this exception.  Once in a while we can do an identity transform.
            # It's exceedingly rare because all rotations and translations should be linear and, thus, invertible.
            pass

        # Perform a blur.
        # At zero, the boxblur is a noop, but we can still save ourselves a call to random.
        if self.maximum_font_blur > 0:
            # Prevent a blur that's larger than the text itself.
            max_blur = min(self.maximum_font_blur, abs(text_bbox[1]-text_bbox[3]))
            text_image_mask = text_image_mask.filter(ImageFilter.BoxBlur(radius=max_blur*random.random()))

        return True, text_image_mask, aabb

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
