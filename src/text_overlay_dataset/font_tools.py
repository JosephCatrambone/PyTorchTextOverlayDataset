from string import printable
from typing import Any, Iterable, List, Mapping, Tuple

from PIL import ImageDraw, ImageFont


try:
    from fonttool import TTFont
    def font_has_all_glyphs(text: str, font: TTFont) -> bool:
        return False
except ImportError:
    def font_has_all_glyphs(text: str, font: Any) -> bool:
        return True


def measure_character_sizes(font: ImageFont, charset: str = printable) -> Mapping[str, Tuple[int, int]]:
    """Measures each charcter in the given charset and returns a Dictionary that maps a character to width/height."""
    img = Image.new()
    canvas = ImageDraw.Draw()


def measure_row_bounding_boxes(text: str, font: ImageFont):
    pass