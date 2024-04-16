import os

import numpy as np
from PIL import Image


def stich_images(
    images: list[Image.Image | str],
    width: int = None,
    height: int = None,
    row: int = None,
    col: int = None
)-> Image.Image:
    """Stich multi images to one

    Args:
        images (Image.Image | str): image files
        width (int): width of each image. Using first width of first image if None
        height (int): height of each image. Using first height of first image if None
        row (int): row count. default 1 if None
        col (int): col count. default len(image_grid) if None
    """
    row = 1 if not row else row
    col = len(images) if not col else col

    image_grid = np.array(images).reshape((row, col))

    output: Image.Image = None
    if width and height:
        output = Image.new('RGB', (width * col, height * row))

    row_index = 0
    col_index = 0
    for img_row in image_grid:
        for img in img_row:
            if not isinstance(img, (str, Image.Image)):
                raise ValueError("Item is not a Pillow image or path")
            if isinstance(img, str):
                if not os.path.exists(img):
                    raise ValueError(f"Cannot open file. {img}")
                img = Image.open(img)
            # If not specify width and height, use size of the first image
            if not output:
                width, height = img.size
                output = Image.new('RGB', (width * col, height * row))

            output.paste(img, (0 + width * col_index, 0 + height * row_index))
            col_index += 1
        row_index += 1

    return output
