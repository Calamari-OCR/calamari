from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from paiargparse import pai_dataclass, pai_meta
import cv2 as cv


@pai_dataclass
@dataclass
class ImageLoaderParams:
    channels: int = field(
        default=1,
        metadata=pai_meta(help="Number of channels to produce, by default 1=grayscale. Use 3 for colour."),
    )
    to_gray_method: str = field(
        default="cv",
        metadata=pai_meta(
            help="Method to apply to convert color to gray.",
            choices=["avg", "cv"],
        ),
    )

    def create(self) -> "ImageLoader":
        return ImageLoader(self)


class ImageLoader:
    def __init__(self, params: ImageLoaderParams):
        self.params = params

    def load_image(self, image_path: str) -> np.ndarray:
        img = load_image(image_path)
        if len(img.shape) == 2:
            img_channels = 1
        elif len(img.shape) == 3:
            img_channels = img.shape[-1]
        else:
            raise ValueError(f"Unknown image format. Must bei either WxH or WxHxC, but got {img.shape}.")

        if img_channels == self.params.channels:
            pass  # good
        elif img_channels == 2 and self.params.channels == 1:
            img = img[:, :, 0]
        elif img_channels == 3 and self.params.channels == 1:
            if self.params.to_gray_method == "avg":
                img = np.mean(img.astype("float32"), axis=-1).astype(dtype=img.dtype)
            elif self.params.to_gray_method == "cv":
                img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
            else:
                raise ValueError(f"Unsupported image conversion method {self.params.method}")
        elif img_channels == 4 and self.params.channels == 1:
            if self.params.to_gray_method == "avg":
                img = np.mean(img[:, :, :3].astype("float32"), axis=-1).astype(dtype=img.dtype)
            elif self.params.to_gray_method == "cv":
                img = cv.cvtColor(img, cv.COLOR_RGBA2GRAY)
            else:
                raise ValueError(f"Unsupported image conversion method {self.params.method}")
        elif img_channels == 1 and self.params.channels == 3:
            img = np.stack([img] * 3, axis=-1)
        else:
            raise ValueError(
                f"Unsupported image files. Trying to convert from {img_channels} channels to "
                f"{self.params.channels} channels."
            )

        return img


def load_image(image_path: str) -> np.ndarray:
    # Load an image in np.uint8 format
    with Image.open(image_path) as img:
        img = np.array(img)
        return to_uint8(img)


def to_uint8(data: np.ndarray) -> np.ndarray:
    """Read an image and returns it as uint8

    All arrays are rescaled to the range 0...255 (unsigned)
    """
    if data.dtype == np.dtype("uint8"):
        pass
    elif data.dtype == np.dtype("int8"):
        data = (data.astype("int16") + 128).astype("uint8")
    elif data.dtype == np.dtype("uint16"):
        data = (data / 256).astype("uint8")
    elif data.dtype == np.dtype("int16"):
        data = ((data / 128).astype("int16") + 128).astype("uint8")
    elif data.dtype in [np.dtype("f"), np.dtype("float32"), np.dtype("float64")]:
        data = (data * 255).astype("uint8")
    elif data.dtype == bool:
        data = data.astype("uint8") * 255
    else:
        raise Exception(f"Unknown image type: {data.dtype}")

    return data


def to_float32(data: np.ndarray) -> np.ndarray:
    """Read an image and returns it as float32

    All arrays are rescaled to the range 0...1
    """
    if data.dtype == np.dtype("uint8"):
        data = data.astype("float32") / 255
    elif data.dtype == np.dtype("int8"):
        data = (data.astype("int16") + 128).astype("float32") / 255
    elif data.dtype == np.dtype("uint16"):
        data = data.astype("float32") / 65535
    elif data.dtype == np.dtype("int16"):
        data = (data.astype("float32") + 32768) / 65535
    elif data.dtype in [np.dtype("f"), np.dtype("float32"), np.dtype("float64")]:
        data = data.astype("float32")
    elif data.dtype == bool:
        data = data.astype("float32")
    else:
        raise Exception(f"Unknown image type: {data.dtype}")

    return data
