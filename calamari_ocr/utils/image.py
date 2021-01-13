import numpy as np
from PIL import Image


def load_image(image_path: str) -> np.ndarray:
    # Load an image in np.uint8 format
    with Image.open(image_path) as img:
        img = np.array(img)
        return to_uint8(img)


def to_uint8(data: np.ndarray) -> np.ndarray:
    """
    Read an image and returns it as uint8
    The optional page number allows images from files containing multiple
    images to be addressed.  All arrays are rescaled to
    the range 0...255 (unsigned)
    """
    if data.dtype == np.dtype('uint8'):
        data = data
    elif data.dtype == np.dtype('int8'):
        data = (data.astype('int16') + 128).astype('uint8')
    elif data.dtype == np.dtype('uint16'):
        data = (data / 256).astype('uint8')
    elif data.dtype == np.dtype('int16'):
        data = ((data / 128).astype('int16') + 128).astype('uint8')
    elif data.dtype in [np.dtype('f'), np.dtype('float32'), np.dtype('float64')]:
        data = (data * 255).astype('uint8')
    elif data.dtype == bool:
        data = data.astype('uint8') * 255
    else:
        raise Exception("unknown image type: {}".format(data.dtype))

    return data
