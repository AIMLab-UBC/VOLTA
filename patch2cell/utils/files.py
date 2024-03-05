import os

import numpy as np
import tifffile
from PIL import Image


def list_files(dir, sort=True, ext=None):
    files = [os.path.join(dir, f) for f in os.listdir(dir)]
    if ext is not None:
        files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(ext)]
    if sort:
        files = sorted(files)
    return files


def get_tif_image_layer_count(file_name):
    """
    :param file_name:
    :return:
    """
    with tifffile.TiffFile(file_name) as tif:
        return len(tif.pages)


def read_tif_image(file_name, factor=None, read_desc=True, threshold=True):
    """
    Read TIF images
    :param file_name: str indicating the path to the files
    :param factor: scaling factor
    :return: list of read images
    """
    layers = {}
    with tifffile.TiffFile(file_name) as tif:
        for i, page in enumerate(tif.pages):
            img = page.asarray()
            if read_desc and 'ImageDescription' in page.tags:
                description = page.tags['ImageDescription'].value
                description = description.split(' ')[0]
            else:
                description = i
            if threshold:
                img[img > 0] = 1
            if factor is not None:
                img = Image.fromarray(np.clip(img, 0, 255).astype('uint8'))
                img = img.resize((img.width * factor, img.height * factor))
                img = np.array(img)
            layers[description] = img
    return layers
