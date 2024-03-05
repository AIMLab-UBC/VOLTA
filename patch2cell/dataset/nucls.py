import os

import numpy as np
from PIL import Image


class Nucls:

    def __init__(self):
        self.input_image_dir_name = "rgb"
        self.input_label_dir_name = "mask"
        self.input_ihc_dir_name = None
        self.skip_labels = [99]
        self.labeling_type = 'mask'
        self.first_valid_instance = 3

    @staticmethod
    def get_instance_name_from_file_name(file_name):
        instance_name, _ = os.path.splitext(os.path.split(file_name)[1])
        return instance_name

    @staticmethod
    def read_instance_mask(file_path, layer=None, fake_background_code=2):
        mask = np.array(Image.open(file_path))
        if layer is None:
            mask = mask[:, :, 1] * mask[:, :, 2]
        else:
            mask = mask[:, :, layer]
        if fake_background_code is not None:
            mask[mask == fake_background_code] = 0
        return mask

    @staticmethod
    def read_type_mask(file_path):
        mask = np.array(Image.open(file_path))
        mask = mask[:, :, 0]
        mask[mask == 253] = 0
        return mask
