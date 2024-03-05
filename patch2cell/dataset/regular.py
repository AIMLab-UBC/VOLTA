import os

import numpy as np
from PIL import Image


class Regular:

    def __init__(self):
        self.input_image_dir_name = "Images"
        self.input_label_dir_name = "Labels"
        self.input_ihc_dir_name = "IHC"
        self.skip_labels = None
        self.labeling_type = 'ihc'
        self.first_valid_instance = 1

    @staticmethod
    def get_instance_name_from_file_name(file_name):
        instance_name, _ = os.path.splitext(os.path.split(file_name)[1])
        return instance_name

    @staticmethod
    def read_instance_mask(file_path):
        mask = np.load(file_path)
        return mask[:, :, 0].astype(int)

    @staticmethod
    def read_type_mask(file_path):
        mask = np.load(file_path)
        return mask[:, :, 1].astype(int)
