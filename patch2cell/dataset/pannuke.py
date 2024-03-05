import scipy.io
import asyncio
import numpy as np
COUNTER = 0

class PanNuke:

    def __init__(self):
        self.input_image_dir_name = "images/images.npy"
        self.input_label_dir_name = "masks/masks.npy"
        self.input_ihc_dir_name = None
        self.skip_labels = None
        self.labeling_type = 'mask'
        self.first_valid_instance = 1
        self._lock = asyncio.Lock()

    @staticmethod
    def get_instance_name_from_file_name(file_name):
        # fixme: Not a good approach but had to do it
        global COUNTER
        COUNTER += 1
        return COUNTER

    @staticmethod
    def read_instance_mask(file_path):
        if not isinstance(file_path, np.ndarray):
            raise RuntimeError('invalid input')
        instance_labels = file_path
        instance_labels = instance_labels[:, :, :5]
        label = instance_labels[:, :, 0].copy()
        for i in range(1, instance_labels.shape[2]):
            mask = instance_labels[:, :, i] != 0
            label = label * (1-mask) + instance_labels[:, :, i] * mask
        return label.astype(int)

    @staticmethod
    def read_type_mask(file_path):
        if not isinstance(file_path, np.ndarray):
            raise RuntimeError('invalid input')
        instance_labels = file_path
        instance_labels = instance_labels[:, :, :5]
        for i in range(instance_labels.shape[2]):
            instance_labels[:, :, i] = (instance_labels[:, :, i] != 0) * (i+1)
        label = instance_labels[:, :, 0].copy()
        for i in range(1, instance_labels.shape[2]):
            mask = instance_labels[:, :, i] != 0
            label = label * (1-mask) + instance_labels[:, :, i] * mask
        return label
