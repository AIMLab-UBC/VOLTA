import scipy.io
import numpy as np


class Lizard:

    def __init__(self):
        self.input_image_dir_name = "Images"
        self.input_label_dir_name = "Labels"
        self.input_tissue_dir_name = None
        self.input_ihc_dir_name = None
        self.skip_labels = None
        self.labeling_type = 'mask'
        self.first_valid_instance = 1

    @staticmethod
    def get_instance_name_from_file_name(file_name):
        instance_name = file_name.split('.')[-2].split('\\')[-1]
        return instance_name

    @staticmethod
    def read_instance_mask(file_path):
        label = scipy.io.loadmat(file_path)
        return label['inst_map'].astype(int)

    @staticmethod
    def read_type_mask(file_path):
        label = scipy.io.loadmat(file_path)
        ids = np.squeeze(label['id']).tolist()
        inst_type = label['inst_map'].copy()
        for id in ids:
            index = ids.index(id)
            inst_type[inst_type == id] = label['class'][index]
        return inst_type
