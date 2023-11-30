import albumentations
import warnings


class LabelTransform(object):

    def __init__(self, n_classes):
        if n_classes == 4:
            self.label_mapper = {0: 0, 1: 1, 2: 2, 3: 3, 99: 99}  # 99 for missing labels
        elif n_classes == 3:  # T-Cell and B-Cell -> 0, NKC -> 1, Tumor -> 2
            self.label_mapper = {0: 0, 1: 1, 2: 0, 3: 2, 99: 99}  # 99 for missing labels
        elif n_classes == 2:  # T-Cell and B-Cell and NKC -> 0, Tumor -> 2
            self.label_mapper = {0: 0, 1: 0, 2: 0, 3: 1, 99: 99}  # 99 for missing labels
        elif n_classes >= 5:
            warnings.warn(f'number of classes is set to {n_classes}')
            self.label_mapper = {0: 0, 1: 1, 2: 2, 3: 3, 99: 99}
        else:
            raise ValueError('invalid number of classes {}'.format(n_classes))


    def __call__(self, x):
        return self.label_mapper[x]


def get_cell_normalization():
    return albumentations.Normalize(mean=[0.4209, 0.2243, 0.4964], std=[0.1291, 0.1157, 0.0896])  # scale 0


def get_patch_normalization(patch_size):
    return albumentations.ToFloat()
