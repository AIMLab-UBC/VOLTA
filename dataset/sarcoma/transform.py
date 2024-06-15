import albumentations
import warnings


class LabelTransform(object):

    def __init__(self, n_classes):
        # Label Guid -> 0: B-Cell, 1: dendritic, 2: T-Cell
        if n_classes == 3:
            self.label_mapper = {0: 0, 1: 1, 2: 2}
        elif n_classes == 2:
            self.label_mapper = {0: 0, 1: 1}
        elif n_classes == 6:
            warnings.warn("number of classes is set to 6")
            self.label_mapper = {0: 0, 1: 1, 2: 2}
        else:
            raise ValueError("invalid number of classes, {}".format(n_classes))

    def __call__(self, x):
        return self.label_mapper[x]


def get_cell_normalization():
    return albumentations.Normalize(mean=[0.4776, 0.3125, 0.6792], std=[0.1731, 0.0950, 0.1052])


def get_patch_normalization(patch_size):
    if patch_size == 0:
        return albumentations.ToFloat()
    if patch_size == 100:
        raise ValueError('need to be calculated!')
    if patch_size == 200:
        return albumentations.Normalize(mean=[0.7129, 0.4117, 0.8065], std=[0.2144, 0.1293, 0.1190])
    raise ValueError('no normalization is found for patch size of {}'.format(patch_size))
