import albumentations


class LabelTransform(object):

    def __init__(self, n_classes):
        if n_classes == 7:
            self.label_mapper = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}
        elif n_classes == 4:
            self.label_mapper = {1: 0, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3}
        else:
            raise ValueError('invalid number of classes {}'.format(n_classes))

    def __call__(self, x):
        return self.label_mapper[x]


def get_cell_normalization():
    return albumentations.Normalize(mean=[0.7367, 0.5941, 0.8084], std=[0.1548, 0.1683, 0.1078])


def get_patch_normalization(patch_size):
    if patch_size == 0:
        return albumentations.ToFloat()
    if patch_size == 100:
        return albumentations.Normalize(mean=[0.7737, 0.6376, 0.8278], std=[0.1569, 0.1789, 0.1094])
    if patch_size == 200:
        return albumentations.Normalize(mean=[0.7933, 0.6646, 0.8378], std=[0.1592, 0.1879, 0.1119])
    raise ValueError('no normalization is found for patch size of {}'.format(patch_size))