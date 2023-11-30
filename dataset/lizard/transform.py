import albumentations


class LabelTransform(object):

    def __init__(self, n_classes):
        # Label Guid -> Class 1: Neutrophil, Class 2: Epithelial, Class 3: Lymphocyte, Class 4: Plasma, Class 5: Neutrophil, Class 6: Connective tissue
        if n_classes == 6:
            self.label_mapper = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
        else:
            raise ValueError("invalid number of classes, {}".format(n_classes))

    def __call__(self, x):
        return self.label_mapper[x]


def get_cell_normalization():
    return albumentations.Normalize(mean=[0.6565, 0.4702, 0.7242], std=[0.1326, 0.1269, 0.0953])


def get_patch_normalization(patch_size):
    if patch_size == 0:
        return albumentations.ToFloat()
    if patch_size == 100:
        return albumentations.Normalize(mean=[0.7174, 0.5355, 0.7601], std=[0.1420, 0.1539, 0.1012])
    if patch_size == 200:
        return albumentations.Normalize(mean=[0.7276, 0.5494, 0.7670], std=[0.1435, 0.1612, 0.1027])
    raise ValueError('no normalization is found for patch size of {}'.format(patch_size))
