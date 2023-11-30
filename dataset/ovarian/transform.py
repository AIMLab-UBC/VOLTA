import albumentations


class LabelTransform(object):

    def __init__(self, n_classes):
        # Label Guid -> Class 1: Neutrophil, Class 2: Epithelial, Class 3: Lymphocyte, Class 4: Plasma, Class 5: Neutrophil, Class 6: Connective tissue
        self.label_mapper = lambda x: x

    def __call__(self, x):
        return self.label_mapper(x)


def get_cell_normalization():
    return albumentations.Normalize(mean=[0.7079, 0.6259, 0.7568], std=[0.1596, 0.1681, 0.1207])


def get_patch_normalization(patch_size):
    if patch_size == 0:
        return albumentations.ToFloat()
#    if patch_size == 100:
#        return albumentations.Normalize(mean=[0.7174, 0.5355, 0.7601], std=[0.1420, 0.1539, 0.1012])
    if patch_size == 200:
        return albumentations.Normalize(mean=[0.7250, 0.6451, 0.7691], std=[0.1621, 0.1752, 0.1237])
    raise ValueError('no normalization is found for patch size of {}'.format(patch_size))
