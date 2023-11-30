import albumentations


class LabelTransform(object):

    def __init__(self, n_classes):
        # Labeling guide -> 0: Neoplastic cells, 1: Inflammatory, 2: Connective/Soft tissue cells, 3: Dead Cells, 4: Epithelial
        if n_classes == 5:
            self.label_mapper = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
        elif n_classes == 4:
            self.label_mapper = {1: 0, 2: 1, 3: 2, 5: 3}
        else:
            raise ValueError("invalid number of classes, {}".format(n_classes))

    def __call__(self, x):
        return self.label_mapper[x]


def get_cell_normalization():
    return albumentations.Normalize(mean=[0.6617, 0.5159, 0.7612], std=[0.1489, 0.1516, 0.1044]) # colon section
    return albumentations.Normalize(mean=[0.5991, 0.4382, 0.6178], std=[0.1627, 0.1505, 0.1196])  # whole section


def get_patch_normalization(patch_size):
    if patch_size == 0:
        return albumentations.ToFloat()
    if patch_size == 100: 
        return albumentations.Normalize(mean=[0.7364, 0.5942, 0.7999], std=[0.1564, 0.1703, 0.1071]) # colon section
#        return albumentations.Normalize(mean=[0.6867, 0.5198, 0.6752], std=[0.1649, 0.1693, 0.1286]) # whole section
    if patch_size == 200:
        return albumentations.Normalize(mean=[0.7510, 0.6121, 0.8079], std=[0.1594, 0.1790, 0.1095]) # colon section
    raise ValueError('no normalization is found for patch size of {}'.format(patch_size))
