import os

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset

from dataset.consep.validation import _erosion_validation_func
from misc.files import list_files

MORPHOLOGICAL_FEATURE_SIZE = 38  # check this


def image_to_patch_name_convertor(img_name):
    patch_name = img_name.replace('Images', 'Patches')
    patch_name, ext = os.path.splitext(patch_name)
    patch_name = '_'.join(patch_name.split('_')[:-1])  # remove the instance number
    return patch_name + ext


def image_to_segmentation_name_convertor(img_name):
    patch_name = img_name.replace('Images', 'Segmentation')
    patch_name, ext = os.path.splitext(patch_name)
    patch_name = '_'.join(patch_name.split('_')[:-1])  # remove the instance number
    return patch_name + '.npy'


def _get_file_name(file):
    return os.path.splitext(os.path.split(file)[1])[0]


class OvarianDataset(Dataset):
    """CoNSep dataset."""

    def __init__(self, root_dir, transform=None, target_transform=None, patch_transform=None, patch_size=None,
                 mask_ratio=None, cache_patch=True, hovernet_enable=False, return_file_name=False, dataset_size=None,
                 valid_labels=None, shared_dictionaries=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(OvarianDataset, self).__init__()
        # todo: make this faster by taking a pickle file
        images_path = os.path.join(root_dir, 'Images')
        labels_path = os.path.join(root_dir, 'Labels')
        location_path = os.path.join(root_dir, 'Locations')
        map_path = os.path.join(root_dir, 'Map')
        embedding_path = os.path.join(root_dir, 'Embedding')

        self.image_files = sorted(list_files(images_path))
        self.label_files = sorted(list_files(labels_path))
        self.location_files = sorted(list_files(location_path))
        self.map_files = sorted(list_files(map_path))
        # take the patch and segmentation names since they're from patch info
        self.patch_files = [image_to_patch_name_convertor(x) for x in self.image_files]
        self.segmentation_files = [image_to_segmentation_name_convertor(x) for x in self.image_files]
        self.embedding_files = None
        if hovernet_enable:
            self.embedding_files = sorted(list_files(embedding_path))

        if valid_labels is not None:
            self.filter_valid_labels(valid_labels)

        if dataset_size is not None:
            self.image_files = self.image_files[:dataset_size]
            self.label_files = self.label_files[:dataset_size]
            self.location_files = self.location_files[:dataset_size]
            self.map_files = self.map_files[:dataset_size]
            self.patch_files = self.patch_files[:dataset_size]
            self.segmentation_files = self.segmentation_files[:dataset_size]
            if self.embedding_files is not None:
                self.embedding_files = self.embedding_files[:dataset_size]

        # transforms
        self.transform = transform
        self.target_transform = target_transform
        self.patch_transform = patch_transform

        # extra variable
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.return_file_name = return_file_name
        if cache_patch:
            assert shared_dictionaries is not None
        self.cache_patch = shared_dictionaries['cache_patch'] if cache_patch else None
        self.cache_segmentation = shared_dictionaries['cache_segmentation'] if cache_patch else None
        self.cache_morphological = shared_dictionaries['cache_morphological'] if cache_patch else None

    @property
    def data(self):
        return [self._read_items(i)[0] for i in range(self.__len__())]

    @property
    def targets(self):
        return [self._read_items(i)[1] for i in range(self.__len__())]

    @property
    def patches(self):
        return [self._read_items(i)[2] for i in range(self.__len__())]

    @property
    def segmentations(self):
        return [self._read_items(i)[6] for i in range(self.__len__())]

    @property
    def extra_features(self):
        return [self._read_items(i)[7] for i in range(self.__len__())]

    def filter_valid_labels(self, valid_labels):
        idx = [i for i, l in enumerate(self.label_files) if int(open(l, 'r').readline()) in valid_labels]
        self.image_files = [self.image_files[j] for j in idx]
        self.label_files = [self.label_files[j] for j in idx]
        self.location_files = [self.location_files[j] for j in idx]
        self.map_files = [self.map_files[j] for j in idx]
        self.patch_files = [self.patch_files[j] for j in idx]
        self.segmentation_files = [self.segmentation_files[j] for j in idx]
        if self.embedding_files is not None:
            self.embedding_files = [self.embedding_files[j] for j in idx]

    @staticmethod
    def patch_path_to_name_convertor(patch_path):
        patch_name, _ = os.path.splitext(patch_path)
        _, patch_name = os.path.split(patch_name)
        patch_name = patch_name.split('_')[0]
        assert patch_name.isdigit()
        return int(patch_name)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        return self._read_items(idx)

    def _read_items(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        assert _get_file_name(self.image_files[idx]) == _get_file_name(self.label_files[idx])
        assert _get_file_name(self.image_files[idx]) == _get_file_name(self.location_files[idx])
        assert _get_file_name(self.image_files[idx]) == _get_file_name(self.map_files[idx])
        if self.embedding_files is not None:
            assert _get_file_name(self.image_files[idx]) == _get_file_name(self.embedding_files[idx])

        image = Image.open(self.image_files[idx]).convert('RGB')
        label = int(open(self.label_files[idx], 'r').readline())
        location = [int(float(x.strip())) for x in open(self.location_files[idx], 'r').readline().split(',')]
        slide_id = self.patch_path_to_name_convertor(self.patch_files[idx])

        # read patch
        if not self.patch_size:
            patch = None
        else:
            if self.cache_patch is not None and slide_id in self.cache_patch:
                patch = self.cache_patch[slide_id].copy()
            else:
                patch = Image.open(self.patch_files[idx]).convert('RGB')
                if self.cache_patch is not None:
                    self.cache_patch[slide_id] = patch.copy()

        # Read the segmentation
        if patch is None:
            segmentation = None
        else:
            if self.cache_segmentation is not None and slide_id in self.cache_segmentation:
                segmentation = self.cache_segmentation[slide_id].copy()
            else:
                inst_map = np.load(self.segmentation_files[idx])
                segmentation = np.zeros_like(inst_map, dtype=float)
                for cell in ndimage.measurements.find_objects(inst_map):
                    if cell is None:
                        continue
                    y_min, y_max = cell[0].start, cell[0].stop
                    x_min, x_max = cell[1].start, cell[1].stop
                    segmentation[y_min:y_max, x_min:x_max] = 1
                if self.cache_segmentation is not None:
                    self.cache_segmentation[slide_id] = segmentation.copy()

        # todo: make this a function
        # resize patch
        patch_left, patch_top = 0, 0
        cell_center_x, cell_center_y = 0, 0
        if self.patch_size:
            patch_left = int(max(0, location[0] - self.patch_size / 2))
            patch_top = int(max(0, location[1] - self.patch_size / 2))
            right, bottom = int(min(patch.width, patch_left + self.patch_size)), int(
                min(patch.height, patch_top + self.patch_size))
            if right - patch_left < self.patch_size:  # check for patch size
                patch_left = right - self.patch_size
            if bottom - patch_top < self.patch_size:  # check for patch size
                patch_top = bottom - self.patch_size
            patch = patch.crop((patch_left, patch_top, right, bottom))
            segmentation = segmentation[patch_top:bottom, patch_left:right]
            cell_center_x, cell_center_y = location[0] - patch_left, location[1] - patch_top

        # todo: make this a function
        if patch is None:
            mask = None
        else:
            mask = torch.zeros((patch.height, patch.width), dtype=torch.bool, requires_grad=False)
            if self.mask_ratio:
                mask_height, mask_width = int(image.height * self.mask_ratio), int(image.width * self.mask_ratio)
                mask_top, mask_left = int(cell_center_y - mask_height / 2), int(cell_center_x - mask_width / 2)

                # set limits
                mask_top = max(mask_top, 0)
                mask_left = max(mask_left, 0)
                mask_bottom = min(mask_top + mask_height, patch.height)
                mask_right = min(mask_left + mask_width, patch.width)

                mask[mask_top:mask_bottom, mask_left:mask_right] = True

        # Calculate morphological features
#        if self.cache_morphological is not None and idx in self.cache_morphological:
#            morphological_feature = self.cache_morphological[idx].copy()
#        else:
#            map_segmentation = np.load(self.map_files[idx])
#            if not _erosion_validation_func(map_segmentation):
#                morphological_feature = np.zeros(MORPHOLOGICAL_FEATURE_SIZE)
#            else:
#                morphological_feature = extract_morphological_features(map_segmentation)
#            if self.cache_morphological is not None:
#                self.cache_morphological[idx] = morphological_feature.copy()
#        morphological_feature = morphological_feature.astype('float32')
        morphological_feature = np.zeros(MORPHOLOGICAL_FEATURE_SIZE)

        # Calculate hovernet features
        if self.embedding_files is not None:
            hovernet_feature = np.load(self.embedding_files[idx])

        # apply transformations
        if self.transform is not None:
            image = self.transform(image=np.array(image))
            if isinstance(image, dict):
                image = image['image']
            elif isinstance(image, tuple):
                image = image[0]
        if self.target_transform is not None:
            label = self.target_transform(label)
        if self.patch_transform is not None and patch is not None:
            # Note: convert mask to float and numpy because of the albumentation
            masks = [mask.float().numpy(), segmentation]
            patch = self.patch_transform(image=np.array(patch), masks=masks)
            if isinstance(patch, dict):  # return output of the albumentation transform
                mask, segmentation = patch['masks']
                mask, segmentation = mask.astype(bool), segmentation.astype(bool)
                patch = patch['image']
            elif isinstance(patch, tuple):  # for when two crop augmentation is used
                patch, mask, segmentation = patch

        extra_feature = morphological_feature
        if self.embedding_files is not None:
            extra_feature = hovernet_feature

        if self.return_file_name:
            return image, os.path.splitext(os.path.split(self.image_files[idx])[1])[0]

        return image, label, patch, slide_id, torch.LongTensor(
            [patch_left, patch_top]), mask, segmentation, extra_feature
