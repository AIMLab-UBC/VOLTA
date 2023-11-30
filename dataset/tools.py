import collections
import random

import numpy as np
import torch
from torch._six import string_classes
from torch.utils.data._utils.collate import np_str_obj_array_pattern, default_collate_err_msg_format


def collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)

    if elem is None:
        return None

    # This part is directly taken from the default_collate_fn from Pytorch
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return collate_fn([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate_fn(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [collate_fn(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def equalize_subtypes(dataset: torch.utils.data.dataset.Dataset):
    targets = dataset.targets
    values, counts = np.unique(targets, return_counts=True)
    count = min(counts)
    indices = []
    for v, c in zip(values, counts):
        indices.extend(random.sample([j for j, t in enumerate(targets) if t == v], count))
    return Subset(dataset, indices)


def stratified_ratio_subset(dataset: torch.utils.data.dataset.Dataset, ratio: float):
    targets = dataset.targets
    values, counts = np.unique(targets, return_counts=True)
    indices = []
    for v, c in zip(values, counts):
        if c == 0:
            continue
        count = int(np.ceil(ratio * c))
        indices.extend(random.sample([j for j, t in enumerate(targets) if t == v], count))
    return Subset(dataset, indices)


class Subset(torch.utils.data.Subset):

    @property
    def targets(self):
        return [t for i, t in enumerate(self.dataset.targets) if i in self.indices]
