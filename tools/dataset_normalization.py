import argparse
import os

import albumentations
from albumentations.pytorch import ToTensorV2
import cv2
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.images = []
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if filename.endswith('.png') or filename.endswith('.jpg'):
                    self.images.append(os.path.join(dirpath, filename))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.images[idx]).convert('RGB'))
        if self.transform:
            image = self.transform(image=image)['image']
        return image


def normalize_dataset(dataset, description):
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=10, num_workers=10, shuffle=False)

    mean = 0.
    std = 0.
    nb_samples = 0.
    for img in tqdm(loader, total=len(loader), desc=description):
        batch_samples = img.size(0)
        assert img.size(1) == 3
        img = img.view(batch_samples, 3, -1)
        mean += img.mean(2).sum(0)
        std += img.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean, std


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dataset normalization metrics')
    parser.add_argument('--root', default='.', type=str, help='path to dataset')
    parser.add_argument('--cell_size', default=32, type=int, help='desired size of cell')
    args = parser.parse_args()


    # normalize cell dataset 
    cell_dataset = ImageDataset(
        root=os.path.join(args.root, 'Images'), 
        transform=albumentations.Compose([albumentations.ToFloat(), albumentations.Resize(args.cell_size, args.cell_size, interpolation=cv2.INTER_CUBIC), ToTensorV2(transpose_mask=True)])
        )

    cell_mean, cell_std = normalize_dataset(cell_dataset, 'cell normalization')

    # normalize patch dataset
    patch_dataset = ImageDataset(
        root=os.path.join(args.root, 'Patches'), 
        transform=albumentations.Compose([albumentations.ToFloat(), ToTensorV2(transpose_mask=True)]),
    )

    patch_mean, patch_std = normalize_dataset(patch_dataset, 'patch normalization')


    print(f'cell mean: {cell_mean}, cell std: {cell_std}')
    print(f'patch mean: {patch_mean}, patch std: {patch_std}')
