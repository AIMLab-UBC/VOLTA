import argparse
import csv
import operator
import os
from functools import partial
from multiprocessing import Manager
from tqdm import tqdm
import numpy as np
from PIL import Image
from scipy import ndimage, stats
from tqdm.contrib.concurrent import process_map  # or thread_map
import random

from dataset.nucls import Nucls
from dataset.regular import Regular
from dataset.pannuke import PanNuke
from dataset.lizard import Lizard
from dataset.hovernet import Hovernet
from utils.files import list_files, read_tif_image
from multiprocessing import Pool


def convert_patch_to_cells(in_img_path: str or np.ndarray, in_mask_path: str or np.ndarray, ihc_path: str, out_img_path: str, out_label_path: str,
                           out_loc_path: str, out_map_path: str, overlay_path: str, out_patch_path: str,
                           out_segmentation_path: str, scale_factor: int, dataset, translator: dict = None, 
                           count_sanity_check: bool = False, n_sample=None):
    # Check for name consistency
    if isinstance(in_img_path, str) and isinstance(in_mask_path, str):
        assert os.path.splitext(os.path.split(in_img_path)[1])[0] == os.path.splitext(os.path.split(in_mask_path)[1])[0]
    if ihc_path is not None and isinstance(in_img_path, str) and isinstance(ihc_path, str):
        assert os.path.splitext(os.path.split(in_img_path)[1])[0] == os.path.splitext(os.path.split(ihc_path)[1])[0]

    # get instance name
    instance_name = dataset.get_instance_name_from_file_name(in_img_path)
    print(f'instance_name: {instance_name}')
    if translator is not None:
        if instance_name not in translator:
            last_index = max([x for x in translator.values()]) if len(translator) != 0 else 1
            translator[instance_name] = last_index + 1
        instance_name = str(translator[instance_name])

    # open image
    if isinstance(in_img_path, str):
        img = Image.open(in_img_path)
    elif isinstance(in_img_path, np.ndarray):
        img = in_img_path
        img = Image.fromarray(img.astype('uint8'))
    else:
        raise TypeError(f'invalid input type {type(in_img_path)}')

    # save patch data
    img.save(os.path.join(out_patch_path, instance_name + '.png'))

    # open instance mask
    instance_mask = dataset.read_instance_mask(in_mask_path)

    # save instance segmentation map
    if img.width != instance_mask.shape[1]:
        if img.width > instance_mask.shape[1]:
            instance_mask = np.pad(instance_mask, mode='edge',
                                   pad_width=((0, 0), (0, img.width - instance_mask.shape[1])))
        else:
            instance_mask = instance_mask[:, :img.width]
    if img.height != instance_mask.shape[0]:
        if img.height > instance_mask.shape[0]:
            instance_mask = np.pad(instance_mask, mode='edge',
                                   pad_width=((0, img.height - instance_mask.shape[0]), (0, 0)))
        else:
            instance_mask = instance_mask[:img.height, :]
    np.save(os.path.join(out_segmentation_path, instance_name + '.npy'), instance_mask)

    # open type mask
    type_mask = None
    if dataset.labeling_type == 'mask':
        type_mask = dataset.read_type_mask(in_mask_path)

        if img.width != type_mask.shape[1]:
            if img.width > type_mask.shape[1]:
                type_mask = np.pad(type_mask, mode='edge', pad_width=((0, 0), (0, img.width - type_mask.shape[1])))
            else:
                type_mask = type_mask[:, :img.width]
        if img.height != type_mask.shape[0]:
            if img.height > type_mask.shape[0]:
                type_mask = np.pad(type_mask, mode='edge', pad_width=((0, img.height - type_mask.shape[0]), (0, 0)))
            else:
                type_mask = type_mask[:img.height, :]


    # get the cell locations in the label map
    cell_locations = ndimage.measurements.find_objects(instance_mask.astype(int))
    if n_sample is not None:
        idex = list(range(len(cell_locations)))
        random.shuffle(idex)
        idex = idex[:n_sample]
        cell_locations = [cell_locations[i] for i in idex]

    # cell name generator
    cell_name_generator = lambda number: instance_name + '_' + str(number)

    saved_cell = 0
    cell_count = 0
    # go through each cell
    for i, cell in enumerate(cell_locations):

        # check for none
        if cell is None:
            print(f'cell is empty, idx: {i}')
            continue
        
        
        # get y location
        y_min, y_max = cell[0].start, cell[0].stop

        # get x location
        x_min, x_max = cell[1].start, cell[1].stop

        # find cell number with getting mode in that region
        region = instance_mask[y_min:y_max, x_min:x_max].reshape(-1, 1).astype(int)
#        cell_number = stats.mode(region[np.nonzero(region)]).mode[0]  ## solve a bug
        cell_number = i + 1

        if cell_number < dataset.first_valid_instance:
            print("did not expect cell number lower than {}, cell_number {}".format(dataset.first_valid_instance, cell_number))
            continue

        cell_count += 1
        if count_sanity_check:
            continue

        # get cell center
        cell_center = '{}, {}'.format((x_min + x_max) / 2, (y_min + y_max) / 2)

        # get cell label
        if type_mask is not None:
            region = type_mask[y_min:y_max, x_min:x_max].reshape(-1, 1)
            cell_label = stats.mode(region[np.nonzero(region)]).mode[0]
        else:
            cell_label = 99

        # ignore unlabeled cells
        if dataset.skip_labels is not None and cell_label in dataset.skip_labels:
            print('ignore unlabeled')
            continue

        # get width and height of the box
        height = y_max - y_min
        width = x_max - x_min
        # make the crop square
        height = width = max(height, width)

        scale = scale_factor / 2

        height_offset = scale * height
        width_offset = scale * width

        # crop cell image
        y_min_map = int(max(y_min - height_offset, 0))
        y_max_map = int(min(y_max + height_offset, img.height))
        x_min_map = int(max(x_min - width_offset, 0))
        x_max_map = int(min(x_max + width_offset, img.width))

        cell_img = img.crop((x_min_map, y_min_map, x_max_map, y_max_map))

        # save cell image
        cell_img.save(os.path.join(out_img_path, cell_name_generator(cell_number) + '.png'))
        saved_cell += 1

        # save cell label
        with open(os.path.join(out_label_path, cell_name_generator(cell_number)) + '.txt', 'w') as f:
            f.write('%d' % cell_label)

        # save cell location
        with open(os.path.join(out_loc_path, cell_name_generator(cell_number)) + '.txt', 'w') as f:
            f.write('%s' % cell_center)

        # save cell segmentation map
        if type_mask is not None:
            label_map = type_mask[y_min_map:y_max_map, x_min_map:x_max_map]
        else:
            label_map = cell_label * np.ones((y_max_map - y_min_map, x_max_map - x_min_map))
        cell_mask = instance_mask[y_min_map:y_max_map, x_min_map:x_max_map].astype(int)
        cell_mask = cell_mask == cell_number
        np.save(os.path.join(out_map_path, cell_name_generator(cell_number)) + '.npy', label_map * cell_mask)
    print(f'cell count: {cell_count}, saved_cell: {saved_cell}, instance count: {len(np.unique(instance_mask))-1}')


def convert_dataset(source_path, output_path, dataset, scale_factor, workers, count_sanity_check=False, n_sample=None):
    # create directory if not exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Make the source paths
    image_path = os.path.join(source_path, dataset.input_image_dir_name)
    label_path = os.path.join(source_path, dataset.input_label_dir_name)
    if dataset.input_ihc_dir_name is not None:
        ihc_path = os.path.join(source_path, dataset.input_ihc_dir_name)

    # get image files
    if os.path.isdir(image_path):
        image_list = sorted(list_files(image_path))
    else:
        image_list = np.load(image_path)

    # get label files
    if os.path.isdir(label_path):
        label_list = sorted(list_files(label_path))
    else:
        label_list = np.load(label_path)

    # get ihc files
    ihc_list = [None] * len(label_list)
    if dataset.input_ihc_dir_name is not None:
        ihc_list = sorted(list_files(ihc_path))

    # output paths
    output_img_path = os.path.join(output_path, "Images")
    output_label_path = os.path.join(output_path, "Labels")
    output_location_path = os.path.join(output_path, "Locations")
    output_map_path = os.path.join(output_path, "Map")
    output_overlay_path = os.path.join(output_path, "Overlay")
    output_patch_path = os.path.join(output_path, "Patches")
    output_segmentation_path = os.path.join(output_path, "Segmentation")

    # name_mapper dictionary
    manager = Manager()
    name_mapper = manager.dict()

    # create directories
    if not os.path.exists(output_img_path):
        os.makedirs(output_img_path)

    if not os.path.exists(output_label_path):
        os.makedirs(output_label_path)

    if not os.path.exists(output_location_path):
        os.makedirs(output_location_path)

    if not os.path.exists(output_map_path):
        os.makedirs(output_map_path)

    if not os.path.exists(output_overlay_path):
        os.makedirs(output_overlay_path)

    if not os.path.exists(output_patch_path):
        os.makedirs(output_patch_path)

    if not os.path.exists(output_segmentation_path):
        os.makedirs(output_segmentation_path)

    for in_image, in_label, in_ihc in tqdm(zip(image_list, label_list, ihc_list)):
        convert_patch_to_cells(in_image, in_label, in_ihc,
                        out_img_path=output_img_path,
                        out_label_path=output_label_path,
                        out_loc_path=output_location_path,
                        out_map_path=output_map_path,
                        overlay_path=output_overlay_path,
                        out_patch_path=output_patch_path,
                        out_segmentation_path=output_segmentation_path,
                        scale_factor=scale_factor,
                        dataset=dataset,
                        translator=name_mapper,
                        count_sanity_check=count_sanity_check, n_sample=n_sample)

    with open(os.path.join(output_path, 'name_mapper.csv'), 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in name_mapper.items():
            writer.writerow([key, value])


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def main():
    parser = argparse.ArgumentParser(description='NuCLS Convertor Arguments')
    parser.add_argument('--workers', type=int, default=1, help='number of workers to be used')
    parser.add_argument('--source', type=str, required=True, help='path to the root of source data')
    parser.add_argument('--destination', type=str, required=True, help='path to the root of destination data')
    parser.add_argument('--scale', type=int, default=1, help='scale factor to be used for cell separation')
    parser.add_argument('--dataset', type=str, default='nucls', help='dataset type: nucls, pannuke, lizard')
    parser.add_argument('--count_sanity_check', type=bool_flag, default=False, help='sanity check for cell count')
    parser.add_argument('--n_sample', type=int, default=None, help='number of cells to be sampled form each patch')
    args = parser.parse_args()

    # load the dataset template
    if args.dataset.lower() == 'nucls':
        dataset = Nucls()
    elif args.dataset.lower() == 'regular':
        dataset = Regular()
    elif args.dataset.lower() == 'pannuke':
        dataset = PanNuke()
    elif args.dataset.lower() == 'lizard':
        dataset = Lizard()
    elif args.dataset.lower() == 'hovernet':
        dataset = Hovernet()
    else:
        raise ValueError('invalid dataset type')

    convert_dataset(args.source, args.destination, dataset, args.scale, args.workers, count_sanity_check=args.count_sanity_check, n_sample=args.n_sample)


if __name__ == '__main__':
    main()
