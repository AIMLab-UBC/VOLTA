#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import datetime
import math
import os
import pprint
import random
import shutil
import time
import warnings
from multiprocessing import Manager

import albumentations
import cv2
import git
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
from albumentations.pytorch import ToTensorV2

import datetime

from misc.weight import weight_init

try:
    import apex
    from apex import amp
except:
    pass

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

import backbones as models
import moco
import moco.loader
import moco.mocov3.builder
from dataset.consep import transform as consep_dataset
from dataset.consep.dataset import CoNSePDataset
from dataset.oracle import transform as oracle_dataset
from dataset.oracle.dataset import OracleDataset
from dataset.nucls import transform as nucls_dataset
from dataset.nucls.dataset import NuCLSDataset
from dataset.pannuke import transform as pannuke_dataset
from dataset.pannuke.dataset import PanNukeDataset
from dataset.lizard import transform as lizard_dataset
from dataset.lizard.dataset import LizardDataset
from dataset.sarcoma import transform as sarcoma_dataset
from dataset.sarcoma.dataset import SarcomaDataset
from dataset.ovarian import transform as ovarian_dataset
from dataset.ovarian.dataset import OvarianDataset
from dataset.tools import collate_fn
from loss.softmax import MaskedCrossEntropyLoss
from misc.loss import focal_loss, LabelSmoothing
from misc.metrics import AverageMeter, ProgressMeter, clustering_metrics
from misc.optimizer import build_optimizer

try:
    import tensorflow as tf
    import tensorboard as tb

    # quick fix for tensorboard embedding issue
    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
except ModuleNotFoundError as e:
    pass

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', default='consep', type=str, metavar='DT',
                    help='dataset type including consep, cifar, imagenet')
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument('--image-size', default=32, type=int, metavar='IS',
                    help='size to rescale the input')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--optim', '--optimizer', default='sgd', type=str,
                    metavar='O', help='optimizer. choices: sgd, adam, adamw, lars, lamb', dest='optim')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--betas', default=(0.9, 0.99), type=tuple,
                    metavar='B', help='initial betas for optimization', dest='betas')
parser.add_argument('--warmup-epoch', default=10, type=int,
                    metavar='WE', help='number of epochs for warmup', dest='warmup_epoch')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x) after the warmup steps')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to pretrained checkpoint (default: none) used for finetune')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--gpus', type=str, default="0")
parser.add_argument('--save-dir', type=str, default='checkpoints', help='where to save models')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--n_classes', default=4, type=int, help='number of classes to be used (default: 4)')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
# parser.add_argument('--moco-k', default=65536, type=int,
#                     help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--mlp', nargs='+', type=int, default=[128, 128],
                    help='mlp head layer sizes')
parser.add_argument('--prediction-head', default=32, type=int,
                    help='size of the prediction head mlp')
parser.add_argument('--mlp-embedding', action='store_true',
                    help='add mlp head as extra embedding layer')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--vertical-flip', action='store_true',
                    help='add vertical flip to augmentations')
parser.add_argument('--rotation', action='store_true',
                    help='add rotation by +/- 45 degrees to augmentations')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--euclidean-nn', action='store_true',
                    help='use euclidean metric for validation nearest neighbor')
parser.add_argument('--validation-interval', default=10, type=int,
                    help='validation interval in terms of epoch. Set to 0 to deactivate')
parser.add_argument('--apex', action='store_true',
                    help='use apex')
parser.add_argument('--optim-level', default='O1', type=str,
                    help='apex optimization level (default: O1)')
parser.add_argument('--job-id', default=datetime.datetime.now().timestamp(), type=int, help='slurm job id')
parser.add_argument('--spectral-norm', action='store_true', help='spectral normalization')
parser.add_argument('--focal-gamma', default=0, type=int, help='focal loss gamma - 0 disables focal loss (default: 0')
parser.add_argument('--smoothing-alpha', default=0., type=float,
                    help='alpha for label smoothing - focal loss is prior to this(default: 0')
parser.add_argument('--queue-size', default=0, type=int,
                    help='negative sample queue size (default: 0)')
parser.add_argument('--co2-weight', default=0., type=float,
                    help='weight used for consistency loss (default: 0)')
parser.add_argument('--co2-t', default=0., type=float,
                    help='tau used for consistency loss (default: 0)')
parser.add_argument('--embedding-size', default=0, type=int,
                    help='embedding size used for quantization of moco embeddings (default: 0)')
parser.add_argument('--commitment-cost', default=0., type=float,
                    help='commitment cost factor used for moco quantization (default: 0)')
parser.add_argument('--moco-type', default='v3', type=str,
                    help='moco type from v3, vq, env (default: v3)')
parser.add_argument('--patch-size', default=None, type=int,
                    help='patch size used for env moco (default: 0)')
parser.add_argument('--env-arch', metavar='ENVARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('--shared-encoder', action='store_true', help='use shared encoder for env model')
parser.add_argument('--mask-ratio', type=float, default=0.0, help='ratio of the mask')
parser.add_argument('--env-coef', type=float, default=0.0, help='loss coefficient of env')
parser.add_argument('--mask-cells', action='store_true', help='mask all cells in the patch')
parser.add_argument('--maskedenv-cat', action='store_true',
                    help='concatenate the env and cell embeddings for maskedenv')
parser.add_argument('--morphological-layers', nargs='+', type=int, default=[32, 32], help='morphological mlp head')
parser.add_argument('--labeling-module', default='', type=str, help='the type of labeling module used in training for '
                                                                    'pseudo label generation')
parser.add_argument('--sanity-check', action='store_true', help='apply sanity check')
parser.add_argument('--negative-pseudo', action='store_true', help='using negative pseudo labels')
parser.add_argument('--teacher', action='store_true', help='using teacher for evaluations')
parser.add_argument('--train-size', default=None, type=int, help='number of training samples')
parser.add_argument('--valid-labels', nargs='+', type=int, default=None, help='valid labels for the dataset')
parser.add_argument('--multi-crop', action='store_true', help='enable multi-crop augmentation')
parser.add_argument('--disable-cache', action='store_true', help='disable caching in dataset')
parser.add_argument('--profiler', action='store_true', help='enable profiler')
parser.add_argument('--normalize-embedding', action='store_true', help='normalize embeddings')

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(config):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        config['rank'] = int(os.environ["RANK"])
        config['world_size'] = int(os.environ['WORLD_SIZE'])
        config['gpu'] = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        config['rank'] = int(os.environ['SLURM_PROCID'])
        config['gpu'] = config['rank'] % torch.cuda.device_count()
    elif torch.cuda.is_available():
        pass
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    print(config, flush=True)

    dist.init_process_group(
        backend="nccl",
        init_method=config['dist_url'],
        world_size=config['world_size'],
        rank=config['rank'],
    )

    torch.cuda.set_device(config['gpu'])
    print('| distributed init (rank {}): {}'.format(
        config['rank'], config['dist_url']), flush=True)
    dist.barrier()
    setup_for_distributed(config['rank'] == 0)


def init_profiler(config: dict):
    if not config['profiler']:
        return None
    profiler = torch.profiler.profile(
        on_trace_ready=torch.profiler.tensorboard_trace_handler,
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        with_stack=True,
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    )
    return profiler


def train_moco(config, reporter=None):
    exec_time = datetime.datetime.now()

    config['save_dir'] = os.path.join(config['save_dir'], 'self-similarity', config['dataset'], config['arch'])
    if config['pretrained']:
        config['save_dir'] = os.path.join(config['save_dir'],
                                          'pretrained_from_{}'.format(config['pretrained'].split('/')[-1]))
    config['save_dir'] = os.path.join(config['save_dir'],
                                      '''id_{}_mlp_{}_dim_{}_lr_{}_bs_{}_apex_{}_optim_{}_mlpem_{}_specnorm_{}_focgam_{}_queuesize_{}_co2_{}_{}_emedding_{}_{}_moco_{}_ps_{}_env_{}_mask_{}_{}_mskcell_{}_catmskenv_{}_time_{}'''.format(
                                          config['job_id'],
                                          '_'.join([str(x) for x in config['mlp']]),
                                          config['moco_dim'],
                                          config['lr'],
                                          config['batch_size'],
                                          config['apex'],
                                          config['optim'],
                                          config['mlp_embedding'],
                                          config['spectral_norm'],
                                          config['focal_gamma'],
                                          config['queue_size'],
                                          config['co2_weight'],
                                          config['co2_t'],
                                          config['embedding_size'],
                                          config['commitment_cost'],
                                          config['moco_type'],
                                          config['patch_size'],
                                          config['env_arch'],
                                          config['mask_ratio'],
                                          config['env_coef'],
                                          config['mask_cells'],
                                          config['maskedenv_cat'],
                                          exec_time.strftime(
                                              "%Y%m%d-%H%M%S")))
    if not os.path.exists(config['save_dir']):
        os.makedirs(config['save_dir'], exist_ok=True)

    config['lr'] = config['lr'] * config['batch_size'] / 256

    if config['gpu'] is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if config['dist_url'] == "env://" and config['world_size'] == -1:
        config['world_size'] = int(os.environ["WORLD_SIZE"])

    config['distributed'] = config['world_size'] > 1 or config['multiprocessing_distributed']

    if config['distributed']:
        init_distributed_mode(config)

    ngpus_per_node = torch.cuda.device_count()
    if config['multiprocessing_distributed']:
        config['world_size'] = ngpus_per_node * config['world_size']
        main_worker(config['gpu'], ngpus_per_node, config, reporter)
    else:
        main_worker(config['gpu'], ngpus_per_node, config, reporter)


def main_worker(gpu, ngpus_per_node, config, reporter):
    config['gpu'] = gpu

    # suppress printing if not master
    if config['multiprocessing_distributed'] and config['gpu'] != 0:
        def print_pass(*args):  # "prevent other threads from writing logs"
            pass

        builtins.print = print_pass

    profiler = None
    if not config['multiprocessing_distributed'] or (config['multiprocessing_distributed'] and config['rank'] == 0):
        profiler = init_profiler(config)

    model = moco.build(config)
    model.apply(weight_init)

    # todo: refactor this section
    distribution_func_arg = {}
    if config['distributed']:
        if config['gpu'] is not None:
            model.cuda(config['gpu'])
            config['batch_size'] = int(config['batch_size'] / ngpus_per_node)
            config['workers'] = int((config['workers'] + ngpus_per_node - 1) / ngpus_per_node)
            distribution_func_arg['device_ids'] = [config['gpu']]
        else:
            model.cuda()
    elif config['gpu'] is not None:
        torch.cuda.set_device(config['gpu'])
        model = model.cuda(config['gpu'])
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    if config['focal_gamma'] == 0:
        if config['smoothing_alpha'] == 0:
            criterion = MaskedCrossEntropyLoss().cuda(config['gpu'])
        else:
            criterion = LabelSmoothing(config['smoothing_alpha'])
    else:
        criterion = focal_loss(gamma=config['focal_gamma'])

    # build the optimizer
    optimizer = build_optimizer(config, model)

    # parallelize the model
    if config['distributed']:
        if config['apex']:
            model, optimizer = amp.initialize(model, optimizer, opt_level=config['optim_level'])
            model = apex.parallel.convert_syncbn_model(model)  # replace BatchNorm with SyncBatchNorm
            model = apex.parallel.DistributedDataParallel(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, **distribution_func_arg)

    cudnn.benchmark = True

    # Data loading code
    train_dir = os.path.join(config['data'], 'train')
    test_dir = os.path.join(config['data'], 'test')

    # -------------------------------- dataset -------------------------------
    train_dataset, val_dataset = get_dataset(config, test_dir, train_dir)

    # -------------------------------- dataloader -------------------------------
    if config['distributed']:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    train_loader, val_loader = get_data_loaders(config, train_dataset, train_sampler, val_dataset, val_sampler)

    # create the labeling module
    labeling_module = None
    if config['labeling_module'] == 'morphological' or config['labeling_module'] == 'hovernet':
        labeling_module = FeatureClustering(n_classes=config['n_classes'], sanity_check=config['sanity_check'])

    # training labeling module
    if labeling_module is not None:
        labeling_module.fit(np.array(train_loader.dataset.extra_features),
                            np.array(train_loader.dataset.targets))

    # -------------------------------- training -------------------------------
    train_start = time.time()
    best_acc1 = 0

    for epoch in range(config['start_epoch'], config['epochs']):
        last_epoch = epoch == (config['epochs'] - 1)
        if config['distributed']:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, config)

        # train for one epoch
        train_loss, train_cell_loss, train_env_loss = train(train_loader, model, criterion, optimizer, epoch, config, labeling_module)

        # evaluate on validation set
        if (config['validation_interval'] != 0) and (epoch % config['validation_interval'] == 0 or last_epoch):
            test_nn_acc, test_kmeans_metric, test_standalone_kmeans_metric, (test_embedding, test_labels) = \
                validation(model, train_loader, val_loader, config)

            if not config['multiprocessing_distributed'] or config['rank'] % ngpus_per_node == 0:
                is_best = test_nn_acc > best_acc1
                best_acc1 = max(test_nn_acc, best_acc1)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': config['arch'],
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best=is_best, root=config['save_dir'])

        if reporter is not None:
            reporter(best_acc1, train_loss)

    if config['validation_interval'] == 0:
        save_checkpoint({
            'epoch': config['epochs'] - 1,
            'arch': config['arch'],
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, is_best=True, root=config['save_dir'])

    train_end = time.time()
    return best_acc1, train_loss


def get_data_loaders(config, train_dataset, train_sampler, val_dataset, val_sampler):
    # training for process
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=(train_sampler is None), collate_fn=collate_fn,
        num_workers=config['workers'], pin_memory=True, sampler=train_sampler, drop_last=True)
    # validation
    val_loader_whole = torch.utils.data.DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn,
        num_workers=config['workers'], pin_memory=True, sampler=val_sampler)
    return train_loader, val_loader_whole


def get_hparam(config):
    hparams = {'lr': config['lr'], 'batch size': config['batch_size'],
               'arch': config['arch'], 'optimizer': config['optim'],
               'image size': config['image_size'], 'epochs': config['epochs'],
               'betas': str(config['betas']), 'warmup epoch': config['warmup_epoch'],
               'schedule': str(config['schedule']), 'momentum': config['momentum'],
               'weight decay': config['weight_decay'], 'moco dim': config['moco_dim'],
               'moco m': config['moco_m'], 'moco t': config['moco_t'],
               'mlp': str(config['mlp']), 'prediction_head': config['prediction_head'],
               'mlp_embedding': config['mlp_embedding'], 'spectral_norm': config['spectral_norm'],
               'focal_gamma': config['focal_gamma'], 'queue_size': config['queue_size'],
               'co2_weight': config['co2_weight'], 'co2_t': config['co2_t'], 'embedding_size': config['embedding_size'],
               'commitment_cost': config['commitment_cost'], 'mask_ratio': config['mask_ratio'],
               'env_coef': config['env_coef'], 'mask_cells': config['mask_cells'],
               'maskedenv_cat': config['maskedenv_cat'],
               'vertical flip': config['vertical_flip'], 'rotation': config['rotation'], 'cos': config['cos']}
    return hparams


def compose_augmentations(augmentations, multi_crop=False):
    base_transform = albumentations.Compose(augmentations)
    if not multi_crop:
        return base_transform, None
    augmentations = [x for x in augmentations if not isinstance(x, albumentations.RandomResizedCrop)]
    whole_view_transform = albumentations.Compose(augmentations)
    return whole_view_transform, base_transform


def get_dataset(config: dict, test_dir: str, train_dir: str):
    # get image augmentation and transformations
    image_augmentation = get_augmentation(config)
    test_transforms = [
        albumentations.Resize(config['image_size'], config['image_size'], interpolation=cv2.INTER_CUBIC),
    ]

    # get patch augmentations
    patch_train_augmentation = get_patch_augmentation(config)
    patch_test_augmentation = []

    # setup multiprocess dictionary for caching
    multi_processing_manager = Manager()
    train_shared_dictionaries = {
        'cache_patch': multi_processing_manager.dict(),
        'cache_segmentation': multi_processing_manager.dict(),
        'cache_morphological': multi_processing_manager.dict()
    }
    test_shared_dictionaries = {
        'cache_patch': multi_processing_manager.dict(),
        'cache_segmentation': multi_processing_manager.dict(),
        'cache_morphological': multi_processing_manager.dict()
    }

    if config['dataset'] == 'consep':

        # add image normalization to train and test transformations
        normalization = [
            consep_dataset.get_cell_normalization(),
            ToTensorV2(transpose_mask=True)
        ]
        image_augmentation.extend(normalization)
        test_transforms.extend(normalization)

        # add patch normalization to train and test transformations
        patch_normalization = [
            consep_dataset.get_patch_normalization(config['patch_size']),
            ToTensorV2(transpose_mask=True)
        ]
        patch_train_augmentation.extend(patch_normalization)
        patch_test_augmentation.extend(patch_normalization)

        train_dataset = CoNSePDataset(train_dir,
                                      transform=moco.loader.TwoCropsTransform(
                                          *compose_augmentations(image_augmentation, config['multi_crop'])),
                                      target_transform=consep_dataset.LabelTransform(n_classes=config['n_classes']),
                                      patch_transform=moco.loader.TwoCropsTransform(
                                          *compose_augmentations(patch_train_augmentation, config['multi_crop'])),
                                      patch_size=config['patch_size'],
                                      mask_ratio=config['mask_ratio'],
                                      hovernet_enable=config['labeling_module'] == 'hovernet',
                                      dataset_size=config['train_size'],
                                      valid_labels=config['valid_labels'],
                                      cache_patch=not config['disable_cache'],
                                      shared_dictionaries=train_shared_dictionaries)
        test_dataset = CoNSePDataset(test_dir,
                                     transform=albumentations.Compose(test_transforms),
                                     target_transform=consep_dataset.LabelTransform(n_classes=config['n_classes']),
                                     patch_transform=albumentations.Compose(patch_test_augmentation),
                                     patch_size=config['patch_size'],
                                     mask_ratio=config['mask_ratio'],
                                     hovernet_enable=False,
                                     valid_labels=config['valid_labels'],
                                     cache_patch=not config['disable_cache'],
                                     shared_dictionaries=test_shared_dictionaries
                                     )

    elif config['dataset'] == 'nucls' or config['dataset'] == 'nucls2':

        # add image normalization to train and test transformations
        normalization = [
            nucls_dataset.get_cell_normalization(),
            ToTensorV2(transpose_mask=True)
        ]
        image_augmentation.extend(normalization)
        test_transforms.extend(normalization)

        # add patch normalization to train and test transformations
        patch_normalization = [
            nucls_dataset.get_patch_normalization(config['patch_size']),
            ToTensorV2(transpose_mask=True)
        ]
        patch_train_augmentation.extend(patch_normalization)
        patch_test_augmentation.extend(patch_normalization)

        train_dataset = NuCLSDataset(train_dir,
                                     transform=moco.loader.TwoCropsTransform(
                                         *compose_augmentations(image_augmentation, config['multi_crop'])),
                                     target_transform=nucls_dataset.LabelTransform(n_classes=config['n_classes']),
                                     patch_transform=moco.loader.TwoCropsTransform(
                                         *compose_augmentations(patch_train_augmentation, config['multi_crop'])),
                                     patch_size=config['patch_size'],
                                     mask_ratio=config['mask_ratio'],
                                     hovernet_enable=config['labeling_module'] == 'hovernet',
                                     dataset_size=config['train_size'],
                                     valid_labels=config['valid_labels'],
                                     cache_patch=not config['disable_cache'],
                                     shared_dictionaries=train_shared_dictionaries)
        test_dataset = NuCLSDataset(test_dir,
                                    transform=albumentations.Compose(test_transforms),
                                    target_transform=nucls_dataset.LabelTransform(n_classes=config['n_classes']),
                                    patch_transform=albumentations.Compose(patch_test_augmentation),
                                    patch_size=config['patch_size'],
                                    mask_ratio=config['mask_ratio'],
                                    hovernet_enable=False,
                                    valid_labels=config['valid_labels'],
                                    cache_patch=not config['disable_cache'],
                                    shared_dictionaries=test_shared_dictionaries)

    elif config['dataset'] == 'ovarian':
        # add image normalization to train and test transformations
        normalization = [
            ovarian_dataset.get_cell_normalization(),
            ToTensorV2(transpose_mask=True)
        ]
        image_augmentation.extend(normalization)
        test_transforms.extend(normalization)

        # add patch normalization to train and test transformations
        patch_normalization = [
            ovarian_dataset.get_patch_normalization(config['patch_size']),
            ToTensorV2(transpose_mask=True)
        ]
        patch_train_augmentation.extend(patch_normalization)
        patch_test_augmentation.extend(patch_normalization)

        train_dataset = OvarianDataset(train_dir,
                                     transform=moco.loader.TwoCropsTransform(
                                         *compose_augmentations(image_augmentation, config['multi_crop'])),
                                     target_transform=ovarian_dataset.LabelTransform(n_classes=config['n_classes']),
                                     patch_transform=moco.loader.TwoCropsTransform(
                                         *compose_augmentations(patch_train_augmentation, config['multi_crop'])),
                                     patch_size=config['patch_size'],
                                     mask_ratio=config['mask_ratio'],
                                     hovernet_enable=config['labeling_module'] == 'hovernet',
                                     dataset_size=config['train_size'],
                                     valid_labels=config['valid_labels'],
                                     cache_patch=not config['disable_cache'],
                                     shared_dictionaries=train_shared_dictionaries)
        test_dataset = OvarianDataset(test_dir,
                                    transform=albumentations.Compose(test_transforms),
                                    target_transform=ovarian_dataset.LabelTransform(n_classes=config['n_classes']),
                                    patch_transform=albumentations.Compose(patch_test_augmentation),
                                    patch_size=config['patch_size'],
                                    mask_ratio=config['mask_ratio'],
                                    hovernet_enable=False,
                                    valid_labels=config['valid_labels'],
                                    cache_patch=not config['disable_cache'],
                                    shared_dictionaries=test_shared_dictionaries)

    elif config['dataset'] == 'sarcoma':
        # add image normalization to train and test transformations
        normalization = [
            sarcoma_dataset.get_cell_normalization(),
            ToTensorV2(transpose_mask=True)
        ]
        image_augmentation.extend(normalization)
        test_transforms.extend(normalization)

        # add patch normalization to train and test transformations
        patch_normalization = [
            sarcoma_dataset.get_patch_normalization(config['patch_size']),
            ToTensorV2(transpose_mask=True)
        ]
        patch_train_augmentation.extend(patch_normalization)
        patch_test_augmentation.extend(patch_normalization)

        train_dataset = SarcomaDataset(train_dir,
                                     transform=moco.loader.TwoCropsTransform(
                                         *compose_augmentations(image_augmentation, config['multi_crop'])),
                                     target_transform=sarcoma_dataset.LabelTransform(n_classes=config['n_classes']),
                                     patch_transform=moco.loader.TwoCropsTransform(
                                         *compose_augmentations(patch_train_augmentation, config['multi_crop'])),
                                     patch_size=config['patch_size'],
                                     mask_ratio=config['mask_ratio'],
                                     hovernet_enable=config['labeling_module'] == 'hovernet',
                                     dataset_size=config['train_size'],
                                     valid_labels=config['valid_labels'],
                                     cache_patch=not config['disable_cache'],
                                     shared_dictionaries=train_shared_dictionaries)
        test_dataset = SarcomaDataset(test_dir,
                                    transform=albumentations.Compose(test_transforms),
                                    target_transform=sarcoma_dataset.LabelTransform(n_classes=config['n_classes']),
                                    patch_transform=albumentations.Compose(patch_test_augmentation),
                                    patch_size=config['patch_size'],
                                    mask_ratio=config['mask_ratio'],
                                    hovernet_enable=False,
                                    valid_labels=config['valid_labels'],
                                    cache_patch=not config['disable_cache'],
                                    shared_dictionaries=test_shared_dictionaries)

    elif config['dataset'] == 'pannuke':
        # add image normalization to train and test transformations
        normalization = [
            pannuke_dataset.get_cell_normalization(),
            ToTensorV2(transpose_mask=True)
        ]
        image_augmentation.extend(normalization)
        test_transforms.extend(normalization)

        # add patch normalization to train and test transformations
        patch_normalization = [
            pannuke_dataset.get_patch_normalization(config['patch_size']),
            ToTensorV2(transpose_mask=True)
        ]
        patch_train_augmentation.extend(patch_normalization)
        patch_test_augmentation.extend(patch_normalization)

        train_dataset = PanNukeDataset(train_dir,
                                     transform=moco.loader.TwoCropsTransform(
                                         *compose_augmentations(image_augmentation, config['multi_crop'])),
                                     target_transform=pannuke_dataset.LabelTransform(n_classes=config['n_classes']),
                                     patch_transform=moco.loader.TwoCropsTransform(
                                         *compose_augmentations(patch_train_augmentation, config['multi_crop'])),
                                     patch_size=config['patch_size'],
                                     mask_ratio=config['mask_ratio'],
                                     hovernet_enable=config['labeling_module'] == 'hovernet',
                                     dataset_size=config['train_size'],
                                     valid_labels=config['valid_labels'],
                                     cache_patch=not config['disable_cache'],
                                     shared_dictionaries=train_shared_dictionaries)
        test_dataset = PanNukeDataset(test_dir,
                                    transform=albumentations.Compose(test_transforms),
                                    target_transform=pannuke_dataset.LabelTransform(n_classes=config['n_classes']),
                                    patch_transform=albumentations.Compose(patch_test_augmentation),
                                    patch_size=config['patch_size'],
                                    mask_ratio=config['mask_ratio'],
                                    hovernet_enable=False,
                                    valid_labels=config['valid_labels'],
                                    cache_patch=not config['disable_cache'],
                                    shared_dictionaries=test_shared_dictionaries)

    elif config['dataset'] == 'lizard':
        # add image normalization to train and test transformations
        normalization = [
            lizard_dataset.get_cell_normalization(),
            ToTensorV2(transpose_mask=True)
        ]
        image_augmentation.extend(normalization)
        test_transforms.extend(normalization)

        # add patch normalization to train and test transformations
        patch_normalization = [
            lizard_dataset.get_patch_normalization(config['patch_size']),
            ToTensorV2(transpose_mask=True)
        ]
        patch_train_augmentation.extend(patch_normalization)
        patch_test_augmentation.extend(patch_normalization)

        train_dataset = LizardDataset(train_dir,
                                     transform=moco.loader.TwoCropsTransform(
                                         *compose_augmentations(image_augmentation, config['multi_crop'])),
                                     target_transform=lizard_dataset.LabelTransform(n_classes=config['n_classes']),
                                     patch_transform=moco.loader.TwoCropsTransform(
                                         *compose_augmentations(patch_train_augmentation, config['multi_crop'])),
                                     patch_size=config['patch_size'],
                                     mask_ratio=config['mask_ratio'],
                                     hovernet_enable=config['labeling_module'] == 'hovernet',
                                     dataset_size=config['train_size'],
                                     valid_labels=config['valid_labels'],
                                     cache_patch=not config['disable_cache'],
                                     shared_dictionaries=train_shared_dictionaries)
        test_dataset = LizardDataset(test_dir,
                                    transform=albumentations.Compose(test_transforms),
                                    target_transform=lizard_dataset.LabelTransform(n_classes=config['n_classes']),
                                    patch_transform=albumentations.Compose(patch_test_augmentation),
                                    patch_size=config['patch_size'],
                                    mask_ratio=config['mask_ratio'],
                                    hovernet_enable=False,
                                    valid_labels=config['valid_labels'],
                                    cache_patch=not config['disable_cache'],
                                    shared_dictionaries=test_shared_dictionaries)

    elif config['dataset'] == 'oracle':
        # add image normalization to train and test transformations
        normalization = [
            oracle_dataset.get_cell_normalization(),
            ToTensorV2(transpose_mask=True)
        ]
        image_augmentation.extend(normalization)
        test_transforms.extend(normalization)

        # add patch normalization to train and test transformations
        patch_normalization = [
            oracle_dataset.get_patch_normalization(config['patch_size']),
            ToTensorV2(transpose_mask=True)
        ]
        patch_train_augmentation.extend(patch_normalization)
        patch_test_augmentation.extend(patch_normalization)

        train_dataset = OracleDataset(train_dir,
                                    transform=moco.loader.TwoCropsTransform(
                                        *compose_augmentations(image_augmentation, config['multi_crop'])),
                                    target_transform=oracle_dataset.LabelTransform(n_classes=config['n_classes']),
                                    patch_transform=moco.loader.TwoCropsTransform(
                                        *compose_augmentations(patch_train_augmentation, config['multi_crop'])),
                                    patch_size=config['patch_size'],
                                    mask_ratio=config['mask_ratio'],
                                    hovernet_enable=config['labeling_module'] == 'hovernet',
                                    dataset_size=config['train_size'],
                                    valid_labels=config['valid_labels'],
                                    cache_patch=not config['disable_cache'],
                                    shared_dictionaries=train_shared_dictionaries)
        test_dataset = OracleDataset(test_dir,
                                   transform=albumentations.Compose(test_transforms),
                                   target_transform=oracle_dataset.LabelTransform(n_classes=config['n_classes']),
                                   patch_transform=albumentations.Compose(patch_test_augmentation),
                                   patch_size=config['patch_size'],
                                   mask_ratio=config['mask_ratio'],
                                   hovernet_enable=False,
                                   valid_labels=config['valid_labels'],
                                   cache_patch=not config['disable_cache'],
                                   shared_dictionaries=test_shared_dictionaries)

    elif config['dataset'] == 'cifar10':
        normalization = [
            albumentations.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ToTensorV2(transpose_mask=True)
        ]
        image_augmentation.extend(normalization)
        test_transforms.extend(normalization)
        train_dataset = torchvision.datasets.CIFAR10(root=config['data'],
                                                     train=True,
                                                     download=True,
                                                     transform=moco.loader.TwoCropsTransform(
                                                         *compose_augmentations(image_augmentation,
                                                                                config['multi_crop']))
                                                     )
        test_dataset = torchvision.datasets.CIFAR10(root=config['data'],
                                                    train=False,
                                                    download=True,
                                                    transform=albumentations.compose(test_transforms)
                                                    )
    elif config['dataset'] == 'imagenet':
        normalization = [
            albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(transpose_mask=True)
        ]
        image_augmentation.extend(normalization)
        test_transforms.extend(normalization)
        train_dataset = torchvision.datasets.ImageNet(root=config['data'],
                                                      train=True,
                                                      download=True,
                                                      transform=moco.loader.TwoCropsTransform(
                                                          *compose_augmentations(image_augmentation,
                                                                                 config['multi_crop']))
                                                      )
        test_dataset = torchvision.datasets.ImageNet(root=config['data'],
                                                     train=False,
                                                     download=True,
                                                     transform=albumentations.Compose(test_transforms)
                                                     )
    return train_dataset, test_dataset


def get_augmentation(config):
    augmentation = [
        albumentations.Resize(config['image_size'], config['image_size'], interpolation=cv2.INTER_CUBIC),
        albumentations.RandomResizedCrop(config['image_size'], config['image_size'], scale=(0.2, 1.)),
        albumentations.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        albumentations.ToGray(p=0.2),
        albumentations.GaussianBlur(blur_limit=0, sigma_limit=(0.1, 2.0), p=0.5),
        albumentations.HorizontalFlip(),
    ]
    if config['vertical_flip']:
        augmentation.append(albumentations.VerticalFlip())
    if config['rotation']:
        augmentation.append(albumentations.Rotate(180))
    return augmentation


def get_patch_augmentation(config):
    augmentation = [
        albumentations.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        albumentations.ToGray(p=0.2),
        albumentations.GaussianBlur(blur_limit=0, sigma_limit=(0.1, 2.0), p=0.5),
        albumentations.HorizontalFlip()
    ]
    if config['vertical_flip']:
        augmentation.append(albumentations.VerticalFlip())
    if config['rotation']:
        augmentation.append(albumentations.Rotate(180))
    return augmentation

def multi_label_ctr(logits, pseudo_label, tau):
    """
    Multi-label NCE loss with pseudo labels
    :param logits: values from the model (N * M where N: number of anchors, M: number of samples)
    :param pseudo_label: labels of each sample (M: number of samples)
    :param tau: value to be used for loss
    :return: multi-label loss
    """
    if not isinstance(logits, torch.Tensor):
        logits = torch.FloatTensor(logits)
    if not isinstance(pseudo_label, torch.Tensor):
        pseudo_label = torch.LongTensor(pseudo_label)
    assert len(pseudo_label.size()) == 1, 'just one dimensional vector is handled'
    # change criterion to binary classification
    criterion = torch.nn.BCEWithLogitsLoss()
    # convert labels to a pairwise score matrix
    anchor_size = logits.size(0)
    label = pseudo_label[:anchor_size, None] != pseudo_label[None, :]
    label = label.type(torch.FloatTensor).to(logits.device)
    return 2 * tau * criterion(logits / tau, label)

def ctr(q, k, criterion, tau, loss_mask, pseudo_label):
    if loss_mask is not None:
        assert torch.all(torch.diag(loss_mask))  # make sure the diag is all set
    N = q.size(0)
    logits = torch.mm(q, k.t())
    if pseudo_label is not None:
        assert loss_mask is None, 'multiple label with loss mask is not implemented yet!'
        return multi_label_ctr(logits, pseudo_label, tau)
    labels = range(N)
    labels = torch.LongTensor(labels).cuda()
    loss = criterion(logits / tau, labels, mask=loss_mask)
    return 2 * tau * loss



def get_loss_mask(query_slide_id, query_patch_coordinates, key_slide_id, key_patch_coordinate, patch_size,
                  diagonal=True):
    if query_slide_id is None or key_slide_id is None:
        return None
    # slide id mask
    slide_id_mask = torch.zeros((query_slide_id.size(0), key_slide_id.size(0)), dtype=torch.bool,
                                device=query_slide_id.device)
    slide_id_mask[query_slide_id.unsqueeze(1) != key_slide_id] = True

    # coordinate mask
    coordinate_diff = torch.abs(query_patch_coordinates[:, None] - key_patch_coordinate[None, :])
    patch_coordinate_mask = (coordinate_diff[:, :, 0] > patch_size) & (coordinate_diff[:, :, 1] > patch_size)

    # return if not diagonal is set
    if not diagonal:
        return slide_id_mask | patch_coordinate_mask

    # add diagonal true label
    diagonal_matrix = torch.zeros((query_slide_id.size(0), key_slide_id.size(0)), dtype=torch.bool,
                                  device=query_slide_id.device)
    diagonal_matrix.fill_diagonal_(True)
    return slide_id_mask | patch_coordinate_mask | diagonal_matrix


def train(train_loader, model, criterion, optimizer, epoch, config, labeling_module):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    total_losses = AverageMeter('Loss', ':.4e')
    env_losses = AverageMeter('Loss', ':.4e')
    cell_losses = AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, total_losses, cell_losses, env_losses],
        prefix="Epoch: [{}]".format(epoch), logger=None)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _, patch, slide_id, coordinates, mask, segmentation, extra_feat) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if mask is not None:  # set masked values to zero
            patch[0][mask[0].unsqueeze(1).repeat((1, 3, 1, 1))] = 0
            patch[1][mask[1].unsqueeze(1).repeat((1, 3, 1, 1))] = 0

        if segmentation is not None and config['mask_cells']:
            patch[0][segmentation[0].unsqueeze(1).repeat((1, 3, 1, 1))] = 0
            patch[1][segmentation[1].unsqueeze(1).repeat((1, 3, 1, 1))] = 0

        if config['gpu'] is not None:
            images[0] = images[0].cuda(config['gpu'], non_blocking=True)
            images[1] = images[1].cuda(config['gpu'], non_blocking=True)
            slide_id = slide_id.cuda(config['gpu'], non_blocking=True)
            coordinates = coordinates.cuda(config['gpu'], non_blocking=True)
            extra_feat = extra_feat.cuda(config['gpu'], non_blocking=True)
            if patch is not None:
                patch[0] = patch[0].cuda(config['gpu'], non_blocking=True)
                patch[1] = patch[1].cuda(config['gpu'], non_blocking=True)

        # todo: refactor this
        # compute output
        (q1, q2), (k1, k2), (q_env_1, q_env_2, env), (key_slide_id, key_patch_coordinate), (extra_feat1, _), \
        k1_instances, k2_instances, quantization_loss = \
            model(x1=images[0], x2=images[1],
                  patch1=patch[0] if patch is not None else None,
                  patch2=patch[1] if patch is not None else None,
                  extra_feat1=extra_feat, extra_feat2=extra_feat,
                  patch_meta_data=(slide_id, coordinates))

        # generate pseudo labels if you have any
        pseudo_label = None
        if labeling_module is not None:
            pseudo_label = labeling_module.predict(extra_feat1)
            if isinstance(pseudo_label, np.ndarray):
                pseudo_label = torch.LongTensor(pseudo_label)
            pseudo_label = pseudo_label.to(extra_feat1.device)

        loss_mask = None
        if config['moco_type'].lower() == 'env':
            loss_mask = get_loss_mask(slide_id, coordinates, key_slide_id, key_patch_coordinate,
                                      config['patch_size'], diagonal=True)

        if labeling_module is not None and config['negative_pseudo']:
            anchor_size = q1.size(0)
            loss_mask = pseudo_label[:anchor_size, None] != pseudo_label[None, :]
            loss_mask.fill_diagonal_(True)
            pseudo_label = None

        loss = ctr(q1, k2, criterion, config['moco_t'], loss_mask, pseudo_label) + \
               ctr(q2, k1, criterion, config['moco_t'], loss_mask, pseudo_label)
        cell_loss = loss

        if quantization_loss is not None:
            loss = loss + quantization_loss

        env_loss = torch.zeros(1, dtype=torch.float)
        if q_env_1 is not None and q_env_2 is not None:
            env_loss = ctr(q_env_1, env, criterion, config['moco_t'], None, pseudo_label) + \
                       ctr(q_env_2, env, criterion, config['moco_t'], None, pseudo_label)
            env_loss = config['env_coef'] * env_loss
            loss = loss + env_loss

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        cell_losses.update(cell_loss.item(), images[0].size(0))
        env_losses.update(env_loss.item(), images[0].size(0))
        total_losses.update(loss.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        # backward based on if it's apex or not
        if not config['apex']:
            loss.backward()
        else:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

        # take optimizer and profiling steps
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config['print_freq'] == 0:
            progress.display(i)

    return total_losses.avg, cell_losses.avg, env_losses.avg  # todo: collect this from other processes too


def validation(model, train_loader, test_loader, config):
    correct = 0.
    total = 0
    test_size = len(test_loader.dataset)

    # set in eval mode
    model.eval()

    with torch.no_grad():

        # switch transform to not augment the dataset
        original_transform = train_loader.dataset.transform
        train_loader.dataset.transform = test_loader.dataset.transform

        # switch patch transform to not augment the dataset
        original_patch_transform = train_loader.dataset.patch_transform
        train_loader.dataset.patch_transform = test_loader.dataset.patch_transform

        test_embedding = []
        test_true_labels = []

        # compute train features
        train_features = []
        train_labels = []
        for images, target, patch, _, _, _, segmentation, extra_feat in train_loader:

            if segmentation is not None and config['mask_cells']:
                patch[segmentation.unsqueeze(1).repeat((1, 3, 1, 1))] = 0

            if config['gpu'] is not None:
                images = images.cuda(config['gpu'], non_blocking=True)
                extra_feat = extra_feat.cuda(config['gpu'], non_blocking=True)
                if patch is not None:
                    patch = patch.cuda(config['gpu'], non_blocking=True)

            # get embedding
            output, patch_embedding = model(x1=images, patch1=patch, extra_feat1=extra_feat, return_patch=True,
                                            normalize_embedding=config['normalize_embedding'])
            if config['maskedenv_cat']:
                output = torch.cat((output, patch_embedding), dim=1)
            train_features.append(output)
            train_labels.append(target)
        train_features = torch.cat(train_features, dim=0)
        train_labels = torch.cat(train_labels, dim=0)

        # Gather all the tensors from all GPUs
        if torch.distributed.is_initialized():
            train_features = concat_all_gather(train_features.cuda())
            train_labels = concat_all_gather(train_labels.cuda())
        train_features = train_features.cpu()
        train_labels = train_labels.cpu()

        # train classifiers
        kmeans_classifier = KMeans(n_clusters=config['n_classes'], n_jobs=-1).fit(train_features)
        if config['euclidean_nn']:
            nn_classifier = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
            nn_classifier.fit(train_features, train_labels)

        # prediction test samples
        for batch_idx, (images, targets, patch, _, _, _, _, extra_feat) in enumerate(test_loader):
            if config['gpu'] is not None:
                images = images.cuda(config['gpu'], non_blocking=True)
                extra_feat = extra_feat.cuda(config['gpu'], non_blocking=True)
                if patch is not None:
                    patch = patch.cuda(config['gpu'], non_blocking=True)
            batch_size = images.size(0)
            features, patch_embedding = model(x1=images, patch1=patch, extra_feat1=extra_feat, return_patch=True,
                                              normalize_embedding=config['normalize_embedding'])
            if config['maskedenv_cat']:
                features = torch.cat((features, patch_embedding), dim=1)
            test_embedding.append(features)
            test_true_labels.append(targets)

        # cat the results
        test_embedding = torch.cat(test_embedding, dim=0)
        test_true_labels = torch.cat(test_true_labels, dim=0)

        # Gather all the tensors from all GPUs
        if torch.distributed.is_initialized():
            test_embedding = concat_all_gather(test_embedding.cuda())
            test_true_labels = concat_all_gather(test_true_labels.cuda())
        test_embedding = test_embedding.cpu()
        test_true_labels = test_true_labels.cpu()

        if not config['euclidean_nn']:
            distances = torch.mm(test_embedding, train_features.t())

            yd, yi = distances.topk(1, dim=1, largest=True, sorted=True)
            candidates = train_labels.view(1, -1).expand(test_embedding.size(0), -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval = retrieval.narrow(1, 0, 1).clone().view(-1)
            yd = yd.narrow(1, 0, 1)
        else:
            retrieval = torch.LongTensor(nn_classifier.predict(test_embedding))
        #            retrieval = torch.LongTensor(nn_classifier.predict(features))

        total += test_true_labels.size(0)
        correct += retrieval.eq(test_true_labels.data).sum().item()
        #        correct += retrieval.eq(targets.data).sum().item()

        test_cluster_prediction = kmeans_classifier.predict(test_embedding)
        kmeans_metrics = clustering_metrics(test_true_labels.numpy(), test_cluster_prediction)
        standalone_kmeans = clustering_metrics(test_true_labels.numpy(), kmeans_classifier.fit_predict(test_embedding))

        # reset the original transform of the train dataset
        train_loader.dataset.transform = original_transform
        train_loader.dataset.patch_transform = original_patch_transform

    return correct / total, kmeans_metrics, standalone_kmeans, (test_embedding, test_true_labels)


def save_checkpoint(state, is_best, root, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(root, filename))
    if is_best:
        shutil.copyfile(os.path.join(root, filename), os.path.join(root, 'model_best.pth.tar'))


def learning_rate_scheduler(epoch, config):
    """Schedules learning rate based on schedule"""
    lr = config['lr']
    if epoch < config['warmup_epoch']:
        return lr * (epoch + 1) / config['warmup_epoch']
    if config['cos']:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - config['warmup_epoch']) / config['epochs']))
    else:  # stepwise lr schedule
        for milestone in config['schedule']:
            lr *= 0.1 if (epoch - config['warmup_epoch']) >= milestone else 1.
    return lr


def adjust_learning_rate(optimizer, epoch, config):
    """Decay the optimizer learning rate based on schedule"""
    lr = learning_rate_scheduler(epoch, config)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output




if __name__ == '__main__':
    args = parser.parse_args()
    train_moco(vars(args))
