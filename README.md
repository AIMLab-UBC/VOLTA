# VOLTA: an Environment-Aware Contrastive Cell Representation Learning for Histopathology

## Introduction

In clinical practice, many diagnosis tasks rely on the identification of cells in histopathology images. While supervised machine learning techniques require labels, providing manual cell annotations is time-consuming due to the large number of cells. In this paper, we propose a self-supervised framework (VOLTA) for cell representation learning in histopathology images using a novel technique that accounts for the cell’s mutual relationship with its environment for improved cell representations. We subjected our model to extensive experiments on the data collected from multiple institutions around the world comprising of over 800,000 cells, six cancer types, and cell types ranging from two to six categories for each dataset. The results show that our model outperforms the state-of-the-art models in cell representation learning. To showcase the potential power of our proposed framework, we applied VOLTA to ovarian and endometrial cancers with very small sample sizes (10-20 samples) and demonstrated that our cell representations can be utilized to identify the known histotypes of ovarian cancer and provide novel insights that link histopathology and molecular subtypes of endometrial cancer. Unlike supervised deep learning models that require large sample sizes for training, we provide a framework that can empower new discoveries without any annotation data in situations where sample sizes are limited. This repository is a Pytorch implementation of this model.


[paper (Nature Communications)](https://www.nature.com/articles/s41467-024-48062-1)


## Requirements

As the first step, please use the `requirement.txt` file to install the requirements.


```
pip install -r requirements.txt
```

## Data Prepration

To prepare the data, after extracting patches from the whole slide image, you can use the `convertor.py` code with the appropriate flags within the patch2cell directory (use the help function to see the flags):

```
cd patch2cell
python convertor.py
```


## Run

To run the code, first create a data folder and add the above generaeted data to it. It should contain a folder with dataset name, two subfolders of train and test. Then, you can use the below command and adjust the configs to run the code.


```
python3 main.py --dataset nucls --image-size 32 --arch resnet18 --workers 2 --epochs 20 --batch-size 256 --lr 0.01 --n_classes 5 --moco-dim 32 --moco-t 0.07 --mlp 32 32 --prediction-head 32 --cos --euclidean-nn --validation-interval 0 --gpu 0 --moco-type maskedenv --patch-size 100 --env-arch resnet18 --mask-ratio 1.2 --env-coef 1.0 --teacher --multi-crop data/nucls
```


Below are the available configs.

```
- **data**: Path to the dataset.
- **--dataset**: Type of dataset, including options like 'consep', 'cifar', or 'imagenet'.
- **--local_rank**: Integer specifying the local rank (default: 0).
- **--image-size**: Integer specifying the size to rescale the input (default: 32).
- **-a, --arch**: Model architecture to be used (default: 'resnet50'). Available options depend on 'model_names'.
- **-j, --workers**: Number of data loading workers (default: 32).
- **--epochs**: Total number of epochs to run (default: 200).
- **--start-epoch**: Manual epoch number (useful on restarts) (default: 0).
- **-b, --batch-size**: Mini-batch size (default: 256). This is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel.
- **--optim, --optimizer**: Optimizer to use. Choices: 'sgd', 'adam', 'adamw', 'lars', 'lamb' (default: 'sgd').
- **--lr, --learning-rate**: Initial learning rate (default: 0.03).
- **--betas**: Initial betas for optimization (default: (0.9, 0.99)).
- **--warmup-epoch**: Number of epochs for warmup (default: 10).
- **--schedule**: Learning rate schedule (when to drop lr by 10x) after the warmup steps (default: [120, 160]).
- **--momentum**: Momentum of SGD solver (default: 0.9).
- **--wd, --weight-decay**: Weight decay (default: 1e-4).
- **-p, --print-freq**: Print frequency (default: 10).
- **--pretrained**: Path to pretrained checkpoint used for finetuning (default: none).
- **--rank**: Node rank for distributed training (default: -1).
- **--seed**: Seed for initializing training.
- **--gpu**: GPU id to use.
- **--gpus**: Comma-separated string specifying GPU IDs to use (default: "0").
- **--save-dir**: Directory to save models (default: 'checkpoints').
- **--multiprocessing-distributed**: Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the fastest way to use PyTorch for either single node or multi-node data parallel training.
- **--n_classes**: Number of classes to be used (default: 4).
- **--moco-dim**: Feature dimension (default: 128).
- **--moco-m**: Moco momentum of updating key encoder (default: 0.999).
- **--moco-t**: Softmax temperature (default: 0.07).
- **--mlp**: MLP head layer sizes.
- **--prediction-head**: Size of the prediction head MLP.
- **--mlp-embedding**: Add MLP head as an extra embedding layer.
- **--aug-plus**: Use moco v2 data augmentation.

```

## License

This repository is protected by https://creativecommons.org/licenses/by-nc/4.0/deed.en


## Citation

```
@article{nakhli2023volta,
  title={Volta: an environment-aware contrastive cell representation learning for histopathology},
  author={Nakhli, Ramin and Zhang, Allen and Farahani, Hossein and Darbandsari, Amirali and Shenasa, Elahe and Thiessen, Sidney and Milne, Katy and McAlpine, Jessica and Nelson, Brad and Gilks, C Blake and others},
  journal={arXiv preprint arXiv:2303.04696},
  year={2023}
}
```
