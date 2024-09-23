##### Table of contents
1. [Environment setup](#environment-setup)
2. [Dataset preparation](#dataset-preparation)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Contacts](#contacts)

# Official PyTorch implementation of "A High-Quality Robust Diffusion Framework for Corrupted Dataset" (ECCV'24)

<div align="center">
  <a href="https://quandao10.github.io/" target="_blank">Quan&nbsp;Dao</a> &emsp;
  <a href="https://github.com/Tahuubinh" target="_blank">Binh&nbsp;Ta</a> &emsp;
  <a href="https://github.com/" target="_blank">Tung&nbsp;Pham</a> &emsp;
  <a href="https://sites.google.com/site/anhttranusc/" target="_blank">Anh&nbsp;Tran</a>
  <br> <br>
  
  
  <a href="https://www.vinai.io/">VinAI Research</a>
</div>

> **Abstract**: Developing image-generative models, which are robust to outliers in the training process, has recently drawn attention from the research community. Due to the ease of integrating unbalanced optimal transport (UOT) into adversarial frameworks, existing works focus mainly on developing robust frameworks for generative adversarial model (GAN). Meanwhile, diffusion models have recently dominated GAN in various tasks and datasets. However, according to our knowledge, none of them are robust to corrupted datasets. Motivated by DDGAN, our work introduces the first robust-to-outlier diffusion. We suggest replacing the UOT-based generative model for GAN in DDGAN to learn the backward diffusion process. Additionally, we demonstrate that the Lipschitz property of divergence in our framework contributes to more stable training convergence. Remarkably, our method not only exhibits robustness to corrupted datasets but also achieves superior performance on clean datasets.

**TLDR**: This work introduces the first robust-to-outlier diffusion and suggests replacing the UOT-based generative model for GAN in DDGAN to learn the backward diffusion process, and demonstrates that the Lipschitz property of divergence in the framework contributes to more stable training convergence.

Details of algorithms, experimental results, and configurations can be found in [our following paper](https://arxiv.org/abs/2311.17101):

```bibtex
@InProceedings{dao2024highqualityrobustdiffusionframework,
    title     = {A High-Quality Robust Diffusion Framework for Corrupted Dataset},
    author    = {Quan Dao and Binh Ta and Tung Pham and Anh Tran},
    booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
    year      = {2024}
}
```

**Please CITE** our paper whenever this repository is used to help produce published results or incorporated into other software.

## Environment setup
First, install Pytorch v1.12.1:
```
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```
Then, install other modules using:
```
pip install -r requirements.txt
```

## Set up Datasets ##

We use these datasets to create corrupted datasets:

- **CIFAR-10** 
- **STL-10**
- **LSUN Church Outdoor 256**
- **CelebA HQ 256**
- **MNIST**
- **FashionMNIST**

For MNIST, FashionMNIST, CIFAR-10, and STL-10, they will be automatically downloaded in the first time execution. 

For CelebA HQ 256 and LSUN, please check out [here](https://github.com/NVlabs/NVAE#set-up-file-paths-and-data) for dataset preparation.

Once a dataset is downloaded, please put it in `data/` directory as follows:
```
data/
├── cifar-10-batches-py
├── STL-10
├── celeba-lmdb
├── LSUN
├── mnist 
└── fashion_mnist
```

Each corrupted dataset consists of two components: clean data and outliers. In our paper, we use **CIFAR-10**, **STL-10**, or **CelebA HQ 256** as the clean data; and any other datasets as the outliers.

## Training ##
We use the following commands to train our proposed model.

#### CIFAR-10 perturbed by MNIST (3%) ####

```
python3 train.py --dataset cifar10 --batch_size 256 --num_channels_dae 128 --num_epoch 1800 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --ch_mult 1 2 2 2 --version bs256 --perturb_dataset mnist --perturb_percent 3
```

#### STL-10 ####

```
python3 train.py --dataset stl10 --image_size 64 --num_channels_dae 128 --ch_mult 1 2 2 2 --batch_size 72 --num_epoch 1800 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 --version bs64 --tau 1e-4
```

#### CelebA HQ 256 ####

```
CUDA_VISIBLE_DEVICES=0,1 python3 train.py --image_size 256 --dataset celeba_hq --num_timesteps 2 --batch_size 12 --r1_gamma 2.0 --lazy_reg 10 --num_process_per_node 2 --ch_mult 1 1 2 2 4 4 --version bs12 --tau 1e-7 --schedule 800
```

#### CelebA-64 perturbed by FashionMNIST (5%) ####

```
python3 train.py --dataset celeba_hq --image_size 64 --num_channels_dae 96 --num_timesteps 2 --batch_size 72 --num_epoch 800 --r1_gamma 0.02 --lr_d 1e-4 --lr_g 1.6e-4 --lazy_reg 15 --ch_mult 1 2 2 2 4 --perturb_dataset fashion_mnist --perturb_percent 5 --tau 0.0003
```

#### CelebA-64 perturbed by LSUN Church (5%) ####

```
python3 train.py --dataset celeba_hq --image_size 64 --num_channels_dae 96 --num_timesteps 2 --batch_size 72 --num_epoch 800 --r1_gamma 0.02 --lr_d 1e-4 --lr_g 1.6e-4 --lazy_reg 15 --ch_mult 1 2 2 2 4 --perturb_dataset lsun --perturb_percent 5 --tau 0.0003
```

Meaning of hyperparameters:

`--dataset`: Name of the clean dataset

`--batch_size`: Batch size at each training iteration

`--image_size`: Size of images

`--num_channels_dae`: Number of initial channels in denosing model

`--ch_mult`: Channel multiplier per scale

`--num_epoch` #: Number of epochs for training

`--r1_gamma` #: Coefficient for R1 regularization

`--lr_d` #: Learning rate for potential (discriminator) network

`--lr_g` #: Learning rate for generator network

`--lazy_reg` #: Number of training iterations for each regularization

`--num_process_per_node` #: Number of GPUs

`--version`: Training version (name of experiment)

`--tau` #: Proportion of the cost c in UOT

`--schedule` #: Number of beginning epochs for cosine scheduler

`--perturb_dataset`: Name of the outlier dataset

`--perturb_percent`: Percentage of perturbed training samples

`--num_timesteps`: Number of timesteps to generate samples

Note: Remove `--perturb_dataset` and `--perturb_percent` for a clean training dataset. Hyperparameters with # are only needed in training phase.


## Evaluation ##
After training, samples can be generated by calling ```test.py```. 

You can also download pretrained checkpoints from this [link](https://drive.google.com/drive/folders/1e7FyELPlqnoHJPpehvDv9Mi6nta-78n4?usp=sharing). Each checkpoint should be placed in the following directory:

```
saved_info/rduot/<dataset>_<perturb_percent>p_<perturb_dataset>/<version>/
```

Remove ```_<perturb_percent>p_<perturb_dataset>``` if training dataset does not contain any outliers. For example, directories containing checkpoints for these experiments are:

CIFAR-10 perturbed by MNIST (3%):
```
saved_info/rduot/cifar10_3p_mnist/bs256/
```

STL-10 (without perturbed dataset):
```
saved_info/rduot/stl10/bs72/
```

We use `--epoch_start` (first epoch), `--epoch_end` (last epoch), `--epoch_jump` (number of epochs before the next evaluation) to specify the checkpoint saved at a particular epoch; and `--compute_fid` to calculate FID score. All other hyperparameters have the same meaning as in the training phase.

We use the following commands to train our proposed model.

#### CIFAR-10 perturbed by MNIST (3%) ####

```
python3 test.py --dataset cifar10 --ch_mult 1 2 2 2 --num_timesteps 4 --num_channels_dae 128 --version bs256 --compute_fid --epoch_start 1200 --epoch_end 1800 --epoch_jump 25 --perturb_dataset mnist --perturb_percent 3
```

#### STL-10 #### 

```
python3 test.py --dataset stl10 --image_size 64 --ch_mult 1 2 2 2 --num_timesteps 4 --num_channels_dae 128 --version bs72 --compute_fid --epoch_start 1200 --epoch_end 1800 --epoch_jump 25
```

#### CelebA HQ 256 ####

```
python3 test.py --dataset celeba_hq --image_size 256 --ch_mult 1 1 2 2 4 4 --num_timesteps 2 --version bs12 --compute_fid --epoch_start 500 --epoch_end 800 --epoch_jump 25
```

#### CelebA-64 perturbed by FashionMNIST (5%) ####

```
python3 test.py --dataset celeba_hq --image_size 64 --num_channels_dae 96 --ch_mult 1 2 2 2 4 --num_timesteps 2 --version bs12 --compute_fid --epoch_start 500 --epoch_end 800 --epoch_jump 25 --perturb_dataset fashion_mnist --perturb_percent 5
```

#### CelebA-64 perturbed by LSUN Church (5%) ####

```
python3 test.py --dataset celeba_hq --image_size 64 --num_channels_dae 96 --ch_mult 1 2 2 2 4 --num_timesteps 2 --version bs12 --compute_fid --epoch_start 500 --epoch_end 800 --epoch_jump 25 --perturb_dataset lsun --perturb_percent 5
```

We use the [PyTorch](https://github.com/mseitzer/pytorch-fid) implementation to compute the FID scores, and in particular, codes for computing the FID are adapted from [FastDPM](https://github.com/FengNiMa/FastDPM_pytorch).

For Improved Precision and Recall, follow the instructions [here](https://github.com/kynkaat/improved-precision-and-recall-metric).

## Contacts
If you have any problems, please open an issue in this repository or send an email to [kevinquandao10@gmail.com](mailto:kevinquandao10@gmail.com) or [tahuubinh2001@gmail.com](mailto:tahuubinh2001@gmail.com).
