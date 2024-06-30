import os

import numpy as np
from PIL import Image

import torchvision

# Builtin datasets

BUILTIN_DATASETS = {
    'mnist': torchvision.datasets.MNIST,
    'fashion_mnist': torchvision.datasets.FashionMNIST,
    'cifar10': torchvision.datasets.CIFAR10,
    'cifar100': torchvision.datasets.CIFAR100
}


def update_builtin_path(name, path):
    path = os.path.join(path, name)
    os.makedirs(path, exist_ok=True)

    return path


def load_builtin_dataset(name, path, **kwargs):
    train = kwargs['train'] if 'train' in kwargs else False
    download = kwargs['download'] if 'download' in kwargs else True

    builtin_dataset = BUILTIN_DATASETS[name](root=path, train=train, download=download)
    dataset = dict(input=builtin_dataset.data, output=builtin_dataset.targets)

    # transform

    dataset['input'] = (dataset['input'] / 255.).astype(np.float32)
    if dataset['input'].ndim != 4:
        dataset['input'] = dataset['input'].unsqueeze(1)

    return dataset


# Custom datasets

def get_cave_filenames(path):
    return [os.path.join(path, name, name) for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]


def get_arad_filenames(path):
    pass


CUSTOM_DATASETS = {
    'cave': get_cave_filenames,
    'arad': get_arad_filenames
}


def get_filenames(name, path):
    return CUSTOM_DATASETS[name](path)


def load_arad_sample(filename, preprocessing, **kwargs):
    return Image.open(filename)


def load_cave_sample(filename):
    name = os.path.basename(filename).replace('_ms', '')

    spectral_image = []
    for i in range(1, 32):
        spectral_band_filename = os.path.join(filename, f'{name}_ms_{i:02d}.png')
        spectral_band = np.array(Image.open(spectral_band_filename).convert('L')) / 255.
        spectral_image.append(spectral_band.astype(np.float32))

    spectral_image = np.stack(spectral_image, axis=-1)
    rgb_image = np.array(Image.open(os.path.join(filename, f'{name}_RGB.bmp'))) / 255.
    rgb_image = rgb_image.astype(np.float32)

    return dict(input=rgb_image, output=spectral_image)
