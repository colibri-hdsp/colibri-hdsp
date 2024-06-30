import pytest
from .utils import include_colibri
include_colibri()

from colibri.recovery.terms.transforms import DCT2D

def load_img():

    from colibri.data.datasets import Dataset
    dataset_path = 'cifar10'
    keys = ''
    batch_size = 1
    dataset = Dataset(dataset_path, keys, batch_size)
    sample = next(iter(dataset.train_dataset))[0]
    return sample


def test_dct2d():

    x_true = load_img()
    transform_dct = DCT2D()

    theta = transform_dct.forward(x_true)
    x_hat = transform_dct.inverse(theta)

    mse = (x_true - x_hat).pow(2).mean().item()
    assert mse < 1e-6, f"Mean Squared Error: {mse}"

    
        