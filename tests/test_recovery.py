import pytest
from torch.utils import data

from .utils import include_colibri

include_colibri()

import torch

# Reconstruct image
from colibri.recovery.fista import Fista
from colibri.recovery.terms.prior import Sparsity
from colibri.recovery.terms.fidelity import L2
from colibri.recovery.transforms import DCT2D


@pytest.fixture
def algo_params():
    return {
        'max_iter': 200,
        'alpha': 1e-4,
        'lambda': 0.001,
        'tol': 1e-3
    }


def load_img():
    from colibri.data.datasets import CustomDataset
    name = 'cifar10'
    path = '.'

    builtin_dict = dict(train=True, download=True)
    dataset = CustomDataset(name, path,
                            builtin_dict=builtin_dict,
                            transform_dict=None)
    sample = dataset[0]['input']
    return sample.unsqueeze(0)


def load_acqusition(img_size):
    from colibri.optics import SPC

    acquisition_config = dict(
        input_shape=img_size,
    )

    n_measurements = 25 ** 2
    acquisition_config['n_measurements'] = n_measurements

    acquisiton_model = SPC(**acquisition_config)
    return acquisiton_model


def test_fista_algorithm(algo_params):
    x_true = load_img()
    img_size = x_true.shape[1:]
    acquisition_model = load_acqusition(img_size)

    transform_dct = DCT2D()
    fidelity = L2()
    prior = Sparsity()

    fista = Fista(fidelity, prior, acquisition_model, algo_params, transform_dct)
    y = acquisition_model(x_true)
    x_trivial = acquisition_model(y, type_calculation="backward")
    x_hat = fista(y, x0=x_trivial)

    # Check if the output has the same shape as the input
    assert x_true.shape == x_hat.shape, f"Shape of the input: {x_true.shape}, Shape of the output: {x_hat.shape}"

    error_trivial = torch.norm(x_true - x_trivial)
    error_algo = torch.norm(x_true - x_hat)

    # Check if the error of the algorithm is smaller than the error of the trivial solution
    assert error_algo < error_trivial, f"Error of the algorithm: {error_algo}, Error of the trivial solution: {error_trivial}"
