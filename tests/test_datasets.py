import pytest
from torch.utils import data

from .utils import include_colibri

include_colibri()

from colibri.data.datasets import CustomDataset


@pytest.fixture
def dataset_info():
    name = 'cifar10'
    path = '.'
    batch_size = 16

    builtin_dict = dict(train=True, download=True)
    dataset = CustomDataset(name, path,
                            builtin_dict=builtin_dict,
                            transform_dict=None)

    return batch_size, data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def test_sample_type(dataset_info):
    _, dataset = dataset_info
    sample = next(iter(dataset))
    expected_type = dict

    assert isinstance(sample, expected_type)
    assert 'input' in sample


def test_data_size(dataset_info):
    batch_size, dataset = dataset_info
    sample = next(iter(dataset))
    expected_size = (batch_size, 3, 32, 32)

    assert sample['input'].shape == expected_size