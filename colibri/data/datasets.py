import matplotlib.pyplot as plt
import torch

from torch.utils import data
from torch.utils.data import Dataset
from torchvision import transforms

import colibri.data.utils as D


DATASET_READER = {
    'cave': D.load_cave_sample,
    'arad': D.load_arad_sample,
}


class DefaultTransform:
    def __init__(self, name):
        self.transform_dict = dict(input=transforms.ToTensor(), default=transforms.ToTensor())
        if name in D.BUILTIN_DATASETS:
            self.transform_dict['output'] = transforms.Lambda(lambda x: x)
        else:
            self.transform_dict['output'] = transforms.ToTensor()

    def __call__(self, key, value):
        if key in self.transform_dict:
            return self.transform_dict[key](value)
        else:
            return self.default_transform(value)

    def default_transform(self, data):
        return self.transform_dict['default'](data)


class CustomDataset(Dataset):
    """Custom dataset."""

    def __init__(self, name, path, builtin_dict=None, transform_dict=None):
        """
        Arguments:
            name (string): Name of the dataset.
            path (string): Path to directory with the dataset.
            builtin_dict (dict): Dictionary with the parameters to load the builtin dataset.
            transform_dict (dict,object): Dictionary with the transformations to apply to the data.
        """
        if transform_dict is None:
            transform_dict = {}
        else:
            assert 'input' in transform_dict, "'input' key must be provided in transform_dict"

        self.is_builtin_dataset = False
        if name in D.BUILTIN_DATASETS:  # builtin datasets
            assert builtin_dict is not None, "builtin_dict must be provided for builtin datasets"
            self.is_builtin_dataset = True
            path = D.update_builtin_path(name, path)
            self.dataset = D.load_builtin_dataset(name, path, **builtin_dict)
            self.len_dataset = len(self.dataset['input'])

        else:  # custom datasets
            self.dataset_filenames = D.get_filenames(name, path)
            self.data_reader = DATASET_READER[name]
            self.len_dataset = len(self.dataset_filenames)

        self.transform_dict = transform_dict
        self.default_transform = DefaultTransform(name)

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):

        # load sample

        if self.is_builtin_dataset:
            data = {key: value[idx] for key, value in self.dataset.items()}
        else:
            data = self.data_reader(self.dataset_filenames[idx])

        # apply transformation

        for key, value in data.items():
            if not isinstance(value, torch.Tensor):
                if key in self.transform_dict:
                    data[key] = self.transform_dict[key](value)
                else:
                    data[key] = self.default_transform(key, value)

        return data


if __name__ == '__main__':
    name = 'cave'
    path = 'path_to_cave_dataset'

    builtin_dict = dict(train=True, download=True)
    dataset = CustomDataset(name, path,
                            builtin_dict=builtin_dict,
                            transform_dict=None)

    dataset_loader = data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    # plot 3 x 3 images

    data = next(iter(dataset_loader))
    image = data['input']
    label = data['output']

    plt.figure(figsize=(5, 5))
    plt.suptitle(f'{name.upper()} dataset Samples')

    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(image[i].permute(1, 2, 0).cpu().numpy())
        plt.axis('off')

    plt.tight_layout()
    plt.show()
