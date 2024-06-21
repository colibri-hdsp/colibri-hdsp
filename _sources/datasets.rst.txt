Datasets
========

The datasets module provides a streamlined approach to accessing and utilizing a wide array of datasets for computational imaging tasks. Central to this module is the `Dataset` class, an abstraction designed to facilitate easy selection and handling of datasets from a predefined collection.

Dataset Class
-------------

dataset
~~~~~~~
The `Dataset` class serves as a flexible interface for accessing datasets listed in the `BASIC_DATASETS` dictionary. Each key in this dictionary corresponds to a specific dataset, allowing users to easily select and load datasets for their computational imaging projects.

.. autoclass:: colibri.data.datasets.Dataset
    :members:

Using the `Dataset` class, researchers and developers can swiftly navigate through the available datasets, simplifying the process of dataset selection and loading. This abstraction layer ensures that users can focus on their computational imaging tasks without worrying about the underlying data handling intricacies.
