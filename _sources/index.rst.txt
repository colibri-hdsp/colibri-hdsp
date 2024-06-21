Welcome to Colibri
==================

Colibri is a PyTorch library in development for solving computational imaging tasks where optical systems and
state-of-the-art deep neural networks are implemented to be easily used or modified for new research ideas. The purpose
of Colobri is to boost the research-related areas where optics and networks are required and introduce new researchers
to state-of-the-art algorithms in a straightforward and friendly manner.

💿 Installation
------------------

To get started with Colibri, install the library using the following steps:

1. Clone the repository:

.. code-block:: bash

    git clone https://github.com/pycolibri/pycolibri.git

2. Create a virtual environment with conda:

.. code-block:: bash

    conda create -n colibri python=3.10
    conda activate colibri

3. Install the requirements:

.. code-block:: bash

    pip install -r requirements.txt

4. Enjoy! 😄

🚀 Quick Start
------------------

You can go to ``examples`` folder and run cells of the notebook ``demo_colibri.ipynb`` to see how the library works.

💡 **Examples**

**Dataset Visualization**

.. code-block:: python

    import matplotlib.pyplot as plt
    from colibri.data.datasets import Dataset
    from torchvision.utils import make_grid

    # Load dataset

    dataset_path = 'cifar10'
    keys = ''
    batch_size = 128

    dataset = Dataset(dataset_path, keys, batch_size)
    adquistion_name = 'cassi' #  ['spc', 'cassi']

    # get samples

    sample = next(iter(dataset.train_dataset))[0]
    img = make_grid(sample[:32], nrow=8, padding=1, normalize=True, range=None, scale_each=False, pad_value=0)

    # visualize samples

    plt.figure(figsize=(10,10))
    plt.imshow(img.permute(1, 2, 0))
    plt.title('CIFAR10 dataset')
    plt.axis('off')
    plt.show()


🧰 Features
------------------

- Flexible and customizable code.
- Train end-to-end models in an straightforward manner.
- Easily modify the implemented models.
- Easily add new optical systems.
- Easily add new deep neural networks.

Available Models
------------------

📷 **Optical Systems**

- Coded Aperture Snapshot Spectral Imager (CASSI).
    - Single Disperser CASSI `(SD-CASSI) <https://pycolibri.github.io/pycolibri/stubs/colibri.optics.cassi.SD_CASSI.html#colibri.optics.cassi.SD_CASSI>`_
    - Dual Disperser CASSI `(DD-CASSI) <https://pycolibri.github.io/pycolibri/stubs/colibri.optics.cassi.DD_CASSI.html>`_
    - Color CASSI `(C-CASSI) <https://pycolibri.github.io/pycolibri/stubs/colibri.optics.cassi.C_CASSI.html>`_
- Single Pixel Camera `(SPC) <https://pycolibri.github.io/pycolibri/stubs/colibri.optics.spc.SPC.html>`_

💻️ **Deep Neural Networks**

- `Autoencoder <https://pycolibri.github.io/pycolibri/models.html>`_
- `Unet <https://pycolibri.github.io/pycolibri/models.html>`_

🖥 **Recovery Algorithms**

- Fast Iterative Shrinkage-Thresholding Algorithm `(FISTA) <https://pycolibri.github.io/pycolibri/recovery.html>`_

🎆 **Frameworks**

- `End-to-end framework <https://pycolibri.github.io/pycolibri/architectures.html>`_ with optical systems as encoder models and deep neural networks as decoder models.


💡 Contributing
------------------

Contributions are welcome! If you're interested in improving Colibri, please:

1. Fork the repository.
2. Create your feature branch (``git checkout -b feature/AmazingFeature``).
3. Commit your changes (``git commit -am 'Add some AmazingFeature'``).
4. Push to the branch (``git push origin feature/AmazingFeature``).
5. Open a Pull Request.

🛡️ License
------------------

.. code-block:: text

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Contents:

   architectures
   models
   optics
   datasets
   regularizers
   auto_examples/index
   recovery
   contributing