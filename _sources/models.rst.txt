Models
======

The models module within our library provides implementations of deep learning models tailored for computational imaging :math:`\reconnet`. These models leverage the latest advancements in neural networks to offer robust solutions for image reconstruction, enhancement, and segmentation.


List of models
--------------------
The models module contains the following models:



.. autosummary::
    :toctree: stubs
    :template: class_template.rst
    :nosignatures:

    colibri.models.autoencoder.Autoencoder
    colibri.models.unet.Unet
    

List of custom layers
-----------------------

The custom_layers module within our library provides implementations of custom layers that are used in the models. These layers are designed to enhance the performance of the models by providing additional flexibility and control over the network architecture.



.. autosummary::
    :toctree: stubs
    :template: class_template.rst
    :nosignatures:

    colibri.models.custom_layers.Activation
    colibri.models.custom_layers.convBlock
    colibri.models.custom_layers.downBlock
    colibri.models.custom_layers.upBlock
    colibri.models.custom_layers.upBlockNoSkip
    colibri.models.custom_layers.outBlock

