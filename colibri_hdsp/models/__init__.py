from .unet import Unet
from .autoencoder import Autoencoder
from .DIP import DIP_CASSI

# from tensorflow.keras.layers import Input
# from tensorflow.keras.models import Model
import torch
import torch.nn

__all__ = [ 
    "Unet",
    "Autoencoder",
    "DIP_CASSI",
    "build_network"
]

def build_network(model=Unet, **network_params):
    """ Build the network model

    Args:
        model_name (tf.keras.Layer, optional): Model name. Defaults to "unet".
        in_channels (int, optional): Number of input channels. Defaults to 1.
        size (int, optional): Size of the input image. Defaults to None.
        **network_params (dict): Network parameters
    """

    model = model(**network_params)
    return model





