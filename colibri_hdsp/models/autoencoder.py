""" Autoencoder Architecture """
from . import custom_layers 
import tensorflow as tf


class Autoencoder(tf.keras.layers.Layer):
    """
    Autoencoder layer
    """

    def __init__(self, 
                 out_channels, 
                 features=[32, 64, 128, 256],
                 last_activation='sigmoid',
                 reduce_spatial = False):
        """ Autoencoder Layer

            Args:
                out_channels (int): number of output channels
                features (list, optional): number of features in each level of the Unet. Defaults to [32, 64, 128, 256].
                last_activation (str, optional): activation function for the last layer. Defaults to 'sigmoid'.
                reduce_spatial (bool): select if the autoencder reduce spatial dimension
                
            
            Returns:
                tf.keras.Layer: Autoencoder model
                
        """    
        super(Autoencoder, self).__init__()

        levels = len(features)

        self.inc = custom_layers.convBlock(features[0], mode='CBRCBR') 
        if reduce_spatial:
            self.downs = [
                custom_layers.downBlock(features[i+1]) for i in range(levels-2)
            ]
            self.ups = [
                custom_layers.upBlockNoSkip(features[i] // 2) for i in range(levels-2, 0, -1)
            ]
            self.bottle = custom_layers.downBlock(features[-1] // 2)
            self.ups.append(custom_layers.upBlockNoSkip(features[0] // 2))
        else:
            self.downs = [
                custom_layers.convBlock(features[i+1], mode='CBRCBR') for i in range(levels-2)
            ]

            self.bottle = custom_layers.convBlock(features[-1], mode='CBRCBR')

            self.ups = [
                custom_layers.convBlock(features[i] // 2, mode='CBRCBR') for i in range(levels-2, 0, -1)
            ]
            self.ups.append(custom_layers.convBlock(features[0],mode='CBCBR'))

            
        self.outc = custom_layers.outBlock(out_channels, last_activation)

    def call(self, inputs, get_latent=False,**kwargs):

        x = self.inc(inputs)
        
        for down in self.downs:
            x = down(x)

        xl = self.bottle(x)
        x = xl
        for up in self.ups:
            x = up(x)
        if get_latent:
            return self.outc(x),xl
        else:
            return self.outc(x)