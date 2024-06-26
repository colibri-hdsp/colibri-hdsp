""" Autoencoder Architecture """

from . import custom_layers
import torch.nn as nn


class Autoencoder(nn.Module):
    """
    Autoencoder Model

    Adapted from 

    Goodfellow, Ian, Yoshua Bengio, and Aaron Courville. Deep learning. MIT press, 2016.
    
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        features=[32, 64, 128, 256],
        last_activation="sigmoid",
        reduce_spatial=False,
        **kwargs,
    ):
        """

        Args:

            in_channels (int): number of input channels
            out_channels (int): number of output channels
            features (list, optional): number of features in each level of the Unet. Defaults to [32, 64, 128, 256].
            last_activation (str, optional): activation function for the last layer. Defaults to 'sigmoid'.
            reduce_spatial (bool): select if the autoencder reduce spatial dimension


        Returns:
            torch.nn.Module: Autoencoder model

        """
        super(Autoencoder, self).__init__()

        levels = len(features)

        self.inc = custom_layers.convBlock(in_channels, features[0], mode="CBRCBR")
        if reduce_spatial:
            self.downs = nn.ModuleList(
                [
                    custom_layers.downBlock(features[i], features[i + 1])
                    for i in range(len(features) - 1)
                ]
            )

            self.ups = nn.ModuleList(
                [
                    custom_layers.upBlockNoSkip(features[i + 1], features[i])
                    for i in range(len(features) - 2, 0, -1)
                ]
                + [custom_layers.upBlockNoSkip(features[1], features[0])]
            )
            # self.ups.append(custom_layers.upBlockNoSkip(features[0]))
            self.bottle = custom_layers.convBlock(features[-1], features[-1])

        else:
            self.downs = nn.ModuleList(
                [
                    custom_layers.convBlock(features[i], features[i + 1], mode="CBRCBR")
                    for i in range(levels - 1)
                ]
            )

            self.bottle = custom_layers.convBlock(features[-1], features[-1])

            self.ups = nn.ModuleList(
                [
                    custom_layers.convBlock(features[i + 1], features[i])
                    for i in range(len(features) - 2, 0, -1)
                ]
                + [custom_layers.convBlock(features[1], features[0])]
            )

        self.outc = custom_layers.outBlock(features[0], out_channels, last_activation)

    def forward(self, inputs, get_latent=False, **kwargs):
        """
        Forward pass of the autoencoder

        Args:
            inputs (torch.Tensor): input tensor
            get_latent (bool): if True, return the latent space
        
        Returns:
            torch.Tensor: output tensor
        """

        x = self.inc(inputs)

        for down in self.downs:
            x = down(x)

        xl = self.bottle(x)
        x = xl

        for up in self.ups:
            x = up(x)

        if get_latent:
            return self.outc(x), xl
        else:
            return self.outc(x)
