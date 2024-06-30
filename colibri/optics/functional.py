import torch
import numpy as np

def prism_operator(x, shift_sign = 1):
    r"""

    Prism operator, shifts linearly the input tensor x in the spectral dimension.

    Args:
        x (torch.Tensor): Input tensor with shape (B, L, M, N)
        shift_sign (int): Integer, it can be 1 or -1, it indicates the direction of the shift
            if 1 the shift is to the right, if -1 the shift is to the left
    Returns:
        torch.Tensor: Output tensor with shape (1, L, M, N + L - 1) if shift_sign is 1, or (1, L M, N-L+1) if shift_sign is -1

    """

    assert shift_sign == 1 or shift_sign == -1, "The shift sign must be 1 or -1"
    _, L, M, N = x.shape  # Extract spectral image shape

    x = torch.unbind(x, dim=1)


    if shift_sign == 1:
        # Shifting produced by the prism 
        x = [torch.nn.functional.pad(x[l], (l, L - l - 1)) for l in range(L)]
    else:
        # Unshifting produced by the prism
        x = [x[l][:, :, l:N - (L- 1)+l] for l in range(L)]

    x = torch.stack(x, dim=1)
    return x

def forward_color_cassi(x, ca):

    r"""

    Forward operator of color coded aperture snapshot spectral imager (Color-CASSI)

    For more information refer to: Colored Coded Aperture Design by Concentration of Measure in Compressive Spectral Imaging https://doi.org/10.1109/TIP.2014.2310125

    Args:
        x (torch.Tensor): Spectral image with shape (B, L, M, N)
        ca (torch.Tensor): Coded aperture with shape (1, L, M, N)
    
    Returns: 
        torch.Tensor: Measurement with shape (B, 1, M, N + L - 1)
    """
    y = torch.multiply(x, ca)
    y = prism_operator(y, shift_sign = 1)
    return y.sum(dim=1, keepdim=True)

def backward_color_cassi(y, ca):
    r"""

    Backward operator of color coded aperture snapshot spectral imager (Color-CASSI)
    
    For more information refer to: Colored Coded Aperture Design by Concentration of Measure in Compressive Spectral Imaging https://doi.org/10.1109/TIP.2014.2310125

    Args:
        y (torch.Tensor): Measurement with shape (B, 1, M, N + L - 1)
        ca (torch.Tensor): Coded aperture with shape (1, L, M, N)
    Returns:
        torch.Tensor: Spectral image with shape (B, L, M, N)
    """
    y = torch.tile(y, [1, ca.shape[1], 1, 1])
    y = prism_operator(y, shift_sign = -1)
    x = torch.multiply(y, ca)
    return x


def forward_dd_cassi(x, ca):
    r"""

    Forward operator of dual disperser coded aperture snapshot spectral imager (DD-CASSI)
    
    For more information refer to: Single-shot compressive spectral imaging with a dual-disperser architecture https://doi.org/10.1364/OE.15.014013

    Args:
        x (torch.Tensor): Spectral image with shape (B, L, M, N)
        ca (torch.Tensor): Coded aperture with shape (1, 1, M, N + L - 1)
    Returns:
        torch.Tensor: Measurement with shape (B, 1, M, N)
    """
    _, L, M, N = x.shape  # Extract spectral image shape
    assert ca.shape[-1] == N + L - 1, "The coded aperture must have the same size as a dispersed scene"
    ca = torch.tile(ca, [1, L, 1, 1])
    ca = prism_operator(ca, shift_sign = -1)
    y = torch.multiply(x, ca)
    return y.sum(dim=1, keepdim=True)


def backward_dd_cassi(y, ca):
    r"""

    Backward operator of dual disperser coded aperture snapshot spectral imager (DD-CASSI)
    
    For more information refer to: Single-shot compressive spectral imaging with a dual-disperser architecture https://doi.org/10.1364/OE.15.014013

    Args:
        y (torch.Tensor): Measurement with shape (B, 1, M, N)
        ca (torch.Tensor): Coded aperture with shape (1, 1, M, N + L - 1)
    Returns:
        torch.Tensor: Spectral image with shape (1, L, M, N)
    """
    _, _, M, N_hat = ca.shape  # Extract spectral image shape
    L = N_hat - M + 1  # Number of shifts
    y = torch.tile(y, [1, L, 1, 1])
    ca = torch.tile(ca, [1, L, 1, 1])
    ca = prism_operator(ca, shift_sign = -1)
    return torch.multiply(y, ca)

def forward_sd_cassi(x, ca):
    r"""
    Forward operator of single disperser coded aperture snapshot spectral imager (SD-CASSI)
    
    For more information refer to: Compressive Coded Aperture Spectral Imaging: An Introduction: https://doi.org/10.1109/MSP.2013.2278763

    Args:
        x (torch.Tensor): Spectral image with shape (B, L, M, N)
        ca (torch.Tensor): Coded aperture with shape (1, 1, M, N)
    Returns:
        torch.Tensor: Measurement with shape (B, 1, M, N + L - 1)
    """
    y1 = torch.multiply(x, ca)  # Multiplication of the scene by the coded aperture
    _, M, N, L = y1.shape  # Extract spectral image shape
    # shift and sum
    y2 = prism_operator(y1, shift_sign = 1)
    return y2.sum(dim=1, keepdim=True)


def backward_sd_cassi(y, ca):
    r"""

    Backward operator of single disperser coded aperture snapshot spectral imager (SD-CASSI)
    
    For more information refer to: Compressive Coded Aperture Spectral Imaging: An Introduction: https://doi.org/10.1109/MSP.2013.2278763

    Args:
        y (torch.Tensor): Measurement with shape (B, 1, M, N + L - 1)
        ca (torch.Tensor): Coded aperture with shape (1, 1, M, N)
    Returns:
        torch.Tensor: Spectral image with shape (B, L, M, N)
    """
    _, _, M, N = y.shape  # Extract spectral image shape
    L = N - M + 1  # Number of shifts
    y = torch.tile(y, [1, L, 1, 1])
    y = prism_operator(y, shift_sign = -1)
    return torch.multiply(y, ca)


def forward_spc(x, H):
    r"""

    Forward propagation through the Single Pixel Camera (SPC) model.

    For more information refer to: Optimized Sensing Matrix for Single Pixel Multi-Resolution Compressive Spectral Imaging 10.1109/TIP.2020.2971150

    Args:
        x (torch.Tensor): Input image tensor of size (B, L, M, N).
        H (torch.Tensor): Measurement matrix of size (S, M*N).

    Returns:
        torch.Tensor: Output measurement tensor of size (B, S, L).
    """
    B, L, M, N = x.size()
    x = x.contiguous().view(B, L, M*N)
    x = x.permute(0, 2, 1)

    # measurement
    H = H.unsqueeze(0).repeat(B, 1, 1)
    y = torch.bmm(H, x)
    return y

def backward_spc(y, H, pinv=False):
    r"""

    Inverse operation to reconstruct the image from measurements.

    For more information refer to: Optimized Sensing Matrix for Single Pixel Multi-Resolution Compressive Spectral Imaging  10.1109/TIP.2020.2971150

    Args:
        y (torch.Tensor): Measurement tensor of size (B, S, L).
        H (torch.Tensor): Measurement matrix of size (S, M*N).
    Returns:
        torch.Tensor: Reconstructed image tensor of size (B, L, M, N).
    """

    Hinv   = torch.pinverse(H) if pinv else torch.transpose(H, 0, 1)
    Hinv   = Hinv.unsqueeze(0).repeat(y.shape[0], 1, 1)

    x = torch.bmm(Hinv, y)
    x = x.permute(0, 2, 1)
    b, c, hw = x.size()
    h = int(np.sqrt(hw))
    x = x.reshape(b, c, h, h)
    return x