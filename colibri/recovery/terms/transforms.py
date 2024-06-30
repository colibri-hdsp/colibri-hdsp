import torch_dct as dct


class DCT2D:
    r"""
    2D Discrete Cosine Transform

    The 2D DCT is defined as:

    .. math::
        X(u,v) = \frac{2}{\sqrt{MN}} \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} x(u) y(v) f(x,y) \cos\left(\frac{(2x+1)u\pi}{2M}\right) \cos\left(\frac{(2y+1)v\pi}{2N}\right)

    The 2D DCT is a separable transform, and can be computed as two 1D DCTs along the rows and columns of the image.

    Args:
        norm (str, optional): The normalization to be applied to the transform. Defaults to 'ortho'.

    Returns:
        torch.Tensor: The 2D DCT of the input image.
    """

    def __init__(self, norm='ortho'):
        """Initializes the DCT2D class.

        Args:
            norm (str, optional): The normalization to be applied to the transform. Defaults to 'ortho'.
        """

        self.norm = norm

    def forward(self, x):
        """Computes the 2D DCT of the input image.

        Args:
            x (torch.Tensor): The input image.

        Returns:
            torch.Tensor: The 2D DCT of the input image.
        """
        return dct.dct_2d(x, norm=self.norm)

    def inverse(self, x):
        """Computes the inverse 2D DCT of the input image.

        Args:
            x (torch.Tensor): The input image.

        Returns:
            torch.Tensor: The inverse 2D DCT of the input image.
        """
        return dct.idct_2d(x, norm=self.norm)