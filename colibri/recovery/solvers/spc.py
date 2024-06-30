import torch

from colibri.optics import SPC
from colibri.optics.functional import forward_spc
from .core import Solver


class SPCSolver(Solver):
    r"""
        Solver for the SPC acquisition model.

        It describes the close-form solution of the optimization problem.

        .. math::
            
                \min_{\textbf{X}} \frac{1}{2}||\textbf{X} - \textbf{H}\textbf{X}||_2^2 + \rho||\textbf{X} - \tilde{\textbf{X}}||_2^2

        where :math:`\textbf{X}` is the tensor to be recovered, :math:`\textbf{Y}` is the input tensor, 
        :math:`\textbf{H}` is the sensing matrix, and :math:`\rho` is the regularization parameter.

        in the case of the SPC acquisition model, the :math:`\textbf{X}` is a matrix of size :math:`(h*w, c)`,
        where :math:`h` and :math:`w` are the height and width of the image, and :math:`c` is the number of channels.

        In this sense, :math:`\textbf{X}` is the spatial vectorized form of the image. 
        (since the SPC its broadcasting the sensing matrix over the channels)

        The solution of the optimization problem is given by:

        .. math::
                
                    \hat{\textbf{X}} = (\textbf{H}^\top\textbf{H} + \rho \textbf{I})^{-1}(\textbf{H}^\top \textbf{Y} + \rho \tilde{\textbf{X}})

    """

    def __init__(self, y, acquisition_model: SPC):


        super(SPCSolver, self).__init__(y, acquisition_model)


        self.Hty = acquisition_model(y, type_calculation="backward")
        self.H   = acquisition_model.learnable_optics

    def solve(self, xtilde, rho):

        Hadj = torch.matmul(self.H.permute(1, 0), self.H) + rho * torch.eye(self.H.shape[1])
        Hadj = torch.inverse(Hadj)

        b, c, h, w = xtilde.size()
        x_hat = forward_spc( self.Hty + rho * xtilde, Hadj)
        x_hat = x_hat.permute(0, 2, 1)
        x_hat = x_hat.view(b, c, h, w)

        return x_hat