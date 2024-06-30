import torch
from torch import nn


class Fista(nn.Module):
    r"""
    FISTA algorithm for solving the optimization problem

    .. math::
        \begin{equation}
            \underset{\mathbf{x}}{\text{min}} \quad \frac{1}{2}||\mathbf{y} - \forwardLinear (\mathbf{x})||^2 + \lambda||\mathbf{x}||_1
        \end{equation}

    where :math:`\forwardLinear` is the forward model, :math:`\mathbf{y}` is the data to be reconstructed, :math:`\lambda` is the regularization parameter and :math:`||\cdot||_1` is the L1 norm.

    The FISTA algorithm is an iterative algorithm that solves the optimization problem by performing a gradient step and a proximal step.

    .. math::
        \begin{align*}
         \mathbf{x}_{k+1} &= \text{prox}_{\lambda||\cdot||_1}( \mathbf{z}_k - \alpha \nabla f( \mathbf{z}_k)) \\
        t_{k+1} &= \frac{1 + (1 + 4t_k^2)^{0.5}}{2} \\
        \mathbf{z}_{k+1} &=  \mathbf{x}_{k+1} + \frac{t_k-1}{t_{k+1}}( \mathbf{x}_{k} - \mathbf{x}_{k-1})
        \end{align*}

    where :math:`\alpha` is the step size and :math:`f` is the fidelity term.

    """

    def __init__(self, fidelity, prior, acquistion_model, max_iters=5, alpha=1e-3, _lambda=0.1):
        """Initializes the Fista class.

        Args:

            fidelity (nn.Module): The fidelity term in the optimization problem. This is a function that measures the discrepancy between the data and the model prediction.
            prior (nn.Module): The prior term in the optimization problem. This is a function that encodes prior knowledge about the solution.
            acquistion_model (nn.Module): The acquisition model of the imaging system. This is a function that models the process of data acquisition in the imaging system.
            max_iters (int): The maximum number of iterations for the FISTA algorithm. Defaults to 5.
            alpha (float): The step size for the gradient step. Defaults to 1e-3.
            _lambda (float): The regularization parameter for the prior term. Defaults to 0.1.

        Returns:
            None
        """
        super(Fista, self).__init__()

        self.fidelity = fidelity
        self.acquistion_model = acquistion_model
        self.prior = prior

        self.H = lambda alpha: self.acquistion_model.forward(alpha)

        self.max_iters = max_iters
        self.alpha = alpha
        self._lambda = _lambda


    def forward(self, y, x0=None, verbose=False):
        """Runs the FISTA algorithm to solve the optimization problem.

        Args:
            y (torch.Tensor): The data to be reconstructed.
            x0 (torch.Tensor, optional): The initial guess for the solution. Defaults to None.

        Returns:
            torch.Tensor: The reconstructed image.
        """

        if x0 is None:
            x0 = torch.zeros_like(y)

        x = x0
        t = 1
        z = x.clone()

        for i in range(self.max_iters):
            x_old = x.clone()

            # gradient step
            x = z - self.alpha * self.fidelity.grad(z, y, self.H) 

            # proximal step
            x = self.prior.prox(x, self._lambda)

            # FISTA step
            t_old = t
            t = (1 + (1 + 4 * t_old**2) ** 0.5) / 2
            z = x + ((t_old - 1) / t) * (x - x_old)

            error = self.fidelity.forward(x, y, self.H).item()
            
            if verbose:
                print("Iter: ", i, "fidelity: ", error)

        x_hat = x
        return x_hat
