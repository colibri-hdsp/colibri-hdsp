import torch
from torch import nn

from .solvers import SOLVERS, get_solver


class PnP_ADMM(nn.Module):
    r"""
    Plug-and-Play (PnP) algorithm for solving the optimization problem based on the Alternating Direction Method of Multipliers (ADMM) formulation.

    .. math::
        \begin{equation}
            \underset{\mathbf{x}}{\text{min}} \quad \frac{1}{2}||\mathbf{y} - \forwardLinear (\mathbf{x})||^2 + \lambda||\mathbf{x}||_1
        \end{equation}
    """

    def __init__(self, fidelity, prior, aquisition_model, solver="close", max_iters=20, _lambda=0.1, rho=0.1, alpha=0.01):

        super(PnP_ADMM, self).__init__()

        self.fidelity         = fidelity
        self.aquisition_model = aquisition_model
        self.prior            = prior
        self.solver           = solver

        self.max_iters        = max_iters
        self._lambda          = _lambda
        self.rho              = rho
        self.alpha            = alpha


    def forward(self, y, x0=None, verbose=False):

        # Initialize the solution
        if self.solver == "close":
            ClosedSolution = get_solver(self.aquisition_model)
            x_solver = ClosedSolution(y, self.aquisition_model)

        else:
            GradientDescent = lambda x, xt: x - self.alpha * self.fidelity.grad(x, y, self.H)  - self.rho * (x - xt)
            x_solver = GradientDescent

        if x0 is None:
            x0 = torch.zeros_like(y)
        
        u_t = torch.zeros_like(x0)
        v_t = x0

        for i in range(self.max_iters):

            # x-subproblem update
            xtilde = v_t - u_t
            x_t    = x_solver(xtilde, self.rho) if self.solver == "close" else x_solver(x_t, xtilde)

            # v-subproblem update
            vtilde = x_t + u_t
            v_t    = self.prior.prox(vtilde, self._lambda)

            # u-subproblem update
            u_t = u_t + x_t - v_t

            if verbose:
                error = self.fidelity.forward(x_t, y, self.H).item()
                print("Iter: ", i, "fidelity: ", error)

        x_hat = v_t

        return x_hat


