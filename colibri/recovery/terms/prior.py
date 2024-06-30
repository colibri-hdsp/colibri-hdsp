import torch

from .transforms import DCT2D


class Sparsity(torch.nn.Module):
    '''
        Sparsity prior 
        
        .. math::
        
            g(\mathbf{x}) = \|\Phi x\|_1
        
            where :math:`\Phi` is the sparsity basis and :math:`x` is the input tensor.

    '''
    def __init__(self, basis=None):
        '''
        Args:
            basis (str): Basis function. 'dct', 'None'. Default is None.
        '''
        super(Sparsity, self).__init__()

        if basis == 'dct':
            self.transform = DCT2D()
        else:
            self.transform = None

    def forward(self, x):
        '''
        Compute sparsity term.

        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Sparsity term.
        '''
        x = self.transform.forward(x)
        return torch.norm(x, 1)**2
    
    def prox(self, x, _lambda, type="soft"):
        '''
        Compute proximal operator of the sparsity term.

        Args:
            x (torch.Tensor): Input tensor.
            _lambda (float): Regularization parameter.
            type (str): String, it can be "soft" or "hard".
        
        Returns:
            torch.Tensor: Proximal operator of the sparsity term.
        '''
        x = x.requires_grad_()
        x = self.transform.forward(x)

        if type == 'soft':
            x = torch.sign(x)*torch.max(torch.abs(x) - _lambda, torch.zeros_like(x))
        elif type == 'hard':
            x = x*(torch.abs(x) > _lambda)
        
        x = self.transform.inverse(x)
        return x
        
    def transform(self, x):
        
        if self.transform is not None:
            return self.transform.forward(x)
        else:
            return x
    
    def inverse(self, x):
        
        if self.transform is not None:
            return self.transform.inverse(x)
        else:
            return x
        
        
    

    

