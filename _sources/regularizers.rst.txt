Regularizers
============

Various regularization techniques are implemented regarding the design of optical coding elements. These regularizations force certain physical constraints on the learning parameters. Also, there are regularizers that promotes certain behaviour on the coded measurements to imporve the task perfomance.

The regularizers are used in the optimization problem of the form:

.. math::
    \{\learnedOptics^*,\theta^*\} = \arg \min_{\learnedOptics,\theta} \sum_{p=1}^{P}\mathcal{L}(\mathbf{x}_p,\reconnet( \forwardLinear_{\learnedOptics}(\mathbf{x}_p))) + \lambda \mathcal{R}_1(\phi) + \mu \mathcal{R}_2(\forwardLinear_\learnedOptics (\mathbf{x})) 

where :math:`\mathcal{L}` is the task loss function, :math:`\mathcal{R}_1` is a regularizer over the coding elements, and  :math:`\mathcal{R}_2` is a regularizer over the coded measurements :math:`\lambda` and :math:`\mu` are the regularization weights, and :math:`P` is the number of samples in the training dataset.

List of Regularizers
--------------------

.. autosummary::
    :toctree: stubs
    :template: class_template.rst
    :nosignatures:

    colibri.regularizers.Reg_Binary
    colibri.regularizers.Correlation

    colibri.regularizers.Reg_Transmittance

    colibri.regularizers.KLGaussian
    colibri.regularizers.MinVariance

    
