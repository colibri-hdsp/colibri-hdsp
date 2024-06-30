Recovery
========

The recovery module provides an implementation of state-of-the-art recovery algorithms widely used 
in image restoration on inverse problems.

.. math::

    \min_{x} f(\mathbf{x})+ \lambda g(\mathbf{x})

where :math:`f(\mathbf{x})` is the fidelity term and :math:`g(\mathbf{x})` is the prior term.

List of Algorithms
--------------------

.. autosummary::
    :toctree: stubs
    :template: class_template.rst
    :nosignatures:

    colibri.recovery.fista.Fista
    

List of Solvers
--------------------

.. autosummary::
    :toctree: stubs
    :template: class_template.rst
    :nosignatures:

    colibri.recovery.solvers.core.LinearSolver
    colibri.recovery.solvers.spc.SPCSolver


Fidelity Terms
--------------------
The module contains differents fidelity terms :math:`f(\mathbf{x})` that can be used in the recovery algorithms.


.. autosummary::
    :toctree: stubs
    :template: class_template.rst
    :nosignatures:

    colibri.recovery.terms.fidelity.L2
    colibri.recovery.terms.fidelity.L1



Prior Terms
--------------------
The module contains differents prior terms :math:`g(\mathbf{x})` that can be used in the recovery algorithms.


    
.. autosummary::
    :toctree: stubs
    :template: class_template.rst
    :nosignatures:

    colibri.recovery.terms.prior.Sparsity
    
Transormation
--------------------

The module contains differents signal transforms that can be used in the recovery algorithms.

    
.. autosummary::
    :toctree: stubs
    :template: class_template.rst
    :nosignatures:

    colibri.recovery.terms.transforms.DCT2D