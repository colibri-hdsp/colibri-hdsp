.. _optics:

Optics
======

The PyColibri optics module encompasses a collection of optical systems extensively employed in the computational imaging research for spectral, depth, phase and other imaging applications. This module facilitates the simulation of optical propagation and the design of optical elements inside the systems, offering researchers and developers a powerful toolkit for advancing imaging technologies.


Mathematically, all implemented optical systems are of the form

.. math::

    \mathbf{y} = \forwardLinear_{\learnedOptics}(\mathbf{x}) + \noise

where :math:`\mathbf{x}\in\xset` is the input optical field, :math:`\mathbf{y}\in\yset` are the acquired signal,
:math:`\forwardLinear:\xset\mapsto \yset` is a deterministic (linear or non-linear) optics model of the acquisition system, 
:math:`\learnedOptics` is a set of learnable parameters characterizing the optical system,
and :math:`\noise` is a stochastic mapping which characterizes the noise affecting the measurements.



Spectral Imaging systems
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: stubs
    :template: class_template.rst
    :nosignatures:

    colibri.optics.cassi.SD_CASSI
    colibri.optics.cassi.DD_CASSI
    colibri.optics.cassi.C_CASSI
    colibri.optics.spc.SPC


Functional operators of the optical systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: stubs
    :template: methods_template.rst
    :nosignatures:

    colibri.optics.functional.forward_color_cassi
    colibri.optics.functional.backward_color_cassi
    colibri.optics.functional.forward_dd_cassi
    colibri.optics.functional.backward_dd_cassi
    colibri.optics.functional.forward_sd_cassi
    colibri.optics.functional.backward_sd_cassi
    colibri.optics.functional.forward_spc
    colibri.optics.functional.backward_spc

Functional operators of optical elements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: stubs
    :template: methods_template.rst
    :nosignatures:

    colibri.optics.functional.prism_operator


