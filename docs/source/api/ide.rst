IDE Model Simulations
=====================

.. currentmodule:: ionization.ide

Simulations that use our IDE model and implemented in the ``ide`` submodule.

An :class:`IDESimulation` delegates the decision of how to evolve the wavefunction to an :class:`EvolutionMethod`.
It is also provided with a :class:`Kernel`, which implements some version of the analytic kernel function.

This sub-package also includes tools for working with delta-kicks.


Specifications
--------------

.. autoclass:: IntegroDifferentialEquationSpecification

    .. automethod:: info

Simulations
-----------

.. autoclass:: IntegroDifferentialEquationSimulation

    .. automethod:: run

    .. automethod:: info

Evolution Methods
-----------------

.. autoclass:: EvolutionMethod

    .. automethod:: evolve

.. autoclass:: ForwardEulerMethod

.. autoclass:: BackwardEulerMethod

.. autoclass:: TrapezoidMethod

.. autoclass:: RungeKuttaFourMethod

.. autoclass:: AdaptiveRungeKuttaFourMethod

Kernels
-------

.. autoclass:: Kernel

    .. automethod:: __call__

    .. automethod:: info

.. autoclass:: LengthGaugeHydrogenKernel

    .. automethod:: evaluate_kernel_function

    .. autoattribute:: kernel_function

.. autoclass:: LengthGaugeHydrogenKernelWithContinuumContinuumInteraction


Delta-Kicks
-----------

.. currentmodule:: ionization.ide.delta_kicks

.. autoclass:: DeltaKicks

    .. automethod:: info

.. autofunction:: decompose_potential_into_kicks

.. autoclass:: DeltaKickSpecification

    .. automethod:: info

.. autoclass:: DeltaKickSimulation

    .. automethod:: run

    .. automethod:: info
