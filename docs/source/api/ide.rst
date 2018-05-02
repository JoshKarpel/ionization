IDE Model Simulations
=====================

.. currentmodule:: ionization.ide

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

.. autoclass:: ionization.ide.delta_kicks.DeltaKicks

    .. automethod:: info

.. autofunction:: ionization.ide.delta_kicks.decompose_potential_into_kicks

.. autoclass:: ionization.ide.delta_kicks.DeltaKickSpecification

    .. automethod:: info

.. autoclass:: ionization.ide.delta_kicks.DeltaKickSimulation

    .. automethod:: run

    .. automethod:: info
