IDE Simulations
===============

Simulations that use our IDE model and implemented in the ``ide`` submodule.

* ``ide`` - ``Simulation`` and ``Specification`` classes for IDE simulations.
* ``evolution_methods`` - implementations of various evolution algorithms: Forward Euler, Trapezoid Method, Fourth-order Runge-Kutta, etc.
* ``kernels`` - various implementations of the kernel in the IDE model.

An :class:`IDESimulation` delegates the decision of how to evolve the wavefunction to an :class:`EvolutionMethod`.
It is also provided with a :class:`Kernel`, which implements some version of the analytic kernel function.

This sub-package also includes architecture for working with delta-kicks.
