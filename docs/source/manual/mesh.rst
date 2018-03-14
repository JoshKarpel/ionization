Mesh-Based TDSE Simulations
===========================

.. currentmodule:: ionization.mesh

Mesh-based TDSE simulations and related features are implemented in the ``mesh`` submodule.
``mesh`` has a large number of submodules, intended to break up the complexity of the mesh-based TDSE simulations.

* ``sims`` - ``Simulation`` and ``Specification`` classes for mesh-based TDSE simulations.
* ``evolution_methods`` - implementations of evolution algorithms: Alternating Direction Implicit, split-operator methods, and spectral evolution.
* ``mesh_operators`` - objects that meshes use to get their sparse matrix operators.
* ``data`` - time-indexed data storage of electric field properties, wavefunction norm and inner products, and operator expectation values.
* ``anim`` - tools for animating wavefunction evolution.
* ``sim_plotters`` - plotting code for mesh-based simulations.
* ``mesh_plotters`` - plotting code for meshes.

This module as a whole makes heavy use of the `strategy pattern <https://en.wikipedia.org/wiki/Strategy_pattern>`_.
Each :class:`QuantumMesh` references several strategy objects, which are instances of concrete subclasses of various classes:

- :class:`MeshOperators` generates the sparse matrix operators that act on the mesh
- :class:`EvolutionMethod` defines how the wavefunction will be evolved in time
- :class:`MeshPlotter` that the mesh defers plotting instructions to.

The simulation (a :class:`MeshSimulation`) also defers to a :class:`MeshSimulationPlotter` for plotting.

Storage of time-indexed data is handled by various :class:`Datastore` objects.
Note some quirks of datastores: they're passed to the spec as instances, but the real datastores are clones of those instances owned by the sim.
Trying to access a simulation's datastores, either through itself or its specification, is not wise.
Instead, all access to a simulation's time-indexed data should go through it's ``data`` attribute, which is explained in detail at :class:`Data`.
