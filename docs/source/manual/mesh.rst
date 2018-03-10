Mesh-Based TDSE Simulations
===========================

Mesh-based TDSE simulations and implemented in the ``mesh`` submodule.
``mesh`` has a large number of submodules, intended to break up the complexity of the mesh-based TDSE simulations.

* ``sims`` - ``Simulation`` and ``Specification`` classes for mesh-based TDSE simulations.
* ``evolution_methods`` - implementations of evolution algorithms: Alternating Direction Implicit, split-operator methods, and spectral evolution.
* ``mesh_operators`` - objects that meshes use to get their sparse matrix operators.
* ``data`` - time-indexed data storage of electric field properties, wavefunction norm and inner products, and operator expectation values.
* ``anim`` - tools for animating wavefunction evolution.
* ``sim_plotters`` - plotting code for mesh-based based simulations.
* ``mesh_plotters`` - plotting code for meshes.

Note some quirks of datastores: they're passed to the spec, but the real datastores are owned by the sim.
The datastores given to the spec are "prototypes", which are deep-copied by the sim during startup before they're initialized.
