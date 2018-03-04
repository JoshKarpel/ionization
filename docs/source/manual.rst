Manual
======

.. currentmodule:: ionization

The ``ionization`` library implements a variety of functionality, all with the goal of simulating the interaction between electric fields and electrons.
It builds off of ``simulacra``, making heavy use of the ``Specification``/``Simulation`` architecture.

The different approaches are grouped into submodules:

* ``mesh`` - mesh-based time-dependent Schrödinger equation (TDSE) simulations.
* ``ide`` - an ionization model based on the TDSE.
* ``tunneling`` - various tunneling ionization models.

These approaches are supported by the other submodules, which provide abstractions of quantum states, electrical potentials, and other things.

* ``states`` - quantum states.
* ``potentials`` - electrical potentials, wavefunction masks, imaginary potentials, and similar.


Mesh-based TDSE Simulations
---------------------------

Mesh-based TDSE simulations and implemented in the ``mesh`` submodule.
``mesh`` has a large number of submodules, intended to break up the complexity of the mesh-based TDSE simulations.

* ``sims`` - ``Simulation`` and ``Specification`` classes for mesh-based TDSE simulations.
* ``evolution_methods`` - implementations of evolution algorithms: Alternating Direction Implicit, split-operator methods, and spectral evolution.
* ``mesh_operators`` - objects that meshes use to get their sparse matrix operators.
* ``data`` - time-indexed data storage of electric field properties, wavefunction norm and inner products, and operator expectation values.
* ``anim`` - tools for animating wavefunction evolution.

Note some quirks of datastores: they're passed to the spec, but the real datastores are owned by the sim.
The datastores given to the spec are "prototypes", which are deep-copied by the sim during startup before they're initialized.

IDE Simulations
---------------

Simulations that use our IDE model and implemented in the ``ide`` submodule.

* ``ide`` - ``Simulation`` and ``Specification`` classes for IDE simulations.
* ``evolution_methods`` - implementations of various evolution algorithms: Forward Euler, Trapezoid Method, Fourth-order Runge-Kutta, etc.
* ``kernels`` - various implementations of the kernel in the IDE model.

Tunneling Simulations
---------------------

Tunneling models are implemented in the ``tunneling`` submodule.

* ``tunneling`` - ``Simulation`` and ``Specification`` classes for tunneling simulations.
* ``models`` - implementations of various tunneling models. These can also be used with IDE simulations.
