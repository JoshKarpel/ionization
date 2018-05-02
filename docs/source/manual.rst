Manual
======

.. currentmodule:: ionization

.. toctree::
   :hidden:
   :maxdepth: 2

   manual/mesh
   manual/ide
   manual/tunneling
   manual/states
   manual/potentials

The ``ionization`` library implements a variety of functionality, all with the goal of simulating the interaction between electric fields and electrons.
It builds off of ``simulacra``, making heavy use of the ``Specification``/``Simulation`` architecture.

It also makes consistent use of the ``Info`` pattern from ``ionization``.
Most objects have a ``.info()`` method that displays information about that object and any objects that it "contains".
For example, printing the ``.info()`` of a :class:`MeshSimulation` will print information about that simulation and the specification it was made from (and therefore the ``.info()`` of the specification, and on and on).

The different approaches to thinking about the interaction are grouped into submodules:

* ``mesh`` - mesh-based time-dependent Schrödinger equation (TDSE) simulations.
* ``ide`` - an ionization model based on the TDSE.
* ``tunneling`` - various tunneling ionization models.

These approaches are supported by the other submodules, which provide abstractions of quantum states, electrical potentials, and other things.

* ``states`` - quantum states.
* ``potentials`` - electrical potentials, wavefunction masks, imaginary potentials, and similar.


