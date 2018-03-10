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


.. toctree::
   :hidden:
   :maxdepth: 2

   manual/mesh
   manual/ide
   manual/tunneling
   manual/states
   manual/potentials
