API Reference
=============

.. currentmodule:: ionization

.. toctree::
   :hidden:
   :maxdepth: 2

   api/mesh
   api/ide
   api/tunneling
   api/states
   api/potentials

The different approaches to thinking about the interaction are grouped into submodules:

:doc:`api/mesh`
    Mesh-based time-dependent Schrödinger equation (TDSE) simulations.
:doc:`api/ide`
    An ionization model based on the TDSE.
:doc:`api/tunneling`
    Various tunneling ionization models.

These approaches are supported by the other submodules, which provide
abstractions for quantum states, electrical potentials, etc.

:doc:`api/states`
    Object-based representations of quantum states.
:doc:`api/potentials`
    Electrical potentials, imaginary potentials, wavefunction masks, etc.
