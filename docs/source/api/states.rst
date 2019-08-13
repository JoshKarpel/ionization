Quantum States
==============

.. currentmodule:: ionization.states

Quantum states are represented by objects.
A single eigenstate is a :class:`QuantumState`, and a superposition of those
states is a :class:`Superposition`.

:class:`QuantumState` can be added together, and they can be multiplied by real
and complex numbers.
However, we do not support product states or continuous superpositions, so all
of our states represent discrete superpositions of the state of a single particle.


.. autoclass:: QuantumState

    .. automethod:: normalized

    .. automethod:: info

.. autoclass:: Superposition


One-Dimensional States
----------------------

.. autoclass:: OneDPlaneWave

    .. automethod:: from_energy

.. autoclass:: QHOState

    .. automethod:: from_omega_and_mass

    .. automethod:: from_potential

.. autoclass:: FiniteSquareWellState

    .. automethod:: from_potential

    .. automethod:: all_states_of_well_from_parameters

    .. automethod:: all_states_of_well_from_well

.. autoclass:: GaussianWellState

    .. automethod:: from_potential

.. autoclass:: OneDSoftCoulombState

    .. automethod:: from_potential

.. autoclass:: NumericOneDState

Three-Dimensional States
------------------------

.. autoclass:: ThreeDPlaneWave

.. autoclass:: SphericalHarmonicState

.. autoclass:: FreeSphericalWave

    .. automethod:: from_wavenumber

    .. automethod:: radial_function

.. autoclass:: HydrogenBoundState

    .. automethod:: radial_function

.. autoclass:: HydrogenCoulombState

    .. automethod:: radial_function

.. autoclass:: NumericSphericalHarmonicState

    .. automethod:: radial_function

Helper Functions
----------------

.. autofunction:: fmt_amplitude

.. autofunction:: fmt_amplitude_for_tex

.. autofunction:: fmt_inner_product_for_tex
