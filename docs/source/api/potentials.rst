Potentials
==========

All potentials have the feature that their ``__call__`` methods,
used to evaluate the potential energy as function of various arguments,
absorb extraneous keyword arguments.
This lets you construct a sum of potentials which share or don't share certain
arguments, as long as each argument has a consistent meaning between the
potentials.

.. currentmodule:: ionization.potentials

.. autoclass:: PotentialEnergy

    .. automethod:: info

.. autoclass:: PotentialEnergySum

.. autoclass:: NoPotentialEnergy

Static Potentials
-----------------

.. autoclass:: CoulombPotential

    .. automethod:: __call__

.. autoclass:: SoftCoulombPotential

    .. automethod:: __call__

.. autoclass:: HarmonicOscillator

    .. automethod:: __call__

    .. automethod:: omega

    .. automethod:: frequency

.. autoclass:: FiniteSquareWell

    .. automethod:: __call__

.. autoclass:: GaussianPotential

    .. automethod:: __call__

    .. automethod:: fwhm

Electric Potentials (Pulses)
----------------------------

.. autoclass:: ElectricPotential

.. autoclass:: UniformLinearlyPolarizedElectricPotential

    .. automethod:: __call__

    .. automethod:: get_electric_field_amplitude

    .. automethod:: get_vector_potential_amplitude

    .. automethod:: get_electric_field_integral_numeric

    .. automethod:: get_vector_potential_amplitude_numeric

    .. automethod:: get_electric_field_integral_numeric_cumulative

    .. automethod:: get_vector_potential_amplitude_numeric_cumulative

.. autoclass:: NoElectricPotential

.. autoclass:: SineWave

    .. automethod:: from_frequency

    .. automethod:: from_period

    .. automethod:: from_wavelength

    .. automethod:: from_photon_energy

    .. automethod:: from_photon_energy_and_intensity

    .. automethod:: keldysh_parameter

.. autoclass:: SumOfSinesPulse

.. autoclass:: SincPulse

    .. automethod:: from_omega_min

    .. automethod:: from_omega_carrier

    .. automethod:: from_keldysh_parameter

    .. automethod:: from_amplitude

    .. automethod:: get_electric_field_envelope

    .. automethod:: keldysh_parameter

.. autoclass:: GaussianPulse

    .. automethod:: from_omega_min

    .. automethod:: from_omega_carrier

    .. automethod:: from_keldysh_parameter

    .. automethod:: from_amplitude

    .. automethod:: from_power_exclusion

    .. automethod:: from_number_of_cycles

    .. automethod:: get_electric_field_envelope

    .. automethod:: keldysh_parameter

.. autoclass:: SechPulse

    .. automethod:: get_electric_field_envelope

    .. automethod:: keldysh_parameter

.. autoclass:: CosSquaredPulse

    .. automethod:: get_electric_field_envelope

.. autofunction:: DC_correct_electric_potential


Windows
-------

.. autoclass:: TimeWindow

    .. automethod:: __call__

    .. automethod:: info

.. autoclass:: TimeWindowSum

.. autoclass:: NoTimeWindow

    .. automethod:: __call__

.. autoclass:: RectangularWindow

    .. automethod:: __call__

.. autoclass:: LinearRampWindow

    .. automethod:: __call__

.. autoclass:: LogisticWindow

    .. automethod:: __call__

.. autoclass:: SmoothedTrapezoidalWindow

    .. automethod:: __call__

Masks
-----

.. autoclass:: Mask

    .. automethod:: __call__

    .. automethod:: info

.. autoclass:: MaskSum

.. autoclass:: NoMask

    .. automethod:: __call__

.. autoclass:: RadialCosineMask

    .. automethod:: __call__

Imaginary Potentials
--------------------

.. autoclass:: ImaginaryGaussianRing

    .. automethod:: __call__

