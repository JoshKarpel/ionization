Mesh-Based TDSE Simulations
==========================================================

.. currentmodule:: ionization.mesh

Specifications
--------------

.. autoclass:: MeshSpecification

    .. automethod:: to_sim

    .. automethod:: info

.. autoclass:: LineSpecification

.. autoclass:: CylindricalSliceSpecification

.. autoclass:: SphericalSliceSpecification

.. autoclass:: SphericalHarmonicSpecification

Simulations
-----------

.. autoclass:: MeshSimulation

    .. automethod:: run

    .. automethod:: info

.. autoclass:: SphericalHarmonicSimulation


Quantum Meshes
--------------

.. autoclass:: QuantumMesh

    .. automethod:: flatten_mesh

    .. automethod:: wrap_vector

    .. automethod:: wrapping_direction_to_order

    .. automethod:: get_g_for_state

    .. automethod:: state_to_g

    .. automethod:: get_g_with_states_removed

    .. automethod:: inner_product

    .. automethod:: state_overlap

    .. automethod:: expectation_value

    .. automethod:: norm

    .. automethod:: internal_energy_expectation_value

    .. automethod:: total_energy_expectation_value

    .. automethod:: z_expectation_value

    .. automethod:: r_expectation_value

    .. automethod:: evolve

.. autoclass:: LineMesh

.. autoclass:: CylindricalSliceMesh

    .. automethod:: get_probability_current_density_vector_field

    .. automethod:: get_spline_for_mesh

.. autoclass:: SphericalSliceMesh

    .. automethod:: get_spline_for_mesh

.. autoclass:: SphericalHarmonicMesh

    .. automethod:: get_radial_g_for_state

    .. automethod:: norm_by_l

    .. automethod:: inner_product_with_plane_waves


Operators
---------

These classes represent discretized quantum operators.

.. autoclass:: MeshOperator

    .. automethod:: apply

    .. automethod:: _apply

Mesh Operators
--------------

These classes are strategies that :class:`QuantumMesh` instances use to generate operators to act on their wavefunctions.

.. autoclass:: MeshOperators

    .. automethod:: info

.. autoclass:: LineLengthGaugeOperators

.. autoclass:: LineVelocityGaugeOperators

.. autoclass:: CylindricalSliceLengthGaugeOperators

.. autoclass:: SphericalSliceLengthGaugeOperators

.. autoclass:: SphericalHarmonicLengthGaugeOperators

.. autoclass:: SphericalHarmonicVelocityGaugeOperators


Evolution Methods
-----------------

.. autoclass:: EvolutionMethod

    .. automethod:: evolve

    .. automethod:: get_evolution_operators

    .. automethod:: info

.. autoclass:: AlternatingDirectionImplicit

.. autoclass:: SplitInteractionOperator


Datastores
----------

.. autoclass:: Data

.. autoclass:: Datastore

    .. automethod:: init

    .. automethod:: store

    .. automethod:: attach

.. autoclass:: Fields

.. autoclass:: Norm

.. autoclass:: InnerProducts

.. autoclass:: InternalEnergyExpectationValue

.. autoclass:: TotalEnergyExpectationValue

.. autoclass:: ZExpectationValue

.. autoclass:: RExpectationValue

.. autoclass:: NormWithinRadius

.. autoclass:: NormBySphericalHarmonic

.. autoclass:: DirectionalRadialProbabilityCurrent
