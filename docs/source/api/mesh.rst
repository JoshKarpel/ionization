Mesh-Based TDSE Simulations
==========================================================

.. currentmodule:: ionization.mesh

Specifications
--------------

.. autoclass:: MeshSpecification

.. autoclass:: LineSpecification

.. autoclass:: CylindricalSliceSpecification

.. autoclass:: CylindricalSliceSpecification

.. autoclass:: SphericalHarmonicSpecification

Simulations
-----------

.. autoclass:: MeshSimulation

    ..automethod:: run

.. autoclass:: SphericalHarmonicSimulation

    ..automethod:: run


Quantum Meshes
--------------

.. autoclass:: QuantumMesh

.. autoclass:: LineMesh

.. autoclass:: CylindricalSliceMesh

.. autoclass:: SphericalSliceMesh

.. autoclass:: SphericalHarmonicMesh


Operators
---------

.. autoclass:: MeshOperator

.. autoclass:: Operators

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
