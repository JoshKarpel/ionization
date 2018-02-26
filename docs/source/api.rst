API Reference
=============

.. currentmodule:: ionization

Mesh-based Time-Dependent Schrödinger Equation Simulations
----------------------------------------------------------

Meshes
^^^^^^
.. currentmodule:: ionization.mesh

Specifications
++++++++++++++

.. autoclass:: MeshSpecification

.. autoclass:: LineSpecification

.. autoclass:: CylindricalSliceSpecification

.. autoclass:: CylindricalSliceSpecification

.. autoclass:: SphericalHarmonicSpecification

Simulations
+++++++++++

.. autoclass:: MeshSimulation

.. autoclass:: SphericalHarmonicSimulation

Meshes
++++++

.. autoclass:: QuantumMesh

.. autoclass:: LineMesh

.. autoclass:: CylindricalSliceMesh

.. autoclass:: SphericalSliceMesh

.. autoclass:: SphericalHarmonicMesh

Operators
+++++++++

.. autoclass:: MeshOperator

.. autoclass:: Operators

.. autoclass:: LineLengthGaugeOperators

.. autoclass:: LineVelocityGaugeOperators

.. autoclass:: CylindricalSliceLengthGaugeOperators

.. autoclass:: SphericalSliceLengthGaugeOperators

.. autoclass:: SphericalHarmonicLengthGaugeOperators

.. autoclass:: SphericalHarmonicVelocityGaugeOperators


Evolution Methods
+++++++++++++++++

.. autoclass:: EvolutionMethod

.. autoclass:: AlternatingDirectionImplicitCrankNicolson

.. autoclass:: LineSplitOperator

.. autoclass:: SphericalHarmonicSplitOperator
