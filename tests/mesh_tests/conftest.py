import ionization as ion

SPEC_TYPES = [
    ion.mesh.LineSpecification,
    ion.mesh.CylindricalSliceSpecification,
    ion.mesh.SphericalSliceSpecification,
    ion.mesh.SphericalHarmonicSpecification,
]
SPEC_TO_LENGTH_GAUGE_OPERATOR_TYPE = {
    ion.mesh.LineSpecification: ion.mesh.LineLengthGaugeOperators,
    ion.mesh.CylindricalSliceSpecification: ion.mesh.CylindricalSliceLengthGaugeOperators,
    ion.mesh.SphericalSliceSpecification: ion.mesh.SphericalSliceLengthGaugeOperators,
    ion.mesh.SphericalHarmonicSpecification: ion.mesh.SphericalHarmonicLengthGaugeOperators,
}
THREE_DIMENSIONAL_SPEC_TYPES = [
    ion.mesh.CylindricalSliceSpecification,
    ion.mesh.SphericalSliceSpecification,
    ion.mesh.SphericalHarmonicSpecification,
]
SPEC_TYPES_WITH_NUMERIC_EIGENSTATES = [
    ion.mesh.LineSpecification,
    ion.mesh.SphericalHarmonicSpecification,
]
LOW_N_HYDROGEN_BOUND_STATES = [ion.mesh.states.HydrogenBoundState(n, l) for n in range(6) for l in range(n)]
