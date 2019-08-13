SPEC_TYPES = [
    mesh.LineSpecification,
    mesh.CylindricalSliceSpecification,
    mesh.SphericalSliceSpecification,
    mesh.SphericalHarmonicSpecification,
]
SPEC_TO_LENGTH_GAUGE_OPERATOR_TYPE = {
    mesh.LineSpecification: mesh.LineLengthGaugeOperators,
    mesh.CylindricalSliceSpecification: mesh.CylindricalSliceLengthGaugeOperators,
    mesh.SphericalSliceSpecification: mesh.SphericalSliceLengthGaugeOperators,
    mesh.SphericalHarmonicSpecification: mesh.SphericalHarmonicLengthGaugeOperators,
}
THREE_DIMENSIONAL_SPEC_TYPES = [
    mesh.CylindricalSliceSpecification,
    mesh.SphericalSliceSpecification,
    mesh.SphericalHarmonicSpecification,
]
SPEC_TYPES_WITH_NUMERIC_EIGENSTATES = [
    mesh.LineSpecification,
    mesh.SphericalHarmonicSpecification,
]
LOW_N_HYDROGEN_BOUND_STATES = [
    states.HydrogenBoundState(n, l) for n in range(3) for l in range(n)
]
