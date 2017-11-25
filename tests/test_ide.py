import pytest
import hypothesis as hyp
import hypothesis.strategies as st

import numpy as np

import ionization as ion
import ionization.integrodiff as ide
from simulacra.units import *

FIXED_STEP_METHODS = [
    'FE',
    'BE',
    'TRAP',
    'RK4',
]


@pytest.mark.parametrize(
    'evolution_method',
    FIXED_STEP_METHODS
)
@hyp.settings(
    max_examples = 10,
    deadline = None,
)
@hyp.given(
    b_initial_real = st.floats(allow_nan = False, allow_infinity = False),
    b_initial_imag = st.floats(allow_nan = False, allow_infinity = False),
)
def test_if_no_potential_final_state_is_initial_state(evolution_method, b_initial_real, b_initial_imag):
    b_initial = b_initial_real + (1j * b_initial_imag)

    sim = ide.IntegroDifferentialEquationSpecification(
        'test',
        b_initial = b_initial,
        time_initial = 0 * asec,
        time_final = 100 * asec,
        evolution_method = evolution_method
    ).to_simulation()

    sim.run_simulation()

    np.testing.assert_allclose(sim.b[-1], b_initial)
