import itertools

import pytest
import hypothesis as hyp
import hypothesis.strategies as st

import numpy as np

import ionization as ion
from simulacra.units import *


@hyp.given(
    n = st.integers(min_value = 1),
)
def test_hydrogen_bound_state_can_be_constructed_with_positive_n(n):
    ion.HydrogenBoundState(n)


@hyp.given(
    n = st.integers(max_value = 0),
)
def test_hydrogen_bound_state_cannot_be_constructed_with_non_positive_n(n):
    with pytest.raises(ion.IllegalQuantumState):
        ion.HydrogenBoundState(n)


@pytest.mark.parametrize(
    'n', range(1, 50)
)
def test_hydrogen_bound_state_energy_degeneracy(n):
    ref = ion.HydrogenBoundState(n)
    assert all(ion.HydrogenBoundState(n, l, m).energy == ref.energy for l in range(n) for m in range(-l, l + 1))
