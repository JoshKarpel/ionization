import itertools

import pytest
import hypothesis as hyp
import hypothesis.strategies as st

import numpy as np

import ionization as ion
from simulacra.units import *


@hyp.settings(
    deadline = None,
)
@hyp.given(
    n = st.integers(min_value = 1, max_value = 100)
)
def test_hydrogen_bound_state_energy_degeneracy(n):
    ref = ion.HydrogenBoundState(n)
    assert all(ion.HydrogenBoundState(n, l, m).energy == ref.energy for l in range(n) for m in range(-l, l + 1))
