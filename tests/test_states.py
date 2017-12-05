import pytest
import hypothesis as hyp
import hypothesis.strategies as st

import ionization as ion
from simulacra.units import *


class TestHydrogenBoundState:
    @hyp.given(
        n = st.integers(min_value = 1),
    )
    def test_can_construct_with_positive_n(self, n):
        ion.states.HydrogenBoundState(n)

    @hyp.given(
        n = st.integers(max_value = 0),
    )
    def test_cannot_construct_with_non_positive_n(self, n):
        with pytest.raises(ion.exceptions.IllegalQuantumState):
            ion.states.HydrogenBoundState(n)

    @hyp.given(
        n = st.integers(min_value = 1),
        data = st.data(),
    )
    def test_can_construct_with_good_l(self, n, data):
        good_l = data.draw(st.integers(min_value = 0, max_value = n - 1))

        ion.states.HydrogenBoundState(n, good_l)

    @hyp.given(
        n = st.integers(min_value = 1),
        data = st.data(),
    )
    def test_cannot_construct_with_bad_l(self, n, data):
        bad_l = data.draw(st.integers(min_value = n + 1))

        with pytest.raises(ion.exceptions.IllegalQuantumState):
            ion.states.HydrogenBoundState(n, bad_l)

    @hyp.given(
        n = st.integers(min_value = 1),
        data = st.data(),
    )
    def test_can_construct_with_good_m(self, n, data):
        good_l = data.draw(st.integers(min_value = 0, max_value = n - 1))
        good_m = data.draw(st.integers(min_value = -good_l, max_value = good_l))

        ion.states.HydrogenBoundState(n, good_l, good_m)

    @hyp.given(
        n = st.integers(min_value = 1),
        data = st.data(),
    )
    def test_cannot_construct_with_bad_m(self, n, data):
        good_l = data.draw(st.integers(min_value = 0, max_value = n - 1))
        bad_m = data.draw(st.one_of(
            st.integers(max_value = -good_l - 1),
            st.integers(min_value = good_l + 1),
        ))

        with pytest.raises(ion.exceptions.IllegalQuantumState):
            ion.states.HydrogenBoundState(n, good_l, bad_m)

    @hyp.settings(
        max_examples = 20,
        deadline = None,
    )
    @hyp.given(
        n = st.integers(min_value = 1, max_value = 100),
    )
    def test_energy_degeneracy(self, n):
        ref = ion.states.HydrogenBoundState(n).energy
        assert all(ion.states.HydrogenBoundState(n, l, m).energy == ref for l in range(n) for m in range(-l, l + 1))


class TestHydrogenCoulombState:
    @hyp.given(
        energy = st.floats(min_value = 0, max_value = MeV),
    )
    def test_can_construct_with_non_negative_energy(self, energy):
        ion.states.HydrogenCoulombState(energy = energy)

    @hyp.given(
        energy = st.floats(max_value = 0),
    )
    def test_cannot_construct_with_negative_energy(self, energy):
        hyp.assume(energy < 0)

        with pytest.raises(ion.exceptions.IllegalQuantumState):
            ion.states.HydrogenCoulombState(energy = energy)

    @hyp.given(
        l = st.integers(min_value = 0)
    )
    def test_can_construct_with_non_negative_l(self, l):
        ion.states.HydrogenCoulombState(l = l)

    @hyp.given(
        l = st.integers(max_value = -1)
    )
    def test_cannot_construct_with_negative_l(self, l):
        with pytest.raises(ion.exceptions.IllegalQuantumState):
            ion.states.HydrogenCoulombState(l)

    @hyp.given(
        l = st.integers(min_value = 0),
        data = st.data(),
    )
    def test_can_construct_with_good_m(self, l, data):
        good_m = data.draw(st.integers(min_value = -l, max_value = l))

        ion.states.HydrogenCoulombState(l = l, m = good_m)

    @hyp.given(
        l = st.integers(min_value = 0),
        data = st.data(),
    )
    def test_cannot_construct_with_bad_m(self, l, data):
        bad_m = data.draw(st.one_of(
            st.integers(max_value = -(l + 1)),
            st.integers(min_value = l + 1),
        ))

        with pytest.raises(ion.exceptions.IllegalQuantumState):
            ion.states.HydrogenCoulombState(l = l, m = bad_m)
