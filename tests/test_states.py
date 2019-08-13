import pytest
import hypothesis as hyp
import hypothesis.strategies as st

import numpy as np

from simulacra.units import *


def test_single_state_normalized():
    s = states.HydrogenBoundState(amplitude=0.2 + 0.3j)

    assert np.allclose(s.normalized().norm, 1)


def test_superposition_normalized():
    s = states.HydrogenBoundState(amplitude=0.5)
    s += states.HydrogenBoundState(n=2, amplitude=0.5j)

    sn = s.normalized()

    assert np.allclose(sn.norm, 1)
    for state in sn:
        assert np.allclose(np.abs(state.amplitude), np.sqrt(1 / 2))


def test_superposition_normalized_with_three_states():
    s = states.HydrogenBoundState(n=1, amplitude=1)
    s += states.HydrogenBoundState(n=2, amplitude=1)
    s += states.HydrogenBoundState(n=3, amplitude=1)

    sn = s.normalized()

    for state in sn:
        assert np.allclose(state.amplitude, np.sqrt(1 / 3))


class TestHydrogenBoundStateConstructor:
    @hyp.given(n=st.integers(min_value=1))
    def test_can_construct_with_positive_n(self, n):
        states.HydrogenBoundState(n)

    @hyp.given(n=st.integers(max_value=0))
    def test_cannot_construct_with_non_positive_n(self, n):
        with pytest.raises(exceptions.IllegalQuantumState):
            states.HydrogenBoundState(n)

    @hyp.given(n=st.integers(min_value=1), data=st.data())
    def test_can_construct_with_good_l(self, n, data):
        good_l = data.draw(st.integers(min_value=0, max_value=n - 1))

        states.HydrogenBoundState(n, good_l)

    @hyp.given(n=st.integers(min_value=1), data=st.data())
    def test_cannot_construct_with_bad_l(self, n, data):
        bad_l = data.draw(st.integers(min_value=n + 1))

        with pytest.raises(exceptions.IllegalQuantumState):
            states.HydrogenBoundState(n, bad_l)

    @hyp.given(n=st.integers(min_value=1), data=st.data())
    def test_can_construct_with_good_m(self, n, data):
        good_l = data.draw(st.integers(min_value=0, max_value=n - 1))
        good_m = data.draw(st.integers(min_value=-good_l, max_value=good_l))

        states.HydrogenBoundState(n, good_l, good_m)

    @hyp.given(n=st.integers(min_value=1), data=st.data())
    def test_cannot_construct_with_bad_m(self, n, data):
        good_l = data.draw(st.integers(min_value=0, max_value=n - 1))
        bad_m = data.draw(
            st.one_of(
                st.integers(max_value=-good_l - 1), st.integers(min_value=good_l + 1)
            )
        )

        with pytest.raises(exceptions.IllegalQuantumState):
            states.HydrogenBoundState(n, good_l, bad_m)

    @hyp.settings(max_examples=20, deadline=None)
    @hyp.given(n=st.integers(min_value=1, max_value=100))
    def test_energy_degeneracy(self, n):
        ref = states.HydrogenBoundState(n).energy
        assert all(
            states.HydrogenBoundState(n, l, m).energy == ref
            for l in range(n)
            for m in range(-l, l + 1)
        )


class TestHydrogenCoulombState:
    @hyp.given(energy=st.floats(min_value=0, max_value=MeV))
    def test_can_construct_with_non_negative_energy(self, energy):
        states.HydrogenCoulombState(energy=energy)

    @hyp.given(energy=st.floats(max_value=0))
    def test_cannot_construct_with_negative_energy(self, energy):
        hyp.assume(energy < 0)

        with pytest.raises(exceptions.IllegalQuantumState):
            states.HydrogenCoulombState(energy=energy)

    @hyp.given(l=st.integers(min_value=0))
    def test_can_construct_with_non_negative_l(self, l):
        states.HydrogenCoulombState(l=l)

    @hyp.given(l=st.integers(max_value=-1))
    def test_cannot_construct_with_negative_l(self, l):
        with pytest.raises(exceptions.IllegalQuantumState):
            states.HydrogenCoulombState(l)

    @hyp.given(l=st.integers(min_value=0), data=st.data())
    def test_can_construct_with_good_m(self, l, data):
        good_m = data.draw(st.integers(min_value=-l, max_value=l))

        states.HydrogenCoulombState(l=l, m=good_m)

    @hyp.given(l=st.integers(min_value=0), data=st.data())
    def test_cannot_construct_with_bad_m(self, l, data):
        bad_m = data.draw(
            st.one_of(st.integers(max_value=-(l + 1)), st.integers(min_value=l + 1))
        )

        with pytest.raises(exceptions.IllegalQuantumState):
            states.HydrogenCoulombState(l=l, m=bad_m)
