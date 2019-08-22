import pytest
import hypothesis as hyp
import hypothesis.strategies as st

import numpy as np

import ionization as ion

import simulacra.units as u


class TestImaginaryGaussianRing:
    @hyp.given(decay_time=st.floats(min_value=0, allow_nan=False, allow_infinity=False))
    def test_decay_time_can_be_positive(self, decay_time):
        hyp.assume(decay_time > 0)

        ion.potentials.ImaginaryGaussianRing(decay_time=decay_time)

    @hyp.given(decay_time=st.floats(max_value=0, allow_nan=False, allow_infinity=False))
    def test_decay_time_cannot_be_non_positive(self, decay_time):
        with pytest.raises(ion.exceptions.InvalidPotentialParameter):
            ion.potentials.ImaginaryGaussianRing(decay_time=decay_time)

    @hyp.given(center=st.floats(min_value=0, allow_nan=False, allow_infinity=False))
    def test_center_can_be_non_negative(self, center):
        ion.potentials.ImaginaryGaussianRing(center=center)

    @hyp.given(center=st.floats(max_value=0, allow_nan=False, allow_infinity=False))
    def test_center_cannot_be_negative(self, center):
        hyp.assume(center < 0)
        with pytest.raises(ion.exceptions.InvalidPotentialParameter):
            ion.potentials.ImaginaryGaussianRing(center=center)

    @hyp.given(width=st.floats(min_value=0, allow_nan=False, allow_infinity=False))
    def test_width_can_be_positive(self, width):
        hyp.assume(width > 0)

        ion.potentials.ImaginaryGaussianRing(width=width)

    @hyp.given(width=st.floats(max_value=0, allow_nan=False, allow_infinity=False))
    def test_width_cannot_be_non_positive(self, width):
        with pytest.raises(ion.exceptions.InvalidPotentialParameter):
            ion.potentials.ImaginaryGaussianRing(width=width)

    def test_decay_time_can_be_real(self):
        ion.potentials.ImaginaryGaussianRing(decay_time=100 * u.asec)

    def test_decay_time_cannot_have_imaginary_part(self):
        with pytest.raises(ion.exceptions.InvalidPotentialParameter):
            ion.potentials.ImaginaryGaussianRing(decay_time=(100 + 100j) * u.asec)

    def test_decay_time_cannot_be_imaginary(self):
        with pytest.raises(ion.exceptions.InvalidPotentialParameter):
            ion.potentials.ImaginaryGaussianRing(decay_time=100j * u.asec)

    def test_potential_is_imaginary_everywhere(self):
        pot = ion.potentials.ImaginaryGaussianRing()
        r = np.linspace(0, pot.center * 2, 10000)

        assert np.allclose(np.real(pot(r=r)), 0, atol=1e-10)
