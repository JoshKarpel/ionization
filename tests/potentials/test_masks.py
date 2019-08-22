import pytest
import hypothesis as hyp
import hypothesis.strategies as st


import numpy as np

import ionization as ion


@hyp.given(inner_radius=st.floats(min_value=0, allow_nan=False, allow_infinity=False))
def test_can_construct_radial_cosine_mask_with_positive_inner_radius(inner_radius):
    hyp.assume(inner_radius > 0)

    outer_radius = inner_radius + 1
    hyp.assume(outer_radius > inner_radius)

    ion.potentials.RadialCosineMask(
        inner_radius=inner_radius, outer_radius=outer_radius
    )


@hyp.given(inner_radius=st.floats(max_value=0, allow_nan=False, allow_infinity=False))
def test_cannot_construct_radial_cosine_mask_with_non_positive_inner_radius(
    inner_radius
):
    outer_radius = abs(inner_radius)
    hyp.assume(outer_radius > inner_radius)

    with pytest.raises(ion.exceptions.InvalidMaskParameter):
        ion.potentials.RadialCosineMask(
            inner_radius=inner_radius, outer_radius=outer_radius
        )


@hyp.given(
    inner_radius=st.floats(min_value=0, allow_infinity=False, allow_nan=False),
    data=st.data(),
)
def test_can_construct_radial_cosine_mask_with_inner_radius_less_than_outer_radius(
    inner_radius, data
):
    outer_radius = data.draw(st.floats(min_value=inner_radius))

    hyp.assume(inner_radius != outer_radius)

    ion.potentials.RadialCosineMask(
        inner_radius=inner_radius, outer_radius=outer_radius
    )


@hyp.given(
    inner_radius=st.floats(min_value=0, allow_infinity=False, allow_nan=False),
    data=st.data(),
)
def test_cannot_construct_radial_cosine_mask_with_inner_radius_larger_than_outer_radius(
    inner_radius, data
):
    outer_radius = data.draw(st.floats(min_value=0, max_value=inner_radius))
    print(outer_radius)

    with pytest.raises(ion.exceptions.InvalidMaskParameter):
        ion.potentials.RadialCosineMask(
            inner_radius=inner_radius, outer_radius=outer_radius
        )


@hyp.given(smoothness=st.floats(min_value=1, allow_nan=False, allow_infinity=False))
def test_can_construct_radial_cosine_mask_with_good_smoothness(smoothness):
    ion.potentials.RadialCosineMask(smoothness=smoothness)


@hyp.given(smoothness=st.floats(max_value=1, allow_nan=False, allow_infinity=False))
def test_cannot_construct_radial_cosine_mask_with_bad_smoothness(smoothness):
    hyp.assume(smoothness < 1)

    with pytest.raises(ion.exceptions.InvalidMaskParameter):
        ion.potentials.RadialCosineMask(smoothness=smoothness)


@hyp.given(outer_radius=st.floats(min_value=0, allow_infinity=False, allow_nan=False))
def test_radial_cosine_mask_is_one_at_inner_radius(outer_radius):
    inner_radius = 0
    hyp.assume(outer_radius > inner_radius)

    mask = ion.potentials.RadialCosineMask(
        inner_radius=inner_radius, outer_radius=outer_radius
    )

    mask_at_outer_radius = mask(r=inner_radius)

    assert np.allclose(mask_at_outer_radius, 1)


# sometimes outer radius is very close to inner radius and makes cos misbehave
@pytest.mark.filterwarnings("ignore: invalid value encountered in cos")
@hyp.given(outer_radius=st.floats(min_value=0, allow_infinity=False, allow_nan=False))
def test_radial_cosine_mask_is_zero_at_outer_radius(outer_radius):
    inner_radius = 0
    hyp.assume(outer_radius > inner_radius)

    mask = ion.potentials.RadialCosineMask(
        inner_radius=inner_radius, outer_radius=outer_radius
    )

    mask_at_outer_radius = mask(r=outer_radius)

    assert np.allclose(mask_at_outer_radius, 0, atol=1e-14)
