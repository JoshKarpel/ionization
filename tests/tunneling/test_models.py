import pytest

import simulacra.units as u

from .conftest import TUNNELING_MODEL_TYPES


@pytest.mark.parametrize("model_type", TUNNELING_MODEL_TYPES)
def test_tunneling_rate_is_zero_for_zero_amplitude(model_type):
    model = model_type()

    assert (
        model.tunneling_rate(
            electric_field_amplitude=0, ionization_potential=-u.rydberg
        )
        == 0
    )


@pytest.mark.parametrize("model_type", TUNNELING_MODEL_TYPES)
@pytest.mark.parametrize(
    "electric_field_amplitude",
    [
        10 * u.atomic_electric_field,
        0.1 * u.atomic_electric_field,
        0.01 * u.atomic_electric_field,
        0 * u.atomic_electric_field,
        -0.01 * u.atomic_electric_field,
        -0.1 * u.atomic_electric_field,
        -10 * u.atomic_electric_field,
    ],
)
def test_tunneling_rate_is_never_positive(model_type, electric_field_amplitude):
    model = model_type()

    assert (
        model.tunneling_rate(
            electric_field_amplitude=electric_field_amplitude,
            ionization_potential=-u.rydberg,
        )
        <= 0
    )


@pytest.mark.parametrize("model_type", TUNNELING_MODEL_TYPES)
@pytest.mark.parametrize(
    "electric_field_amplitude",
    [
        10 * u.atomic_electric_field,
        0.1 * u.atomic_electric_field,
        0.01 * u.atomic_electric_field,
        0 * u.atomic_electric_field,
    ],
)
def test_tunneling_rate_is_same_for_positive_or_negative_amplitude(
    model_type, electric_field_amplitude
):
    model = model_type()

    assert model.tunneling_rate(
        electric_field_amplitude, -u.rydberg
    ) == model.tunneling_rate(-electric_field_amplitude, -u.rydberg)


@pytest.mark.parametrize("model_type", TUNNELING_MODEL_TYPES)
@pytest.mark.parametrize(
    "upper_amplitude_cutoff",
    [1 * u.atomic_electric_field, 0.3 * u.atomic_electric_field],
)
def test_upper_cutoff_amplitude(model_type, upper_amplitude_cutoff):
    model_without_cutoff = model_type()
    model_with_cutoff = model_type(upper_amplitude_cutoff=upper_amplitude_cutoff)

    efield_shift = upper_amplitude_cutoff / 10

    assert (
        model_with_cutoff.tunneling_rate(
            upper_amplitude_cutoff + efield_shift, ionization_potential=-u.rydberg
        )
        == 0
    )
    assert model_with_cutoff.tunneling_rate(
        upper_amplitude_cutoff - efield_shift, ionization_potential=-u.rydberg
    ) == model_without_cutoff.tunneling_rate(
        upper_amplitude_cutoff - efield_shift, ionization_potential=-u.rydberg
    )
    assert model_with_cutoff.tunneling_rate(
        -(upper_amplitude_cutoff - efield_shift), ionization_potential=-u.rydberg
    ) == model_without_cutoff.tunneling_rate(
        -(upper_amplitude_cutoff - efield_shift), ionization_potential=-u.rydberg
    )
    assert (
        model_with_cutoff.tunneling_rate(
            -(upper_amplitude_cutoff + efield_shift), ionization_potential=-u.rydberg
        )
        == 0
    )
