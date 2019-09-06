from typing import Union

import simulacra.units as u

LENGTH_UNITS = ((u.bohr_radius, "a_0"), (u.nm, "nm"))
INVERSE_LENGTH_UNITS = ((u.per_bohr_radius, "1/a_0"), (u.per_nm, "1/nm"))
TIME_UNITS = ((u.asec, "as"), (u.fsec, "fs"), (u.atomic_time, "a.u."))
FREQUENCY_UNITS = ((u.THz, "THz"),)
ELECTRIC_FIELD_UNITS = ((u.atomic_electric_field, "a.u."), (u.V_per_m, "V/m"))
ENERGY_UNITS = ((u.eV, "eV"), (u.rydberg, "Rydberg"), (u.hartree, "Hartree"))
ANGLE_UNITS = ((u.pi, "𝜋 rad"), (u.deg, "deg"))
FLUENCE_UNITS = ((u.Jcm2, "J/cm^2"),)
CHARGE_UNITS = ((u.proton_charge, "e"),)
FORCE_UNITS = ((u.N_per_m, "N/m"), (u.atomic_force / u.bohr_radius, "a.u./a_0"))
MASS_UNITS = ((u.electron_mass, "m_e"), (u.electron_mass_reduced, "mu_e"))


def fmt_quantity(quantity, units):
    """Format a single quantity for multiple units, as in Info fields."""
    strs = [f"{quantity / v:.4g} {s}" for v, s in units]
    return " | ".join(strs)


def make_repr(obj, *fields, digits: int = 3):
    """
    Generate a repr-like string from the object's attributes.

    Each field should be a string containing the name of an attribute or a ('attribute_name', 'unit_name') pair.

    :param obj: the object to get attributes from
    :param fields: the attributes or (attribute, unit) pairs to get from obj
    :param digits: the number of digits to round to
    :return: the formatted string
    """
    field_strings = []
    for field in fields:
        try:
            field_name, unit = field
            try:
                field_strings.append(
                    f"{field_name} = {getattr(obj, field_name)/ unit:.{digits}f)} {unit}"
                )
            except TypeError:
                field_strings.append(f"{field_name} = {getattr(obj, field_name)}")
        except (ValueError, TypeError):
            field_strings.append(f"{field} = {getattr(obj, field)}")
    return f"{obj.__class__.__name__}({', '.join(field_strings)})"


def complex_j_to_i(z: complex) -> str:
    """
    Given a complex number, return its string representation, using ``i`` as the
    imaginary unit (instead of the default ``j``).
    """
    return str(z).replace("j", "i")
