import simulacra.units as u

LENGTH_UNITS = (
    (u.bohr_radius, 'a_0'),
    (u.nm, 'nm'),
)
TIME_UNITS = (
    (u.asec, 'as'),
    (u.fsec, 'fs'),
    (u.atomic_time, 'a.u.'),
)
FREQUENCY_UNITS = (
    (u.THz, 'THz'),
)
ELECTRIC_FIELD_UNITS = (
    (u.atomic_electric_field, 'a.u.'),
    (u.V_per_m, 'V/m'),
)
ENERGY_UNITS = (
    (u.eV, 'eV'),
    (u.rydberg, 'Rydberg'),
    (u.hartree, 'Hartree'),
)
ANGLE_UNITS = (
    (u.pi, '𝜋 rad'),
    (u.deg, 'deg'),
)
FLUENCE_UNITS = (
    (u.Jcm2, 'J/cm^2'),
)
CHARGE_UNITS = (
    (u.proton_charge, 'e'),
)
FORCE_UNITS = (
    (u.N_per_m, 'N/m'),
    (u.atomic_force / u.bohr_radius, 'a.u./a_0'),
)
MASS_UNITS = (
    (u.electron_mass, 'm_e'),
    (u.electron_mass_reduced, 'mu_e'),
)


def fmt_quantity(quantity, units):
    """Format a single quantity for multiple units, as in Info fields."""
    strs = [f'{u.uround(quantity, v):.4g} {s}' for v, s in units]
    return ' | '.join(strs)


def fmt_fields(obj, *fields, digits: int = 3):
    """
    Generate a repr-like string from the object's attributes.

    Each field should be a string containing the name of an attribute or a ('attribute_name', 'unit_name') pair. uround will be used to format in the second case.

    :param obj: the object to get attributes from
    :param fields: the attributes or (attribute, unit) pairs to get from obj
    :param digits: the number of digits to round to for uround
    :return: the formatted string
    """
    field_strings = []
    for field in fields:
        try:
            field_name, unit = field
            try:
                field_strings.append('{} = {} {}'.format(field_name, u.uround(getattr(obj, field_name), unit, digits = digits), unit))
            except TypeError:
                field_strings.append('{} = {}'.format(field_name, getattr(obj, field_name)))
        except (ValueError, TypeError):
            field_strings.append('{} = {}'.format(field, getattr(obj, field)))
    return '{}({})'.format(obj.__class__.__name__, ', '.join(field_strings))
