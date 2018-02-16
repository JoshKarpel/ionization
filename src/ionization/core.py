import collections
import logging
import enum
from typing import NewType

import numpy as np
from scipy.misc import factorial

import simulacra as si
import simulacra.units as u

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

AngularMomentum = NewType('AngularMomentum', int)


def electron_energy_from_wavenumber(k):
    return (u.hbar * k) ** 2 / (2 * u.electron_mass)


def electron_wavenumber_from_energy(energy):
    return np.sqrt(2 * u.electron_mass * energy + 0j) / u.hbar


def triangle_coef(a, b, c):
    return factorial(a + b - c) * factorial(a - b + c) * factorial(-a + b + c) / factorial(a + b + c + 1)


@si.utils.memoize
def cg_coef(j1, m1, j2, m2, j, m):
    """
    Return the Clebsch-Gordan coefficient <j1, j2; m1, m2 | j1, j2; j, m> using the Racah formula.

    See:
    Rudnicki-Bujnowski, G. Explicit formulas for Clebsch-Gordan coefficients. Comput. Phys. Commun. 10, 245–250 (1975)


    Parameters
    ----------
    j1
    m1
    j2
    m2
    j
    m

    Returns
    -------
    float
        A Clebsch-Gordan coefficient.
    """
    if m1 + m2 != m or not abs(j1 - j2) <= j <= j1 + j2:
        return 0

    cg = np.sqrt(((2 * j) + 1) * triangle_coef(j1, j2, j) * factorial(j1 + m1) * factorial(j1 - m1) * factorial(j2 + m2) * factorial(j2 - m2) * factorial(j + m) * factorial(j - m))

    t_min = int(max(-j + j2 - m, -j + j1 + m2, 0))
    t_max = int(min(j1 + j2 - j, j1 - m1, j2 + m2))

    s = 0
    for t in range(t_min, t_max + 1):
        s += ((-1) ** t) / (factorial(t) * factorial(j - j2 + t + m1) * factorial(j - j1 + t - m2) * factorial(j1 + j2 - j - t) * factorial(j1 - t - m1) * factorial(j2 - t + m2))

    return cg * s


@si.utils.memoize
def triple_y_integral(j1, m1, j2, m2, j, m):
    """
    j, m is whichever angular momentum is complex-conjugated in the integrand

    Parameters
    ----------
    j1
    m1
    j2
    m2
    j
    m

    Returns
    -------

    """
    if m1 + m2 != m or not abs(j1 - j2) <= j <= j1 + j2:
        return 0

    y1 = si.math.SphericalHarmonic(j1, m1)
    y2 = si.math.SphericalHarmonic(j2, m2)
    y3 = si.math.SphericalHarmonic(j, m)

    def integrand(theta, phi):
        return y1(theta, phi) * y2(theta, phi) * np.conj(y3(theta, phi)) * np.sin(theta)

    result, *errs = si.math.complex_nquad(integrand, [(0, u.pi), (0, u.twopi)], opts = {'limit': 1000})

    return np.real(result)


warning_record = collections.namedtuple('warning_record', ['data_time_index', 'message'])


class Gauge(si.utils.StrEnum):
    LENGTH = 'LEN'
    VELOCITY = 'VEL'
