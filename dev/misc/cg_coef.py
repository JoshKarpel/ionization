"""
This script tests all of the evolution methods on each mesh.
"""

import itertools as it
import logging
import os

import numpy as np
import scipy.integrate as integ
from scipy.misc import factorial

import simulacra as si
from simulacra.units import *

import ionization as ion


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

logman = si.utils.LogManager('simulacra', 'ionization',
                             stdout_level = logging.DEBUG)


def triangle(a, b, c):
    print(a + b - c, a - b + c, -a + b + c, a + b + c + 1)
    return factorial(a + b - c) * factorial(a - b + c) * factorial(-a + b + c) / factorial(a + b + c + 1)


# @si.utils.timed
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
        print('short')
        return 0

    cg = np.sqrt(((2 * j) + 1) * triangle(j1, j2, j) * factorial(j1 + m1) * factorial(j1 - m1) * factorial(j2 + m2) * factorial(j2 - m2) * factorial(j + m) * factorial(j - m))

    t_min = int(max(-j + j2 - m, -j + j1 + m2, 0))
    t_max = int(min(j1 + j2 - j, j1 - m1, j2 + m2))

    s = 0
    for t in range(t_min, t_max + 1):
        s += ((-1) ** t) / (factorial(t) * factorial(j - j2 + t + m1) * factorial(j - j1 + t - m2) * factorial(j1 + j2 - j - t) * factorial(j1 - t - m1) * factorial(j2 - t + m2))

    return cg * s

@si.utils.timed
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
        print('short')
        return 0

    y1 = si.math.SphericalHarmonic(j1, m1)
    y2 = si.math.SphericalHarmonic(j2, m2)
    y3 = si.math.SphericalHarmonic(j, m)

    def integrand(theta, phi):
        return y1(theta, phi) * y2(theta, phi) * np.conj(y3(theta, phi)) * np.sin(theta)

    result = si.math.complex_nquad(integrand, [(0, pi), (0, twopi)], opts = {'limit': 1000})
    logger.debug(result)

    return np.real(result[0])


def pre(j1, j2, j):
    return np.sqrt(((2 * j1) + 1) * ((2 * j2) + 1) / (4 * pi * ((2 * j) + 1)))


if __name__ == '__main__':
    with logman as logger:
        print(cg_coef(.5, .5, .5, .5, 1, 1), 1)

        print(cg_coef(1, 1, .5, .5, 1.5, 1.5), 1)

        print(cg_coef(1, 1, .5, -.5, 1.5, .5), np.sqrt(1 / 3))
        print(cg_coef(1, 1, .5, -.5, .5, .5), np.sqrt(2 / 3))

        print(cg_coef(1, 0, .5, .5, 1.5, .5), np.sqrt(2 / 3))
        print(cg_coef(1, 0, .5, .5, .5, .5), -np.sqrt(1 / 3))

        print(cg_coef(1, -1, .5, .5, .5, -.5), -np.sqrt(2 / 3))

        print(cg_coef(1.5, -.5, 1, -1, 2.5, -1.5), np.sqrt(3 / 5))

        print(cg_coef(1, 0, 1, 0, 2, 0), np.sqrt(2 / 3))

        print(cg_coef(30, 18, 21, -5, 19, 13))

        print()

        print(integ.dblquad(lambda x, y: x * y, 0, 0.5, lambda y: 0, lambda y: 1 - 2 * y))
        print(si.math.complex_dblquad(lambda x, y: x * y, 0, 0.5, lambda y: 0, lambda y: 1 - 2 * y))

        print('num', triple_y_integral(1, 0, 1, 0, 2, 0), 1.5 * np.sqrt(1 / (5 * pi)) * (cg_coef(1, 0, 1, 0, 2, 0) ** 2))
        print('num', triple_y_integral(1, 1, 1, 0, 2, 1), 1.5 * np.sqrt(1 / (5 * pi)) * cg_coef(1, 0, 1, 0, 2, 0) * cg_coef(1, 1, 1, 0, 2, 1))

        print('num', triple_y_integral(30, 18, 21, -5, 19, 13), 0.5 * np.sqrt(2623 / (39 * pi)) * cg_coef(30, 0, 21, 0, 19, 0) * cg_coef(30, 18, 21, -5, 19, 13))
        print('num', triple_y_integral(30, 0, 21, 0, 19, 0), 0.5 * np.sqrt(2623 / (39 * pi)) * cg_coef(30, 0, 21, 0, 19, 0) * cg_coef(30, 0, 21, 0, 19, 0))

        print()
        args = (400, 0, 350, 0, 60, 0)
        print('num', triple_y_integral(*args))
        print('exact', pre(args[0], args[2], args[4]) * cg_coef(args[0], 0, args[2], 0, args[4], 0) * cg_coef(*args))
