import logging
from abc import ABC, abstractmethod
import functools
import warnings

import numpy as np
import scipy.integrate as integ
import scipy.special as special
import sympy as sym

import simulacra as si
import simulacra.units as u

from .. import states

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Kernel(ABC):
    @abstractmethod
    def __call__(self, current_time, previous_time, electric_potential, vector_potential):
        """
        Parameters
        ----------
        current_time
            The current time (i.e., t).
        previous_time
            The previous time (i.e., t').
        electric_potential
            The electric potential used in the simulation.
        vector_potential
            The interpolated vector potential of the electric potential.

        Returns
        -------
            The value of the kernel K(t, t'; E, A)
        """
        raise NotImplementedError

    def info(self):
        info = si.Info(header = f'Kernel: {self.__class__.__name__}')

        return info


class LengthGaugeHydrogenKernel(Kernel):
    """The kernel for the hydrogen ground state with spherical Bessel functions, with no continuum-continuum coupling, in the length gauge."""

    def __init__(self, bound_state_energy = states.HydrogenBoundState(1).energy):
        self.bound_state_energy = bound_state_energy
        self.omega_bound = bound_state_energy / u.hbar

        self.kernel_prefactor = (512 / (3 * u.pi)) * (u.bohr_radius ** 7)
        self.kernel_at_time_difference_zero_with_prefactor = u.bohr_radius ** 2

    def __call__(self, time_current, time_previous, electric_potential, vector_potential):
        time_difference = time_current - time_previous

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            kernel = self.kernel_prefactor * self.kernel_function(time_difference) * np.exp(1j * self.omega_bound * time_difference)

        kernel = np.where(
            time_difference != 0,
            kernel,
            self.kernel_at_time_difference_zero_with_prefactor
        )
        kernel[time_difference > 5 * u.fsec] = 0

        return kernel

    @property
    @si.utils.memoize
    def kernel_function(self):
        k = sym.Symbol('k', real = True)
        td = sym.Symbol('td', real = True, positive = True)
        a = sym.Symbol('a', real = True, positive = True)
        m = sym.Symbol('m', real = True, positive = True)
        hb = sym.Symbol('hb', real = True, positive = True)

        integrand = ((k ** 4) / ((1 + ((a * k) ** 2)) ** 6)) * (sym.exp(-sym.I * hb * (k ** 2) * td / (2 * m)))

        kernel = sym.integrate(integrand, (k, 0, sym.oo))

        kernel_func = sym.lambdify(
            (a, m, hb, td),
            kernel,
            modules = ['numpy', {'erfc': special.erfc}]
        )
        kernel_func = functools.partial(kernel_func, u.bohr_radius, u.electron_mass, u.hbar)

        return kernel_func

    def info(self):
        info = super().info()

        info.add_field('Bound State Energy', f'{u.uround(self.bound_state_energy, u.eV)} eV | {u.uround(self.bound_state_energy, u.hartree)} hartree')

        return info


class ApproximateLengthGaugeHydrogenKernelWithContinuumContinuumInteraction(LengthGaugeHydrogenKernel):
    """
    The kernel for the hydrogen ground state with plane wave continuum states.
    This version uses an approximation of the continuum-continuum interaction, including only the A^2 phase factor.
    """

    def __init__(self, bound_state_energy = states.HydrogenBoundState(1).energy):
        super().__init__(bound_state_energy = bound_state_energy)

        self.phase_prefactor = (u.electron_charge ** 2) / (2 * u.electron_mass * u.hbar)

    def __call__(self, current_time, previous_time, electric_potential, vector_potential):
        kernel = super().__call__(current_time, previous_time, electric_potential, vector_potential)

        return kernel * self._vector_potential_phase_factor(current_time, previous_time, vector_potential)

    def _vector_potential_phase_factor(self, current_time, previous_time, vector_potential):
        vp_previous = vector_potential(previous_time)

        time_difference = current_time - previous_time

        vp_integral = -integ.cumtrapz(
            y = vector_potential(previous_time)[::-1],
            x = previous_time[::-1],
            initial = 0,
        )[::-1]

        vp2_integral = integ.cumtrapz(
            y = vector_potential(previous_time)[::-1] ** 2,
            x = previous_time[::-1],
            initial = 0,
        )[::-1]

        integral = vp2_integral - (2 * vp_previous * vp_integral) + ((vp_previous ** 2) * time_difference)

        return np.exp(-1j * self.phase_prefactor * integral)
