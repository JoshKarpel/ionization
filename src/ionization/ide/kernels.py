import logging
from abc import ABC, abstractmethod
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

    def __init__(self):
        self.bound_state_energy = states.HydrogenBoundState(1).energy
        self.omega_bound = self.bound_state_energy / u.hbar

        self.kernel_prefactor = (512 / (3 * u.pi)) * (u.bohr_radius ** 7)
        self.kernel_at_time_difference_zero_with_prefactor = u.bohr_radius ** 2

    def __call__(self, time_current, time_previous, electric_potential, vector_potential):
        time_difference = time_current - time_previous

        return self._evaluate_kernel_function(time_difference)

    def _evaluate_kernel_function(self, time_difference):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            kernel = self.kernel_prefactor * np.exp(1j * self.omega_bound * time_difference) * self.kernel_function(time_difference)

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
        kernel = kernel.subs([
            (a, u.bohr_radius),
            (m, u.electron_mass),
            (hb, u.hbar),
        ])
        kernel = kernel.evalf()  # partial evaluation of coefficients

        kernel_func = sym.lambdify(
            (td,),
            kernel,
            modules = ['numpy', {'erfc': special.erfc}]
        )

        return kernel_func

    def info(self):
        info = super().info()

        info.add_field('Bound State Energy', f'{u.uround(self.bound_state_energy, u.eV)} eV | {u.uround(self.bound_state_energy, u.hartree)} hartree')

        return info


class ApproximateLengthGaugeHydrogenKernelWithContinuumContinuumInteraction(LengthGaugeHydrogenKernel):
    """
    The kernel for the hydrogen ground state with plane wave continuum states.
    This version adds an approximation of the continuum-continuum interaction, including only the A^2 phase factor.
    """

    def __init__(self):
        super().__init__()

        self.phase_prefactor = (u.electron_charge ** 2) / (2 * u.electron_mass * u.hbar)

    def __call__(self, current_time, previous_time, electric_potential, vector_potential):
        kernel = super().__call__(current_time, previous_time, electric_potential, vector_potential)

        return kernel * self._vector_potential_phase_factor(current_time, previous_time, vector_potential)

    def _vector_potential_phase_factor(self, current_time, previous_time, vector_potential):
        return np.exp(-1j * self._vector_potential_phase_factor_integral(current_time, previous_time, vector_potential))

    def _vector_potential_phase_factor_integral(self, current_time, previous_time, vector_potential):
        vp_current = vector_potential(current_time)
        time_difference = current_time - previous_time

        vp_integral = self._integrate_vector_potential_over_previous_times(previous_time, vector_potential, power = 1)
        vp2_integral = self._integrate_vector_potential_over_previous_times(previous_time, vector_potential, power = 2)

        integral = ((vp_current ** 2) * time_difference) - (2 * vp_current * vp_integral) + vp2_integral

        return np.mod(self.phase_prefactor * integral, u.twopi)

    def _integrate_vector_potential_over_previous_times(self, previous_time, vector_potential, power = 1):
        # strategy: flip integrals around in time (adds negative sign), then unflip the resulting array to get the desired integrals
        return -integ.cumtrapz(
            y = vector_potential(previous_time)[::-1] ** power,
            x = previous_time[::-1],
            initial = 0,
        )[::-1]


class FullHydrogenKernel(Kernel):
    def __init__(self):
        super().__init__()

        self.bound_state_energy = states.HydrogenBoundState(1).energy
        self.omega_bound = self.bound_state_energy / u.hbar

        self.x_conversion = u.bohr_radius / u.hbar

        self.matrix_element_prefactor = 1  # get this right
        self.phase_integral_prefactor = -1j / (2 * u.electron_mass * u.hbar)

    def __call__(self, current_time, previous_time, electric_potential, vector_potential):
        raise NotImplementedError

    def x(self, p):
        return self.x_conversion * p

    def z_dipole_matrix_element(self, p, theta_p):
        x2 = (self.x(p)) ** 2
        return self.matrix_element_prefactor * np.cos(theta_p) * x2 / ((1 + x2) ** 3)

    def phase_factor(self, p, theta_p, current_time, previous_time, vector_potential):
        return np.exp(self.phase_integral(p, theta_p, current_time, previous_time, vector_potential) % u.twopi)

    def phase_integral(self, p, theta_p, current_time, previous_time, vector_potential):
        integral = -integ.cumtrapz(
            y = self.phase_integrand(p, theta_p, current_time, previous_time, vector_potential)[::-1],
            x = previous_time[::-1],
            initial = 0,
        )

        return self.phase_integral_prefactor * integral

    def phase_integrand(self, p, theta_p, current_time, previous_time, vector_potential):
        vp_diff = vector_potential(current_time) - vector_potential(previous_time)
        return (p ** 2) + (vp_diff ** 2) + (2 * p * np.cos(theta_p) * vp_diff)
