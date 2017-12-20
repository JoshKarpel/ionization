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
    def __init__(self, theta_p_points = 200):
        super().__init__()

        self.bound_state_energy = states.HydrogenBoundState(1).energy
        self.omega_bound = self.bound_state_energy / u.hbar

        self.x_conversion = u.bohr_radius / u.hbar

        self.matrix_element_prefactor = 1j * np.sqrt(512 * 3 * (u.bohr_radius ** 7) / (3 * u.pi * 4 * u.pi * (u.hbar ** 5)))
        self.phase_integral_prefactor = 1 / (2 * u.electron_mass * u.hbar)

        self.p = np.linspace(0, 20, 500) * u.atomic_momentum
        self.theta_p = np.linspace(0, u.pi, theta_p_points)

        self.p_mesh, self.theta_p_mesh = np.meshgrid(self.p, self.theta_p, indexing = 'ij')

    def __call__(self, current_time, previous_time, electric_potential, vector_potential):
        raise NotImplementedError

    def z_dipole_matrix_element_per_momentum(self, p, theta_p):
        return self.matrix_element_prefactor * np.cos(theta_p) * p / (1 + ((p * u.bohr_radius / u.hbar) ** 2)) ** 3

    def kernel(self, current_time, previous_time, vector_potential):
        warnings.warn('MATRIX ELEMENT DOESNT HAVE qA in it yet')
        # PREVIOUS TIME IS A VECTOR

        kernel = np.empty_like(previous_time, dtype = np.complex128)
        for t_prime_index, t_prime in previous_time:
            integrand_vs_p = self.integrand_vs_p(current_time, previous_time, vector_potential)

            kernel[t_prime_index] = integ.simps(
                y = integrand_vs_p,
                x = self.p
            )

        return kernel

    def integrand_vs_p(self, current_time, previous_time, vector_potential):
        integrand_vs_p = np.empty_like(self.p, dtype = np.complex128)

        for p_index, p in enumerate(self.p):
            integrand_for_p_vs_theta_p = np.empty_like(self.theta_p, dtype = np.complex128)

            first_matrix_element = self.z_dipole_matrix_element_per_momentum(p, self.theta_p)
            second_matrix_element = self.z_dipole_matrix_element_per_momentum(p, self.theta_p)  # eventually depends on A(t, t')

            matrix_elements = np.conj(first_matrix_element) * second_matrix_element

            for theta_p_index, theta_p in enumerate(self.theta_p):
                # print('theta_p', theta_p_index, theta_p / u.pi)
                phase_factor = self.phase_factor(p, theta_p, current_time, previous_time, vector_potential)  # very inefficient

                integrand_for_p_vs_theta_p[theta_p_index] = matrix_elements[theta_p_index] * phase_factor * np.sin(theta_p)

            # twopi from phi integral
            integrand_vs_p[p_index] = u.twopi * (p ** 2) * integ.simps(
                y = integrand_for_p_vs_theta_p,
                x = self.theta_p
            )

        return integrand_vs_p

    def phase_factor(self, p, theta_p, current_time, previous_time, vector_potential):
        """VECTORIZED IN PREVIOUS TIME"""
        return np.exp(-1j * self.phase_integral(p, theta_p, current_time, previous_time, vector_potential))

    # REMEMBER: p and previous_time might both be arrays!

    def phase_integral(self, p, theta_p, current_time, previous_time, vector_potential):
        """VECTORIZED IN PREVIOUS TIME"""
        # print(self.phase_integrand(p, theta_p, current_time, previous_time, vector_potential))
        # integral = -integ.cumtrapz(
        #     y = self.phase_integrand(p, theta_p, current_time, previous_time, vector_potential)[::-1],
        #     x = previous_time[::-1],
        #     initial = 0,
        # )

        # print(vector_potential.x)
        integral, *err = integ.quad(
            lambda t: self.phase_integrand(p, theta_p, current_time, t, vector_potential),
            previous_time,
            current_time,
            # points = vector_potential.x
        )

        return np.mod(self.phase_integral_prefactor * integral, u.twopi)

    def phase_integrand(self, p, theta_p, current_time, previous_time, vector_potential):
        vp_diff = vector_potential(current_time) - vector_potential(previous_time)
        return (p ** 2) + (vp_diff ** 2) + (2 * p * np.cos(theta_p) * vp_diff)
