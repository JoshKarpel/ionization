import logging
from abc import ABC, abstractmethod
import warnings

import numpy as np
import scipy.integrate as integ
import scipy.special as special
import sympy as sym

import simulacra as si
import simulacra.units as u

from .. import states, utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Kernel(ABC):
    """
    An abstract class that represents an IDE model kernel.
    """

    @abstractmethod
    def __call__(self, current_time, previous_time, electric_potential, vector_potential):
        """
        Parameters
        ----------
        current_time
            The current time (i.e., :math:`t`).
        previous_time
            The previous time (i.e., :math:`t'`).
        electric_potential
            The electric potential used in the simulation.
        vector_potential
            The interpolated vector potential of the electric potential.

        Returns
        -------
            The value of the kernel :math:`K(t, t'; E, A)`.
        """
        raise NotImplementedError

    def __repr__(self):
        return utils.fmt_fields(self)

    def info(self) -> si.Info:
        info = si.Info(header = self.__class__.__name__)

        return info


class LengthGaugeHydrogenKernel(Kernel):
    """
    The kernel for the hydrogen ground state with spherical Bessel functions, with no continuum-continuum coupling, in the length gauge.
    """

    def __init__(self, bound_state_energy = states.HydrogenBoundState(1).energy):
        self.bound_state_energy = bound_state_energy
        self.omega_bound = bound_state_energy / u.hbar

        self.kernel_prefactor = (512 / (3 * u.pi)) * (u.bohr_radius ** 7)
        self.kernel_at_time_difference_zero_with_prefactor = u.bohr_radius ** 2

    def __call__(self, time_current, time_previous, electric_potential, vector_potential):
        time_difference = time_current - time_previous

        return self.evaluate_kernel_function(time_difference)

    def evaluate_kernel_function(self, time_difference: float) -> float:
        """
        Evaluate the approximate kernel function at a time different or over some array of time differences.

        Parameters
        ----------
        time_difference
            The time difference :math:`t-t'`.

        Returns
        -------
        kernel
            The value of the approximate kernel function at the given time difference.
        """
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
        """The approximate kernel function :math:`K(t - t')`."""
        k = sym.Symbol('wavenumber', real = True)
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

    def __repr__(self):
        return utils.fmt_fields(self, 'bound_state_energy')

    def info(self) -> si.Info:
        info = super().info()

        info.add_field('Bound State Energy', f'{u.uround(self.bound_state_energy, u.eV)} eV | {u.uround(self.bound_state_energy, u.hartree)} hartree')

        return info


class LengthGaugeHydrogenKernelWithContinuumContinuumInteraction(LengthGaugeHydrogenKernel):
    """
    The kernel for the hydrogen ground state with plane wave continuum states.
    This version adds an approximation of the continuum-continuum interaction, including only the :math:`\\mathcal{A}^2` phase factor.
    """

    def __init__(self, bound_state_energy = states.HydrogenBoundState(1).energy):
        super().__init__(bound_state_energy = bound_state_energy)

        self.phase_prefactor = -1j * (u.electron_charge ** 2) / (2 * u.electron_mass * u.hbar)

    def __call__(self, current_time, previous_time, electric_potential, vector_potential):
        kernel = super().__call__(current_time, previous_time, electric_potential, vector_potential)

        return kernel * self._vector_potential_phase_factor(current_time, previous_time, vector_potential)

    def _vector_potential_phase_factor(self, current_time, previous_time, vector_potential):
        vp_current = vector_potential(current_time)
        time_difference = current_time - previous_time

        vp_integral = self._integrate_vector_potential_over_previous_times(previous_time, vector_potential, power = 1)
        vp2_integral = self._integrate_vector_potential_over_previous_times(previous_time, vector_potential, power = 2)

        integral = ((vp_current ** 2) * time_difference) - (2 * vp_current * vp_integral) + vp2_integral
        integral %= u.twopi  # it's a phase, so we're free to do this, which should make the exp more stable

        return np.exp(self.phase_prefactor * integral)

    def _integrate_vector_potential_over_previous_times(self, previous_time, vector_potential, power = 1):
        # strategy: flip integrals around in time (adds negative sign), then unflip the resulting array to get the desired integrals
        return -integ.cumtrapz(
            y = vector_potential(previous_time)[::-1] ** power,
            x = previous_time[::-1],
            initial = 0,
        )[::-1]
