import logging
import itertools
from typing import List

import numpy as np
import scipy.optimize as optimize
import scipy.special as special
import scipy.misc as spmisc

import simulacra as si
import simulacra.units as u

from .. import core, mesh, potentials, utils, exceptions

from . import state

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TwoDPlaneWave(state.QuantumState):
    """
    A plane wave in three dimensions.
    """

    eigenvalues = state.Eigenvalues.CONTINUOUS
    binding = state.Binding.FREE
    derivation = state.Derivation.ANALYTIC

    def __init__(
        self,
        wavenumber_x: float,
        wavenumber_z: float,
        amplitude: state.ProbabilityAmplitude = 1,
    ):
        """
        Parameters
        ----------
        wavenumber_x
            The :math:`x`-component of the state's wavevector.
        wavenumber_z
            The :math:`z`-component of the state's wavevector.
        """
        self.wavenumber_x = wavenumber_x
        self.wavenumber_z = wavenumber_z

        super().__init__(amplitude = amplitude)

    @property
    def wavenumber(self):
        return np.sqrt(self.wavenumber_x ** 2 + self.wavenumber_z ** 2)

    @property
    def energy(self):
        return core.electron_energy_from_wavenumber(self.wavenumber)

    @property
    def tuple(self):
        return self.wavenumber_x, self.wavenumber_z

    def eval_z_x(self, z, x):
        return np.exp(1j * self.wavenumber_x * x) * np.exp(1j * self.wavenumber_z * z) / u.pi

    def eval_r_theta(self, r, theta):
        z = r * np.cos(theta)
        x = r * np.sin(theta)

        return self.eval_z_x(z, x)

    def __call__(self, r, theta):
        return self.eval_r_theta(r, theta)

    def __repr__(self):
        return utils.fmt_fields(self, 'wavenumber_x', 'wavenumber_y', 'wavenumber_z')

    @property
    def ket(self):
        return f'{state.fmt_amplitude(self.amplitude)}|kx={u.uround(self.wavenumber_x, u.per_nm)} 1/nm, kz={u.uround(self.wavenumber_z, u.per_nm)} 1/nm>'

    @property
    def tex(self):
        return fr'{state.fmt_amplitude_for_tex(self.amplitude)}\phi_{{k_x={u.uround(self.wavenumber_x, u.per_nm)} \, \mathrm{{nm^-1}}, k_x={u.uround(self.wavenumber_z, u.per_nm)} \, \mathrm{{nm^-1}}}}'

    @property
    def tex_ket(self):
        return fr'{state.fmt_amplitude_for_tex(self.amplitude)}\left| \phi_{{k_x={u.uround(self.wavenumber_x, u.per_nm)} \, \mathrm{{nm^-1}}, k_x={u.uround(self.wavenumber_z, u.per_nm)} \, \mathrm{{nm^-1}}}} \right\rangle'

    def info(self) -> si.Info:
        info = super().info()

        info.add_field('X Wavenumber', utils.fmt_quantity(self.wavenumber, utils.INVERSE_LENGTH_UNITS))
        info.add_field('Y Wavenumber', utils.fmt_quantity(self.wavenumber, utils.INVERSE_LENGTH_UNITS))
        info.add_field('Energy', utils.fmt_quantity(self.energy, utils.ENERGY_UNITS))
        info.add_field('Wavenumber', utils.fmt_quantity(self.wavenumber, utils.INVERSE_LENGTH_UNITS))

        return info
