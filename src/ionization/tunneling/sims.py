import logging
from typing import Callable

from tqdm import tqdm

import numpy as np

import simulacra as si
import simulacra.units as u

from .. import potentials, utils
from . import models

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TunnelingSimulation(si.Simulation):
    """
    A :class:`simulacra.Simulation` for a wavefunction that decays based on the instantaneous electric field amplitude.

    Attributes
    ----------
    b
        The bound state wavefunction probability amplitude.
        May not be an actual probability amplitude of an actual state, depending on the model used.
    b2
        The absolute value squared of the bound state wavefunction probability amplitude (i.e., probability to remaing in the bound state).
        May not be an actual probability of an actual state, depending on the model used.
    """

    def __init__(self, spec: 'TunnelingSpecification'):
        super().__init__(spec)

        self.spec = spec

        total_time = self.spec.time_final - self.spec.time_initial
        self.times = np.linspace(self.spec.time_initial, self.spec.time_final, int(total_time / self.spec.time_step) + 1)

        self.time_steps = len(self.times)
        self.time_index = 0

        if self.spec.electric_potential_dc_correction:
            old_pot = self.spec.electric_potential
            self.spec.electric_potential = potentials.DC_correct_electric_potential(self.spec.electric_potential, self.times)

            logger.warning('Replaced electric potential {} --> {} for {} {}'.format(old_pot, self.spec.electric_potential, self.__class__.__name__, self.name))

        self.b = np.empty_like(self.times)
        self.b.fill(np.NaN)
        self.b[0] = self.spec.b_initial

    @property
    def time(self):
        return self.times[self.time_index]

    @property
    def b2(self):
        return np.abs(self.b) ** 2

    def run(
        self,
        progress_bar: bool = False,
        callback: Callable[['TunnelingSimulation'], None] = None,
    ):
        """
        Run the simulation by repeatedly decaying the wavefunction in increments of the time step.

        Parameters
        ----------
        progress_bar
            If ``True``, a progress bar will be displayed.
        callback
            If given, will be called with the :class:`TunnelingSimulation` as its argument after every time step.
        """

        self.status = si.Status.RUNNING

        if progress_bar:
            pbar = tqdm(total = self.time_steps - 1, ascii = True, ncols = 80)

        while True:
            if callback is not None:
                callback(self)

            if self.time_index == self.time_steps - 1:
                break

            self.time_index += 1

            dt = self.times[self.time_index] - self.times[self.time_index - 1]
            efield = self.spec.electric_potential.get_electric_field_amplitude(self.time + (dt / 2))
            tunneling_rate = self.spec.tunneling_model.tunneling_rate(efield, self.spec.ionization_potential)
            self.b[self.time_index] = self.b[self.time_index - 1] * np.exp(tunneling_rate * dt)

            logger.debug(f'{self.__class__.__name__} {self.name} ({self.file_name}) evolved to time index {self.time_index} / {self.time_steps - 1} ({np.around(100 * (self.time_index + 1) / self.time_steps, 2)}%)')

            try:
                pbar.update(1)
            except NameError:
                pass

        try:
            pbar.close()
        except NameError:
            pass

        self.status = si.Status.FINISHED
        logger.info(f'Finished performing time evolution on {self.name} ({self.file_name})')


class TunnelingSpecification(si.Specification):
    """A :class:`simulacra.Specification` for a :class:`simulacra.Simulation` based on a :class:`TunnelingModel`."""

    simulation_type = TunnelingSimulation

    def __init__(
        self,
        name: str,
        time_initial: float = 0 * u.asec,
        time_final: float = 200 * u.asec,
        time_step: float = 1 * u.asec,
        electric_potential = potentials.NoElectricPotential(),
        electric_potential_dc_correction = False,
        tunneling_model: models.TunnelingModel = models.LandauRate(),
        ionization_potential: float = -u.rydberg,
        b_initial: complex = 1,
        **kwargs,
    ):
        """
        Parameters
        ----------
        name
            The name of the specification/simulation.
        time_initial
            The time to begin the simulation at.
        time_final
            The time to end the simulation at.
        time_step
            The amount of time to evolve by on each evolution step.
        electric_potential
            The possibly-time varying external electric field.
        electric_potential_dc_correction
            If ``True``, perform DC correction on the ``electric_potential``.
        tunneling_model
            The :class:`TunnelingModel` to use for the simulation.
        ionization_potential
            The ionization potential of the bound state.
        b_initial
            The initial probability amplitude.
        kwargs
            Any additional keyword arguments are passed to the :class:`simulacra.Specification` constructor.
        """
        super().__init__(name, **kwargs)

        self.time_initial = time_initial
        self.time_final = time_final
        self.time_step = time_step

        self.electric_potential = electric_potential
        self.electric_potential_dc_correction = electric_potential_dc_correction

        self.tunneling_model = tunneling_model

        self.ionization_potential = ionization_potential
        self.b_initial = b_initial

    def info(self) -> si.Info:
        info = super().info()

        info_evolution = si.Info(header = 'Time Evolution')
        info_evolution.add_field('Initial Time', utils.fmt_quantity(self.time_initial, utils.TIME_UNITS))
        info_evolution.add_field('Final Time', utils.fmt_quantity(self.time_final, utils.TIME_UNITS))
        info_evolution.add_field('Time Step', utils.fmt_quantity(self.time_step, utils.TIME_UNITS))

        info.add_info(info_evolution)

        info.add_info(self.tunneling_model.info())

        info.add_field('Initial b', self.b_initial)

        info_fields = si.Info(header = 'Electric Fields')
        info_fields.add_field('DC Correct Electric Field', 'yes' if self.electric_potential_dc_correction else 'no')
        for x in self.electric_potential:
            info_fields.add_info(x.info())
        info.add_info(info_fields)

        return info
