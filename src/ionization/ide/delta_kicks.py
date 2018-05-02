import logging
import collections
from typing import Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp

import simulacra as si
import simulacra.units as u

from .. import core, potentials, states, vis

from . import kernels

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

delta_kick = collections.namedtuple('delta_kick', ['time', 'amplitude'])


class DeltaKicks(potentials.PotentialEnergy):
    def __init__(self, kicks):
        super().__init__()

        self.kicks = kicks

    def __iter__(self):
        yield from self.kicks

    def __getitem__(self, item):
        return self.kicks[item]

    def __len__(self):
        return len(self.kicks)

    def __call__(self, *args, **kwargs):
        raise ValueError('DeltaKicks potential cannot be evaluated')

    def info(self) -> si.Info:
        info = super().info()

        info.add_field('Number of Kicks', len(self))
        info.add_field('Maximum Kick Amplitude', max(k.amplitude for k in self.kicks))

        return info


def decompose_potential_into_kicks(
    electric_potential: potentials.ElectricPotential,
    times: np.array,
) -> DeltaKicks:
    """
    Decomposes an electric potential into a series of delta-function kicks.

    Parameters
    ----------
    electric_potential
        The electric potentials to decompose.
    times
        The times to consider when performing the decomposition.

    Returns
    -------
    delta_kicks
        The decomposed electric potential.
    """
    efield_vs_time = electric_potential.get_electric_field_amplitude(times)
    signs = np.sign(efield_vs_time)

    # state machine
    kicks = []
    current_sign = signs[0]
    efield_accumulator = 0
    prev_time = times[0]
    max_field = 0
    max_field_time = 0
    for efield, sign, time in zip(efield_vs_time, signs, times):
        if sign == current_sign and time != times[-1]:
            efield_accumulator += efield * (time - prev_time)
            if max_field < np.abs(efield):
                max_field = np.abs(efield)
                max_field_time = time
        else:
            kicks.append(delta_kick(time = max_field_time, amplitude = efield_accumulator))

            # reset
            current_sign = sign
            efield_accumulator = 0
            max_field = 0
        prev_time = time

    return DeltaKicks(kicks)


class DeltaKickSimulation(si.Simulation):
    """
    A :class:`simulacra.Simulation` that implements the delta-kick model.

    Attributes
    ----------
    data_times
        The times that the simulation calculated data for.
    b
        The probability amplitude of the bound state at the ``data_times``.
    b2
        The absolute value squared of the probability amplitude of the bound state.
    """

    def __init__(self, spec):
        super().__init__(spec)

        total_time = self.spec.time_final - self.spec.time_initial
        self.times = np.linspace(self.spec.time_initial, self.spec.time_final, int(total_time / self.spec.time_step) + 1)
        self.time_index = 0
        self.time_step = self.spec.time_step

        if self.spec.electric_potential_dc_correction and not isinstance(self.spec.electric_potential, DeltaKicks):
            old_pot = self.spec.electric_potential
            self.spec.electric_potential = potentials.DC_correct_electric_potential(self.spec.electric_potential, self.times)

            logger.warning(f'Replaced electric potential {old_pot} --> {self.spec.electric_potential} for {self}')

        if not isinstance(self.spec.electric_potential, DeltaKicks):
            self.spec.kicks = decompose_potential_into_kicks(self.spec.electric_potential, self.times)
        else:
            self.spec.kicks = self.spec.electric_potential

        self.data_times = np.array([self.spec.time_initial] + [k.time for k in self.spec.kicks] + [self.spec.time_final])  # for consistency with other simulations

        self.b = np.empty(len(self.spec.kicks) + 2, dtype = np.complex128) * np.NaN
        self.b[0] = self.spec.b_initial

    @property
    def time_steps(self):
        return len(self.times)

    @property
    def time(self):
        return self.times[self.time_index]

    @property
    def b2(self):
        return np.abs(self.b) ** 2

    def evaluate_kernel(self, time_current, time_previous):
        """
        Calculate the values of the IDE kernel as a function of the time difference.

        Parameters
        ----------
        time_current
        time_previous

        Returns
        -------
        kernel :
            The value of the kernel at the time differences.
        """
        return self.spec.kernel(
            time_current,
            time_previous,
            self.spec.electric_potential,
            None,  # no vector potential, will make some kernels explode
        )

    def _solve(self):
        amplitudes = np.array([kick.amplitude for kick in self.spec.kicks])
        k0 = self.evaluate_kernel(0, 0)

        for t_idx, kick in enumerate(self.spec.kicks, start = 1):
            history_sum = (self.spec.integral_prefactor * kick.amplitude) * np.sum(amplitudes[:t_idx - 1] * self.evaluate_kernel(self.data_times[t_idx], self.data_times[1:t_idx]) * self.b[1:t_idx])
            self.b[t_idx] = (self.b[t_idx - 1] + history_sum) / (1 - (self.spec.integral_prefactor * k0 * (kick.amplitude ** 2)))

        self.b[-1] = self.b[-2]

    def run(self, callback = None):
        """
        Run the simulation by repeatedly evolving it forward in time.

        Parameters
        ----------
        callback : callable
            A function that accepts the ``Simulation`` as an argument, called at the end of each time step.
        """
        logger.info(f'Performing time evolution on {self.name} ({self.file_name}), starting from time index {self.time_index}')
        self.status = si.Status.RUNNING

        self._solve()

        if callback is not None:
            callback(self)

        self.status = si.Status.FINISHED
        logger.info(f'Finished performing time evolution on {self.name} ({self.file_name})')

    def attach_electric_potential_plot_to_axis(
        self,
        axis,
        time_unit = 'asec',
        show_electric_field = True,
        overlay_kicks = True):
        time_unit_value, time_unit_latex = u.get_unit_value_and_latex_from_unit(time_unit)

        if show_electric_field and not isinstance(self.spec.electric_potential, DeltaKicks):
            axis.plot(
                self.times / time_unit_value,
                self.spec.electric_potential.get_electric_field_amplitude(self.times) / u.atomic_electric_field,
                color = vis.COLOR_EFIELD,
                linewidth = 1.5,
            )

            if overlay_kicks:
                for kick in self.spec.kicks:
                    axis.plot(
                        [kick.time / time_unit_value, kick.time / time_unit_value],
                        [0, self.spec.electric_potential.get_electric_field_amplitude(kick.time) / u.atomic_electric_field],
                        linewidth = 1.5,
                        color = si.vis.PINK,
                    )

            axis.set_ylabel(rf'$ {vis.LATEX_EFIELD}(t) $')
        else:
            for kick in self.spec.kicks:
                axis.plot(
                    [kick.time / time_unit_value, kick.time / time_unit_value],
                    [0, kick.amplitude / (u.atomic_electric_field * u.atomic_time)],
                    linewidth = 1.5,
                    color = si.vis.PINK,
                )

            axis.set_ylabel(r'$ \eta $')

        axis.set_xlabel('Time $t$ (${}$)'.format(time_unit_latex), fontsize = 13)

        axis.tick_params(labelright = True)

        axis.set_xlim(self.times[0] / time_unit_value, self.times[-1] / time_unit_value)

        axis.grid(True, **si.vis.GRID_KWARGS)

    def plot_wavefunction_vs_time(self, *args, **kwargs):
        """Alias for plot_a2_vs_time."""
        self.plot_b2_vs_time(*args, **kwargs)

    def plot_b2_vs_time(
        self,
        log = False,
        show_electric_field = True,
        overlay_kicks = True,
        time_unit = 'asec',
        show_title = False,
        y_lower_limit = 0,
        y_upper_limit = 1,
        **kwargs):
        with si.vis.FigureManager(self.file_name + '__b2_vs_time', **kwargs) as figman:
            fig = figman.fig

            t_scale_unit, t_scale_name = u.get_unit_value_and_latex_from_unit(time_unit)

            grid_spec = matplotlib.gridspec.GridSpec(2, 1, height_ratios = [4, 1], hspace = 0.07)  # TODO: switch to fixed axis construction
            ax_b = plt.subplot(grid_spec[0])
            ax_pot = plt.subplot(grid_spec[1], sharex = ax_b)

            self.attach_electric_potential_plot_to_axis(
                ax_pot,
                show_electric_field = show_electric_field,
                overlay_kicks = overlay_kicks,
                time_unit = time_unit,
            )

            # the repeats produce the stair-step pattern
            ax_b.plot(
                np.repeat(self.data_times, 2)[1:-1] / t_scale_unit,
                np.repeat(self.b2, 2)[:-2],
                marker = 'o',
                markersize = 2,
                markevery = 2,
                linestyle = ':',
                color = 'black',
                linewidth = 1,
            )

            if log:
                ax_b.set_yscale('log')
                min_overlap = np.min(self.b2)
                ax_b.set_ylim(bottom = max(1e-9, min_overlap * .1), top = 1.0)
                ax_b.grid(True, which = 'both', **si.vis.GRID_KWARGS)
            else:
                ax_b.set_ylim(0.0, 1.0)
                ax_b.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                ax_b.grid(True, **si.vis.GRID_KWARGS)

            ax_b.set_xlim(self.spec.time_initial / t_scale_unit, self.spec.time_final / t_scale_unit)
            ax_b.set_ylim(y_lower_limit, y_upper_limit)

            ax_b.set_ylabel(r'$\left| b(t) \right|^2$', fontsize = 13)

            # Find at most n+1 ticks on the y-axis at 'nice' locations
            max_yticks = 4
            yloc = plt.MaxNLocator(max_yticks, prune = 'upper')
            ax_pot.yaxis.set_major_locator(yloc)

            max_xticks = 6
            xloc = plt.MaxNLocator(max_xticks, prune = 'both')
            ax_pot.xaxis.set_major_locator(xloc)

            ax_pot.tick_params(axis = 'x', which = 'major', labelsize = 10)
            ax_pot.tick_params(axis = 'y', which = 'major', labelsize = 10)
            ax_b.tick_params(axis = 'both', which = 'major', labelsize = 10)

            ax_b.tick_params(
                labelleft = True,
                labelright = True,
                labeltop = True,
                labelbottom = False,
                bottom = True,
                top = True,
                left = True,
                right = True,
            )
            ax_pot.tick_params(
                labelleft = True,
                labelright = True,
                labeltop = False,
                labelbottom = True,
                bottom = True,
                top = True,
                left = True,
                right = True,
            )

            if show_title:
                title = ax_b.set_title(self.name)
                title.set_y(1.15)

            postfix = ''
            if log:
                postfix += '__log'

            figman.name += postfix

    def info(self) -> si.Info:
        info = super().info()

        info.add_info(self.spec.info())

        return info


class DeltaKickSpecification(si.Specification):
    """
    A Specification for an :class:`IntegroDifferentialEquationSimulation`.
    """
    simulation_type = DeltaKickSimulation

    def __init__(
        self,
        name,
        time_initial = 0 * u.asec,
        time_final = 200 * u.asec,
        time_step = 1 * u.asec,
        test_mass: float = u.electron_mass,
        test_charge: float = u.electron_charge,
        b_initial: Union[float, complex] = 1,
        integral_prefactor: float = -(u.electron_charge / u.hbar) ** 2,
        electric_potential: potentials.PotentialEnergy = potentials.NoElectricPotential(),
        electric_potential_dc_correction: bool = True,
        kernel: kernels.Kernel = kernels.LengthGaugeHydrogenKernel(),
        evolution_gauge: core.Gauge = core.Gauge.LENGTH,
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
        test_mass
            The mass of the test particle.
        test_charge
            The charge of the test particle.
        b_initial
            The initial bound state amplitude.
        integral_prefactor
            The prefactor of the integral term of the IDE.
        electric_potential
            The possibly-time varying external electric field.
        electric_potential_dc_correction
            If ``True``, perform DC correction on the ``electric_potential``.
        kernel
            The :class:`Kernel` to use for the simulation.
        evolution_gauge
            The :class:`Gauge` to work in.
        kwargs
            Any additional keyword arguments are passed to the :class:`simulacra.Specification` constructor.
        """
        super().__init__(name, **kwargs)

        self.time_initial = time_initial
        self.time_final = time_final
        self.time_step = time_step

        self.test_mass = test_mass
        self.test_charge = test_charge

        self.b_initial = b_initial

        self.integral_prefactor = integral_prefactor

        self.electric_potential = electric_potential
        self.electric_potential_dc_correction = electric_potential_dc_correction

        self.kernel = kernel

        self.evolution_gauge = evolution_gauge

    @property
    def test_omega(self):
        return self.test_energy / u.hbar

    @property
    def test_frequency(self):
        return self.test_omega / u.twopi

    def info(self) -> si.Info:
        info = super().info()

        info_evolution = si.Info(header = 'Time Evolution')
        info_evolution.add_field('Initial Time', f'{u.uround(self.time_initial, u.asec, 3)} as')
        info_evolution.add_field('Final Time', f'{u.uround(self.time_final, u.asec, 3)} as')

        info.add_info(info_evolution)

        info_algorithm = si.Info(header = 'Evolution Algorithm')
        info_algorithm.add_field('Evolution Gauge', self.evolution_gauge)

        info.add_info(info_algorithm)

        info_ide = si.Info(header = 'IDE Parameters')
        info_ide.add_field('Initial State', f'a = {self.b_initial}')
        info_ide.add_field('Prefactor', self.integral_prefactor)
        info_ide.add_info(self.kernel.info())
        info_ide.add_field('DC Correct Electric Field', 'yes' if self.electric_potential_dc_correction else 'no')

        info_potential = self.electric_potential.info()
        info_potential.header = 'Electric Potential: ' + info_potential.header
        info_ide.add_info(info_potential)

        info.add_info(info_ide)

        return info
