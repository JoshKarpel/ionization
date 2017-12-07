import logging
import collections

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import simulacra as si
import simulacra.units as u

from .. import core, potentials, states

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

    def info(self):
        info = super().info()

        info.add_field('Number of Kicks', len(self))
        info.add_field('Maximum Kick Amplitude', max(k.amplitude for k in self.kicks))

        return info


def decompose_potential_into_kicks(electric_potential, times):
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
    def __init__(self, spec):
        super().__init__(spec)

        total_time = self.spec.time_final - self.spec.time_initial
        self.times = np.linspace(self.spec.time_initial, self.spec.time_final, int(total_time / self.spec.time_step) + 1)
        self.time_index = 0
        self.time_step = self.spec.time_step

        if self.spec.electric_potential_dc_correction and not isinstance(self.spec.electric_potential, DeltaKicks):
            dummy_times = np.linspace(self.spec.time_initial, self.spec.time_final, self.times)
            old_pot = self.spec.electric_potential
            self.spec.electric_potential = potentials.DC_correct_electric_potential(self.spec.electric_potential, dummy_times)

            logger.warning('Replaced electric potential {} --> {} for {} {}'.format(old_pot, self.spec.electric_potential, self.__class__.__name__, self.name))

        if not isinstance(self.spec.electric_potential, DeltaKicks):
            self.spec.kicks = decompose_potential_into_kicks(self.spec.electric_potential, self.times)
        else:
            self.spec.kicks = self.spec.electric_potential

        self.data_times = np.array([self.spec.time_initial] + [k.time for k in self.spec.kicks] + [self.spec.time_final])  # for consistency with other simulations

        self.b = np.empty(len(self.spec.kicks) + 2, dtype = np.complex128) * np.NaN
        self.b[0] = 1

    @property
    def time_steps(self):
        return len(self.times)

    @property
    def time(self):
        return self.times[self.time_index]

    @property
    def b2(self):
        return np.abs(self.b) ** 2

    def eval_kernel(self, time_difference):
        """
        Calculate the values of the IDE kernel as a function of the time difference.

        Parameters
        ----------
        time_difference
            The time differences to evaluate the kernel at.
        quiver_difference
            The quiver motion differences to evaluate the kernel at. Only used for velocity-gauge kernels.

        Returns
        -------
        kernel :
            The value of the kernel at the time differences.
        """
        return self.spec.kernel(time_difference, **self.spec.kernel_kwargs)

    def _solve(self):
        amplitudes = np.array([kick.amplitude for kick in self.spec.kicks])
        k0 = self.eval_kernel(0)

        t_idx = 1
        for kick in self.spec.kicks:
            eta = kick.amplitude
            print(eta, amplitudes[:t_idx])
            print((self.data_times[t_idx] - self.data_times[1:t_idx + 1]) / u.asec)
            print(self.b[:t_idx])
            history_sum = (self.spec.integral_prefactor * eta) * np.sum(amplitudes[:t_idx - 1] * self.eval_kernel(self.data_times[t_idx] - self.data_times[1:t_idx]) * self.b[1:t_idx])
            self.b[t_idx] = (self.b[t_idx - 1] + history_sum) / (1 - (self.spec.integral_prefactor * k0 * (eta ** 2)))
            t_idx += 1

        self.b[-1] = self.b[-2]

    def run_simulation(self, callback = None):
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

    def attach_electric_potential_plot_to_axis(self,
                                               axis,
                                               time_unit = 'asec',
                                               legend_kwargs = None,
                                               show_y_label = False,
                                               show_electric_field = True,
                                               show_vector_potential = True,
                                               overlay_kicks = True):
        time_unit_value, time_unit_latex = u.get_unit_value_and_latex_from_unit(time_unit)

        if legend_kwargs is None:
            legend_kwargs = dict()
        legend_defaults = dict(
            loc = 'lower left',
            fontsize = 10,
            fancybox = True,
            framealpha = .3,
        )
        legend_kwargs = {**legend_defaults, **legend_kwargs}

        y_labels = []
        if show_electric_field:
            e_label = fr'$ {core.LATEX_EFIELD}(t) $'
            axis.plot(self.times / time_unit_value, self.spec.electric_potential.get_electric_field_amplitude(self.times) / u.atomic_electric_field,
                      color = core.COLOR_ELECTRIC_FIELD,
                      linewidth = 1.5,
                      label = e_label)
            y_labels.append(e_label)
        if show_vector_potential:
            a_label = fr'$ e \, {core.LATEX_AFIELD}(t) $'
            axis.plot(self.times / time_unit_value, u.proton_charge * self.spec.electric_potential.get_vector_potential_amplitude_numeric_cumulative(self.times) / u.atomic_momentum,
                      color = core.COLOR_VECTOR_POTENTIAL,
                      linewidth = 1.5,
                      label = a_label)
            y_labels.append(a_label)

        if show_y_label:
            axis.set_ylabel(', '.join(y_labels), fontsize = 13)

        if overlay_kicks:
            for kick in self.spec.kicks:
                axis.plot(
                    [kick.time / time_unit_value, kick.time / time_unit_value],
                    [0, self.spec.electric_potential.get_electric_field_amplitude(kick.time) / u.atomic_electric_field],
                    color = 'C2',
                    linewidth = 1.5,
                )

        axis.set_xlabel('Time $t$ (${}$)'.format(time_unit_latex), fontsize = 13)

        axis.tick_params(labelright = True)

        axis.set_xlim(self.times[0] / time_unit_value, self.times[-1] / time_unit_value)

        axis.legend(**legend_kwargs)

        axis.grid(True, **si.vis.GRID_KWARGS)

    def plot_wavefunction_vs_time(self, *args, **kwargs):
        """Alias for plot_a2_vs_time."""
        self.plot_b2_vs_time(*args, **kwargs)

    def plot_b2_vs_time(self,
                        log = False,
                        time_unit = 'asec',
                        show_vector_potential = False,
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
                show_vector_potential = show_vector_potential,
                time_unit = time_unit
            )

            ax_b.plot(self.data_times / t_scale_unit,
                      self.b2,
                      marker = 'o',
                      markersize = 2,
                      linestyle = ':',
                      color = 'black',
                      linewidth = 1)

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

            ax_b.tick_params(labelleft = True,
                             labelright = True,
                             labeltop = True,
                             labelbottom = False,
                             bottom = True,
                             top = True,
                             left = True,
                             right = True)
            ax_pot.tick_params(labelleft = True,
                               labelright = True,
                               labeltop = False,
                               labelbottom = True,
                               bottom = True,
                               top = True,
                               left = True,
                               right = True)

            if show_title:
                title = ax_b.set_title(self.name)
                title.set_y(1.15)

            postfix = ''
            if log:
                postfix += '__log'

            figman.name += postfix

    def info(self):
        info = super().info()

        info.add_infos(self.spec)

        return info


class DeltaKickSpecification(si.Specification):
    """
    A Specification for an :class:`IntegroDifferentialEquationSimulation`.
    """
    simulation_type = DeltaKickSimulation

    integration_method = si.utils.RestrictedValues('integration_method', ['simpson', 'trapezoid'])
    pulse_decomposition_strategy = si.utils.RestrictedValues('pulse_decomposition_strategy', ['amplitude', 'fluence'])

    def __init__(self, name,
                 time_initial = 0 * u.asec, time_final = 200 * u.asec, time_step = 1 * u.asec,
                 test_mass = u.electron_mass,
                 test_charge = u.electron_charge,
                 test_energy = states.HydrogenBoundState(1, 0).energy,
                 b_initial = 1,
                 integral_prefactor = -(u.electron_charge / u.hbar) ** 2,
                 electric_potential = potentials.NoElectricPotential(),
                 electric_potential_dc_correction = False,
                 kernel = kernels.hydrogen_kernel_LEN,
                 kernel_kwargs = {'omega_b': states.HydrogenBoundState(1, 0).energy / u.hbar},
                 evolution_gauge = 'LEN',
                 **kwargs):
        """
        The differential equation should be of the form
        da/dt = prefactor * f(t) * integral[ f(t') * a(t') * kernel(t - t', ...)

        Parameters
        ----------
        name : :class:`str`
            The name for the simulation.
        time_initial : :class:`float`
            The initial time.
        time_final : :class:`float`
            The final time.
        time_step : :class:`float`
            The time step to use in the evolution algorithm. For adaptive algorithms, this sets the initial time step.
        test_mass : :class:`float`
            The mass of the test particle.
        test_charge : :class:`float`
            The charge of the test particle.
        b_initial
            The initial value of a, the bound state probability amplitude.
        integral_prefactor
            The overall prefactor of the IDE.
        electric_potential
            The electric potential that provides ``f`` (either as the electric field or the vector potential).
        electric_potential_dc_correction
            If True, DC correction is performed on the electric field.
        kernel
            The kernel function of the IDE.
        kernel_kwargs
            Additional keyword arguments to pass to `kernel`.
        integration_method : {``'trapz'``, ``'simps'``}
            Which integration method to use, when applicable.
        evolution_method : {``'FE'``, ``'BE'``, ``'TRAP'``, ``'RK4'``, ``'ARK4'``}
            Which evolution algorithm/method to use.
        evolution_gauge : {``'LEN'``, ``'VEL'``}
            Which gauge to perform time evolution in.
        checkpoints
        checkpoint_every
        checkpoint_dir
        store_data_every
        time_step_minimum : :class:`float`
            The minimum time step that can be used by an adaptive algorithm.
        time_step_maximum : :class:`float`
            the maximum time step that can be used by an adaptive algorithm.
        epsilon : :class:`float`
            The acceptable fractional error in the quantity specified by `error_on`.
        error_on : {``'a'``, ``'da/dt'``}
            Which quantity to control the fractional error in.
        safety_factor : :class:`float`
            The safety factor that new time steps are multiplicatively fudged by.
        kwargs
        """
        super().__init__(name, **kwargs)

        self.time_initial = time_initial
        self.time_final = time_final
        self.time_step = time_step

        self.test_mass = test_mass
        self.test_charge = test_charge
        self.test_energy = test_energy

        self.b_initial = b_initial

        self.integral_prefactor = integral_prefactor

        self.electric_potential = electric_potential
        self.electric_potential_dc_correction = electric_potential_dc_correction

        self.kernel = kernel
        self.kernel_kwargs = dict()
        if kernel_kwargs is not None:
            self.kernel_kwargs.update(kernel_kwargs)

        self.evolution_gauge = evolution_gauge

    @property
    def test_omega(self):
        return self.test_energy / u.hbar

    @property
    def test_frequency(self):
        return self.test_omega / u.twopi

    def info(self):
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
        info_ide.add_field('Kernel', f'{self.kernel.__name__} with kwargs {self.kernel_kwargs}')
        info_ide.add_field('DC Correct Electric Field', 'yes' if self.electric_potential_dc_correction else 'no')

        info_potential = self.electric_potential.info()
        info_potential.header = 'Electric Potential: ' + info_potential.header
        info_ide.add_info(info_potential)

        info.add_info(info_ide)

        return info