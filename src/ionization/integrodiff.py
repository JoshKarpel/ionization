import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from tqdm import tqdm

import simulacra as si
from simulacra.units import *

from . import core, potentials, states


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def return_one(x, **kwargs):
    return 1


def gaussian_kernel_LEN(time_difference, *, tau_alpha):
    return (1 + (1j * time_difference / tau_alpha)) ** (-1.5)


def gaussian_kernel_VEL(time_difference, *, quiver_difference, tau_alpha, width):
    time_diff_inner = 1 / (1 + (1j * time_difference / tau_alpha))
    alpha_diff_inner = (quiver_difference / width) ** 2

    exp = np.exp(-alpha_diff_inner * time_diff_inner / 8)
    inv = time_diff_inner ** 1.5
    diff = 1 - (.25 * alpha_diff_inner * time_diff_inner)

    return exp * diff * inv


class IntegroDifferentialEquationSimulation(si.Simulation):
    """
    A class that encapsulates a simulation of an IDE with the form
    da/dt = prefactor * f(t) * integral[ f(t') * a(t') * kernel(t - t', ...)

    Attributes
    ----------
    time : :class:`float`
        The current simulation time.
    time_steps : :class:`int`
        The current number of time steps that have been solved for.
    a
        The solution of the IDE vs time (i.e., the bound state probability amplitude).
    a2
        The square of the absolute value of the solution vs time (i.e., the probability that the system is in the bound state).
    """

    def __init__(self, spec):
        super().__init__(spec)

        self.times = [self.spec.time_initial]
        self.time_index = 0
        self.time_step = self.spec.time_step

        self.a = [self.spec.a_initial]

        if self.spec.electric_potential_dc_correction:
            old_pot = self.spec.electric_potential
            self.spec.electric_potential = potentials.DC_correct_electric_potential(self.spec.electric_potential, self.times)

            logger.warning('Replaced electric potential {} --> {} for {} {}'.format(old_pot, self.spec.electric_potential, self.__class__.__name__, self.name))

        if self.spec.integration_method == 'simpson':
            self.integrate = integrate.simps
        elif self.spec.integration_method == 'trapezoid':
            self.integrate = integrate.trapz

        if self.spec.evolution_gauge == 'LEN':
            self.f = self.spec.electric_potential.get_electric_field_amplitude
        elif self.spec.evolution_gauge == 'VEL':
            self.f = self.spec.electric_potential.get_vector_potential_amplitude_numeric

    @property
    def time_steps(self):
        return len(self.times)

    @property
    def time(self):
        return self.times[self.time_index]

    @property
    def a2(self):
        return np.abs(self.a) ** 2

    def eval_kernel(self, time_difference, *, quiver_difference = None):
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
        if self.spec.evolution_gauge == 'LEN':
            return self.spec.kernel(time_difference, **self.spec.kernel_kwargs)
        elif self.spec.evolution_gauge == 'VEL':  # if we're on velocity, f is the vector potential
            return self.spec.kernel(time_difference, quiver_difference = quiver_difference, **self.spec.kernel_kwargs)

    def eval_quiver_motion(self, vector_potential_vs_time, times, time_step):
        """
        Calculate the quiver motion as a function of time using cumulative trapezoid-rule integration.

        Parameters
        ----------
        vector_potential_vs_time
            The vector potential at each of the `times`.
        times
            The times to get the quiver motion at.
        time_step
            The time step (only used if `times` is a single time).

        Returns
        -------
        quiver_motion_vs_time :
            The quiver motion at each of the `times`.
        """
        if len(times) != 1:
            integral = integrate.cumtrapz(y = vector_potential_vs_time, x = times, initial = 0)
        else:
            integral = vector_potential_vs_time * time_step
        return -(self.spec.test_charge / self.spec.test_mass) * integral

    def evolve_FE(self, a, times, time_step):
        """
        Evolve the IDE forward in time by `time_step` using the Forward Euler algorithm.

        Parameters
        ----------
        a
        times
        time_step

        Returns
        -------

        """
        times = np.array(times)
        time_curr = times[-1]

        fs_curr = self.f(times)
        try:
            f_curr = fs_curr[-1]
        except IndexError:
            f_curr = fs_curr
            fs_curr = np.array([fs_curr])

        if self.spec.evolution_gauge == 'VEL':
            quiver = self.eval_quiver_motion(self.f(times), times, time_step)
            try:
                qd = quiver[-1] - quiver
            except IndexError:
                qd = 0
        else:
            qd = None
        kernel = self.eval_kernel(time_curr - times, quiver_difference = qd)

        k = self.spec.prefactor * f_curr * self.integrate(y = fs_curr * kernel * a,
                                                          x = times)

        return a[-1] + (time_step * k), time_curr + time_step

    def evolve_BE(self, a, times, time_step):
        """
        Evolve the IDE forward in time by `time_step` using the Backward Euler algorithm.

        Parameters
        ----------
        a
        times
        time_step

        Returns
        -------

        """
        times_next = np.array(times + [times[-1] + time_step])
        times = np.array(times)

        time_curr = times[-1]

        fs_next = self.f(times_next)
        fs_curr = fs_next[:-1]
        f_next = fs_next[-1]

        if self.spec.evolution_gauge == 'VEL':
            quiver = self.eval_quiver_motion(self.f(times_next), times_next, time_step)
            try:
                qd = quiver[-1] - quiver[:-1]
            except IndexError:
                qd = 0
        else:
            qd = None
        kernel = self.eval_kernel(time_curr + time_step - times, quiver_difference = qd)

        k = self.spec.prefactor * f_next * self.integrate(y = fs_curr * a * kernel,
                                                          x = times)

        return (a[-1] + (time_step * k)) / (1 - self.spec.prefactor * ((time_step * f_next) ** 2)), time_curr + time_step  # estimate next point

    def evolve_TRAP(self, a, times, time_step):
        """
        Evolve the IDE forward in time by `time_step` using the Trapezoidal algorithm.

        Parameters
        ----------
        a
        times
        time_step

        Returns
        -------

        """
        times_next = np.array(times + [times[-1] + time_step])
        times = np.array(times)
        time_curr = times[-1]

        fs_next = self.f(times_next)
        fs_curr = fs_next[:-1]
        f_curr = fs_next[-2]
        f_next = fs_next[-1]

        fs_curr_times_a = fs_curr * a

        if self.spec.evolution_gauge == 'VEL':
            quiver_1 = self.eval_quiver_motion(self.f(times), times, time_step)
            quiver_2 = self.eval_quiver_motion(self.f(times_next), times_next, time_step)
            try:
                qd_1 = quiver_1[-1] - quiver_1
                qd_2 = quiver_2[-1] - quiver_2[:-1]
            except IndexError:
                qd_1 = 0
                qd_2 = 0
        else:
            qd_1 = None
            qd_2 = None

        kernel_1 = self.eval_kernel(time_curr + time_step - times, quiver_difference = qd_1)
        kernel_2 = self.eval_kernel(time_curr - times, quiver_difference = qd_2)

        k_1 = self.spec.prefactor * f_next * self.integrate(y = fs_curr_times_a * kernel_1,
                                                            x = times)
        k_2 = self.spec.prefactor * f_curr * self.integrate(y = fs_curr_times_a * kernel_2,
                                                            x = times)

        return (a[-1] + (time_step * (k_1 + k_2) / 2)) / (1 - (.5 * self.spec.prefactor * ((time_step * f_next) ** 2))), time_curr + time_step  # estimate next point

    def evolve_RK4(self, a, times, time_step):
        """
        Evole the IDE forward in time by `time_step` using the fourth-order Runge-Kutta algorithm.

        Parameters
        ----------
        a
        times
        time_step

        Returns
        -------

        """
        a_curr = a[-1]

        time_curr = times[-1]
        time_half = time_curr + (time_step / 2)
        time_next = time_curr + time_step

        times_half = np.array(times + [time_half])
        times_next = np.array(times + [time_next])
        times = np.array(times)

        time_difference_curr = time_curr - times
        time_difference_half = time_half - times_half
        time_difference_next = time_next - times_next

        fs_half = self.f(times_half)
        f_half = fs_half[-1]

        fs_next = self.f(times_next)
        f_next = fs_next[-1]
        f_curr = fs_next[-2]
        fs_curr = fs_next[:-1]

        if self.spec.evolution_gauge == 'VEL':
            quiver_curr = self.eval_quiver_motion(self.f(times), times, time_step)
            quiver_half = self.eval_quiver_motion(self.f(times_half), times_half, time_step)
            quiver_next = self.eval_quiver_motion(self.f(times_next), times_next, time_step)
            try:
                qd_curr = quiver_curr[-1] - quiver_curr
                qd_half = quiver_half[-1] - quiver_half
                qd_next = quiver_next[-1] - quiver_next
            except IndexError:
                qd_curr = 0
                qd_half = 0
                qd_next = 0
        else:
            qd_curr = None
            qd_half = None
            qd_next = None
        kernel_curr = self.eval_kernel(time_difference_curr, quiver_difference = qd_curr)
        kernel_half = self.eval_kernel(time_difference_half, quiver_difference = qd_half)
        kernel_next = self.eval_kernel(time_difference_next, quiver_difference = qd_next)

        fs_curr_times_a = fs_curr * a

        # integrate through the current time step
        integrand_for_k1 = fs_curr_times_a * kernel_curr
        integral_for_k1 = self.integrate(y = integrand_for_k1, x = times)
        k1 = self.spec.prefactor * f_curr * integral_for_k1
        a_midpoint_for_k2 = a_curr + (time_step * k1 / 2)  # time_step / 2 here because we moved forward to midpoint

        integrand_for_k2 = np.append(fs_curr_times_a, f_half * a_midpoint_for_k2) * kernel_half
        integral_for_k2 = self.integrate(y = integrand_for_k2, x = times_half)
        k2 = self.spec.prefactor * f_half * integral_for_k2  # time_step / 2 because it's half of an interval that we're integrating over
        a_midpoint_for_k3 = a_curr + (time_step * k2 / 2)  # estimate midpoint based on estimate of slope at midpoint

        integrand_for_k3 = np.append(fs_curr_times_a, f_half * a_midpoint_for_k3) * kernel_half
        integral_for_k3 = self.integrate(y = integrand_for_k3, x = times_half)
        k3 = self.spec.prefactor * f_half * integral_for_k3
        a_end_for_k4 = a_curr + (time_step * k3)  # estimate midpoint based on estimate of slope at midpoint

        integrand_for_k4 = np.append(fs_curr_times_a, f_next * a_end_for_k4) * kernel_next
        integral_for_k4 = self.integrate(y = integrand_for_k4, x = times_next)
        k4 = self.spec.prefactor * f_next * integral_for_k4

        return a_curr + (time_step * (k1 + (2 * k2) + (2 * k3) + k4) / 6), time_next  # estimate next point

    def evolve_ARK4(self, a, times, time_step):
        """
        Evolve the IDE forward in time by `time_step` using an adaptive-time-step fourth-order Runge-Kutta algorithm.

        Parameters
        ----------
        a
        times
        time_step

        Returns
        -------

        """
        a_full_step_estimate, _ = self.evolve_RK4(a, times, time_step)

        time_curr = times[-1]
        time_half = time_curr + (time_step / 2)

        a_half_step_estimate, _ = self.evolve_RK4(a, times, time_step / 2)
        a_double_step_estimate, _ = self.evolve_RK4(a + [a_half_step_estimate], times + [time_half], time_step / 2)

        delta_1 = a_double_step_estimate - a_full_step_estimate  # estimate truncation error from difference in estimates of a

        if self.spec.error_on == 'a':
            delta_0 = self.spec.epsilon * a[-1]
        elif self.spec.error_on == 'da/dt':
            delta_0 = self.spec.epsilon * (a[-1] - a_double_step_estimate)

        ratio = np.abs(delta_0 / delta_1)

        if ratio >= 1 or np.isinf(ratio) or np.isnan(ratio) or time_step == self.spec.minimum_time_step:  # step was ok
            old_step = self.time_step  # for log message
            if delta_1 != 0:  # don't adjust time step if truncation error is zero
                self.time_step = self.spec.safety_factor * time_step * (ratio ** (1 / 5))

            # ensure new time step is inside min and max allowable time steps
            self.time_step = min(self.spec.maximum_time_step, self.time_step)
            self.time_step = max(self.spec.minimum_time_step, self.time_step)

            logger.debug(f'Accepted ARK4 step to {uround(times[-1] + time_step, asec, 6)} as. Changed time step to {uround(self.time_step, asec, 6)} as from {uround(old_step, asec, 6)} as')

            return a_double_step_estimate + (delta_1 / 15), time_curr + time_step
        else:  # reject step
            old_step = self.time_step  # for log message

            self.time_step = self.spec.safety_factor * time_step * (ratio ** (1 / 4))  # set new time step
            self.time_step = min(self.spec.maximum_time_step, self.time_step)
            self.time_step = max(self.spec.minimum_time_step, self.time_step)

            logger.debug(f'Rejected ARK4 step. Changed time step to {uround(self.time_step, asec, 6)} as from {uround(old_step, asec, 6)} as')
            return self.evolve_ARK4(a, times, self.time_step)  # retry with new time step

    def run_simulation(self, callback = None):
        """
        Run the IDE simulation by repeatedly evolving it forward in time.


        Parameters
        ----------
        callback : callable
            A function that accepts the ``Simulation`` as an argument, called at the end of each time step.
        """
        logger.info(f'Performing time evolution on {self.name} ({self.file_name}), starting from time index {self.time_index}')
        self.status = si.STATUS_RUN

        while self.time < self.spec.time_final:
            new_a, new_t = getattr(self, f'evolve_{self.spec.evolution_method}')(self.a, self.times, self.time_step)
            self.a.append(new_a)
            self.times.append(new_t)
            self.time_index += 1

            logger.debug(f'{self} evolved to time index {self.time_index}')

            if callback is not None:
                callback(self)

            if self.spec.checkpoints:
                if (self.time_index + 1) % self.spec.checkpoint_every == 0:
                    self.save(target_dir = self.spec.checkpoint_dir)
                    logger.info(f'Checkpointed {self} at time index {self.time_index}')
                    self.status = si.STATUS_RUN

        self.a = np.array(self.a)
        self.times = np.array(self.times)

        time_indices = np.array(range(0, len(self.times)))
        self.data_mask = np.equal(time_indices, 0) + np.equal(time_indices, self.time_steps - 1)
        if self.spec.store_data_every >= 1:
            self.data_mask += np.equal(time_indices % self.spec.store_data_every, 0)

        self.times = self.times[self.data_mask]
        self.a = self.a[self.data_mask]

        self.status = si.STATUS_FIN
        logger.info(f'Finished performing time evolution on {self.name} ({self.file_name})')

    def attach_electric_potential_plot_to_axis(self,
                                               axis,
                                               time_unit = 'asec',
                                               legend_kwargs = None,
                                               show_y_label = False,
                                               show_electric_field = True,
                                               show_vector_potential = True):
        time_unit_value, time_unit_latex = get_unit_value_and_latex_from_unit(time_unit)

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
            axis.plot(self.times / time_unit_value, self.spec.electric_potential.get_electric_field_amplitude(self.times) / atomic_electric_field,
                      color = core.COLOR_ELECTRIC_FIELD,
                      linewidth = 1.5,
                      label = e_label)
            y_labels.append(e_label)
        if show_vector_potential:
            a_label = fr'$ e \, {core.LATEX_AFIELD}(t) $'
            axis.plot(self.times / time_unit_value, proton_charge * self.spec.electric_potential.get_vector_potential_amplitudes_numeric(self.times) / atomic_momentum,
                      color = core.COLOR_VECTOR_POTENTIAL,
                      linewidth = 1.5,
                      label = a_label)
            y_labels.append(a_label)

        if show_y_label:
            axis.set_ylabel(', '.join(y_labels), fontsize = 13)

        axis.set_xlabel('Time $t$ (${}$)'.format(time_unit_latex), fontsize = 13)

        axis.tick_params(labelright = True)

        axis.set_xlim(self.times[0] / time_unit_value, self.times[-1] / time_unit_value)

        axis.legend(**legend_kwargs)

        axis.grid(True, **si.vis.GRID_KWARGS)

    def plot_a2_vs_time(self,
                        log = False,
                        time_unit = 'asec',
                        show_title = False,
                        **kwargs):
        with si.vis.FigureManager(self.file_name + '__a2_vs_time', **kwargs) as figman:
            fig = figman.fig

            t_scale_unit, t_scale_name = get_unit_value_and_latex_from_unit(time_unit)

            grid_spec = matplotlib.gridspec.GridSpec(2, 1, height_ratios = [4, 1], hspace = 0.07)  # TODO: switch to fixed axis construction
            ax_a = plt.subplot(grid_spec[0])
            ax_pot = plt.subplot(grid_spec[1], sharex = ax_a)

            self.attach_electric_potential_plot_to_axis(ax_pot,
                                                        time_unit = time_unit)

            ax_a.plot(self.times / t_scale_unit,
                      self.a2,
                      color = 'black',
                      linewidth = 2)

            if log:
                ax_a.set_yscale('log')
                min_overlap = np.min(overlap)
                ax_a.set_ylim(bottom = max(1e-9, min_overlap * .1), top = 1.0)
                ax_a.grid(True, which = 'both', **si.vis.GRID_KWARGS)
            else:
                ax_a.set_ylim(0.0, 1.0)
                ax_a.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                ax_a.grid(True, **si.vis.GRID_KWARGS)

            ax_a.set_xlim(self.spec.time_initial / t_scale_unit, self.spec.time_final / t_scale_unit)

            ax_a.set_ylabel(r'$\left| a_{\alpha}(t) \right|^2$', fontsize = 13)

            # Find at most n+1 ticks on the y-axis at 'nice' locations
            max_yticks = 4
            yloc = plt.MaxNLocator(max_yticks, prune = 'upper')
            ax_pot.yaxis.set_major_locator(yloc)

            max_xticks = 6
            xloc = plt.MaxNLocator(max_xticks, prune = 'both')
            ax_pot.xaxis.set_major_locator(xloc)

            ax_pot.tick_params(axis = 'x', which = 'major', labelsize = 10)
            ax_pot.tick_params(axis = 'y', which = 'major', labelsize = 10)
            ax_a.tick_params(axis = 'both', which = 'major', labelsize = 10)

            ax_a.tick_params(labelleft = True,
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
                title = ax_a.set_title(self.name)
                title.set_y(1.15)

            postfix = ''
            if log:
                postfix += '__log'

            figman.name += postfix


class IntegroDifferentialEquationSpecification(si.Specification):
    """
    A Specification for an :class:`IntegroDifferentialEquationSimulation`.
    """
    simulation_type = IntegroDifferentialEquationSimulation

    integration_method = si.utils.RestrictedValues('integration_method', ('simpson', 'trapezoid'))

    def __init__(self, name,
                 time_initial = 0 * asec, time_final = 200 * asec, time_step = 1 * asec,
                 test_mass = electron_mass, test_charge = electron_charge,
                 a_initial = 1,
                 prefactor = 1,
                 electric_potential = potentials.NoElectricPotential(),
                 electric_potential_dc_correction = False,
                 kernel = return_one, kernel_kwargs = None,
                 integration_method = 'simpson',
                 evolution_method = 'FE',
                 evolution_gauge = 'LEN',
                 checkpoints = False, checkpoint_every = 20, checkpoint_dir = None,
                 store_data_every = 1,
                 minimum_time_step = .01 * asec, maximum_time_step = 10 * asec,
                 epsilon = 1e-3, error_on = 'da/dt', safety_factor = .98,
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
        a_initial
            The initial value of a, the bound state probability amplitude.
        prefactor
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
        minimum_time_step : :class:`float`
            The minimum time step that can be used by an adaptive algorithm.
        maximum_time_step : :class:`float`
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

        self.a_initial = a_initial

        self.prefactor = prefactor

        self.electric_potential = electric_potential
        self.electric_potential_dc_correction = electric_potential_dc_correction

        self.kernel = kernel
        self.kernel_kwargs = dict()
        if kernel_kwargs is not None:
            self.kernel_kwargs.update(kernel_kwargs)

        self.integration_method = integration_method
        self.evolution_method = evolution_method
        self.evolution_gauge = evolution_gauge

        self.checkpoints = checkpoints
        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir = checkpoint_dir

        self.store_data_every = store_data_every

        self.minimum_time_step = minimum_time_step
        self.maximum_time_step = maximum_time_step

        self.epsilon = epsilon
        self.error_on = error_on

        self.safety_factor = safety_factor

    def info(self):
        info = super().info()

        info_checkpoint = si.Info(header = 'Checkpointing')
        if self.checkpoints:
            if self.checkpoint_dir is not None:
                working_in = self.checkpoint_dir
            else:
                working_in = 'cwd'
            info_checkpoint.header += ': every {} time steps, working in {}'.format(self.checkpoint_every, working_in)
        else:
            info_checkpoint.header += ': disabled'

        info.add_info(info_checkpoint)

        info_evolution = si.Info(header = 'Time Evolution')
        info_evolution.add_field('Initial Time', f'{uround(self.time_initial, asec, 3)} as')
        info_evolution.add_field('Final Time', f'{uround(self.time_final, asec, 3)} as')

        info.add_info(info_evolution)

        info_algorithm = si.Info(header = 'Evolution Algorithm')
        info_algorithm.add_field('Integration Method', self.integration_method)
        info_algorithm.add_field('Evolution Method', self.evolution_method)
        if self.evolution_method == 'ARK4':
            info_algorithm.add_field('Error Control On', self.error_on)
            info_algorithm.add_field('Epsilon', self.epsilon)
            info_algorithm.add_field('Safety Factor', self.safety_factor)
            info_algorithm.add_field('Time Step', f'{uround(self.time_step, asec, 3)} as')
            info_algorithm.add_field('Minimum Time Step', f'{uround(self.minimum_time_step, asec, 3)} as')
            info_algorithm.add_field('Maximum Time Step', f'{uround(self.maximum_time_step, asec, 3)} as')

        info.add_info(info_algorithm)

        info_ide = si.Info(header = 'IDE Parameters')
        info_ide.add_field('Initial State', f'a = {self.a_initial}')
        info_ide.add_field('Prefactor', self.prefactor)
        info_ide.add_field('Kernel', f'{self.kernel.__name__} with kwargs {self.kernel_kwargs}')

        info_potential = self.electric_potential.info()
        info_potential.header = 'Electric Potential: ' + info_potential.header
        info_ide.add_info(info_potential)

        info.add_info(info_ide)

        return info
