import itertools
import logging
from copy import copy

import numpy as np
import numpy.ma as ma

import simulacra as si
import simulacra.cluster as clu
import simulacra.units as u

from . import core, mesh, ide, jobutils

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

PARAMETER_TO_SYMBOL = {
    'pulse_width': r'\tau',
    'fluence': r'H',
    'amplitude': r'\mathcal{E}_0',
    'phase': r'\varphi',
    'number_of_cycles': r'N_c',
    'delta_r': r'\Delta r',
    'delta_t': r'\Delta t',
}

PARAMETER_TO_UNIT_NAME = {
    'pulse_width': 'asec',
    'fluence': 'Jcm2',
    'amplitude': 'atomic_electric_field',
    'phase': 'pi',
    'number_of_cycles': '',
    'delta_r': 'bohr_radius',
    'delta_t': 'asec',
}


def modulation_depth(cosine, sine):
    """
    (c-s) / (c+s)

    Parameters
    ----------
    cosine
    sine

    Returns
    -------

    """
    return (cosine - sine) / (cosine + sine)


def format_ionization_metric_name(ionization_metric):
    return ionization_metric.replace('_', ' ').title()


def format_log_str(is_logs, axis_names):
    log_str = ''
    if any(is_logs):
        log_str = '__log' + ''.join(axis_name for is_log, axis_name in zip(is_logs, axis_names) if is_log)

    return log_str


class PulseParameterScanMixin:
    def make_summary_plots(self):
        super().make_summary_plots()

        if len(self.unprocessed_sim_names) == 0:
            logger.info(f'Generating autoscans for job {self.name}')
            self.make_scans()
            self.make_heatmaps()
            self.make_modulation_depth_heatmaps()

    def _select_two_parameters(self, parameters = None):
        if parameters is None:
            parameters = self.scan_parameters

        two_parameter_generator = itertools.permutations(self.scan_parameters, r = 2)

        while True:
            first_parameter, second_parameter = next(two_parameter_generator)
            other_parameters = [p for p in parameters
                                if p not in (first_parameter, second_parameter)]

            yield first_parameter, second_parameter, other_parameters

    def _get_parameters_to_parameter_sets(self, parameters):
        plot_parameter_to_set = {}
        for parameter in parameters:
            try:
                parameter_set = self.parameter_set(parameter)
                if len(parameter_set) > 10:  # would create too many unique plots
                    continue
                plot_parameter_to_set[parameter] = parameter_set
            except AttributeError:
                continue

        return plot_parameter_to_set

    def _get_plot_parameter_strings(self, plot_parameters, plot_parameter_values):
        str_for_filename = '__'.join(
            f'{param}={u.uround(value, PARAMETER_TO_UNIT_NAME[param], 3)}{PARAMETER_TO_UNIT_NAME[param]}'
            for param, value in zip(plot_parameters, plot_parameter_values)
        )

        str_for_plot = ', '.join(
            fr"${PARAMETER_TO_SYMBOL[param]} \, = {u.uround(value, PARAMETER_TO_UNIT_NAME[param], 3)} \, {u.UNIT_NAME_TO_LATEX[PARAMETER_TO_UNIT_NAME[param]]}$"
            for param, value in zip(plot_parameters, plot_parameter_values)
        )

        return str_for_filename, str_for_plot

    def _get_lines(self, line_parameter, scan_parameter, ionization_metric, base_selector):
        x_data = []
        y_data = []
        line_labels = []

        for line_parameter_value in sorted(self.parameter_set(line_parameter)):
            selector = {
                **base_selector,
                line_parameter: line_parameter_value,
            }
            results = sorted(self.select_by_kwargs(**selector), key = lambda result: getattr(result, scan_parameter))

            x_data.append(np.array([getattr(result, scan_parameter) for result in results]))
            y_data.append(np.array([getattr(result, ionization_metric) for result in results]))

            parameter_name = PARAMETER_TO_UNIT_NAME[line_parameter]
            parameter_unit_tex = u.UNIT_NAME_TO_LATEX[parameter_name]
            label = fr"${PARAMETER_TO_SYMBOL[line_parameter]} \, = {u.uround(line_parameter_value, parameter_name)} \, {parameter_unit_tex}$"
            line_labels.append(label)

        return x_data, y_data, line_labels

    def make_scans(self):
        for line_parameter, scan_parameter, plot_parameters in self._select_two_parameters():
            try:
                line_parameter_set, scan_parameter_set = self.parameter_set(line_parameter), self.parameter_set(scan_parameter)
            except AttributeError:  # parameter is not tracked
                continue

            if any((len(scan_parameter_set) < 10,  # not enough x data, or too many lines
                    len(line_parameter_set) > 8,)):
                continue

            plot_parameter_to_set = self._get_parameters_to_parameter_sets(plot_parameters)
            if len(plot_parameter_to_set) == 0:  # this sometimes happens on heatmap jobs?
                continue

            plot_parameters = tuple(plot_parameter_to_set.keys())
            plot_parameter_value_groups = (plot_parameter_values for plot_parameter_values in itertools.product(*plot_parameter_to_set.values()))

            for plot_parameter_values in plot_parameter_value_groups:
                for ionization_metric in self.ionization_metrics:
                    base_selector = dict(zip(plot_parameters, plot_parameter_values))
                    x_data, y_data, line_labels = self._get_lines(line_parameter, scan_parameter, ionization_metric, base_selector)
                    if any(x[0] == x[-1] for x in x_data):
                        continue

                    str_for_filename, str_for_plot = self._get_plot_parameter_strings(plot_parameters, plot_parameter_values)
                    filename = '__'.join((
                        self.name,
                        '1d',
                        ionization_metric,
                        str_for_filename,
                        f'scanning_{scan_parameter}_grouped_by_{line_parameter}',
                    ))
                    title = self.name + '\n' + str_for_plot

                    for log_x, log_y in itertools.product((True, False), repeat = 2):
                        if any((log_x and scan_parameter == 'phase',
                                log_x and any(np.any(x <= 0) for x in x_data),
                                log_y and any(np.any(y <= 0) for y in y_data),)):
                            continue

                        if log_y:
                            y_lower_limit = None
                        else:
                            y_lower_limit = 0

                        log_str = format_log_str(
                            (log_x, log_y),
                            ('X', 'Y'),
                        )

                        si.vis.xxyy_plot(
                            filename + log_str,
                            x_data,
                            y_data,
                            line_labels = line_labels,
                            x_label = fr'${PARAMETER_TO_SYMBOL[scan_parameter]}$',
                            x_unit = PARAMETER_TO_UNIT_NAME[scan_parameter],
                            x_log_axis = log_x,
                            y_label = format_ionization_metric_name(ionization_metric),
                            y_lower_limit = y_lower_limit,
                            y_upper_limit = 1,
                            y_log_axis = log_y,
                            title = title,
                            legend_on_right = True if len(line_labels) > 5 else False,
                            target_dir = self.summaries_dir,
                        )

    def make_heatmaps(self):
        for x_parameter, y_parameter, plot_parameters in self._select_two_parameters():
            try:
                x_parameter_set, y_parameter_set = self.parameter_set(x_parameter), self.parameter_set(y_parameter)
            except AttributeError:
                continue

            if any((len(x_parameter_set) < 10,
                    len(y_parameter_set) < 10)):  # not dense enough
                continue

            plot_parameter_to_set = self._get_parameters_to_parameter_sets(plot_parameters)
            plot_parameters = tuple(plot_parameter_to_set.keys())
            plot_parameter_value_groups = (plot_parameter_values for plot_parameter_values in itertools.product(*plot_parameter_to_set.values()))

            x, y = np.array(sorted(x_parameter_set)), np.array(sorted(y_parameter_set))
            x_mesh, y_mesh = np.meshgrid(x, y, indexing = 'ij')

            for plot_parameter_values in plot_parameter_value_groups:
                results = self.select_by_kwargs(**dict(zip(plot_parameters, plot_parameter_values)))
                xy_to_result = {(getattr(r, x_parameter), getattr(r, y_parameter)): r for r in results}

                for ionization_metric in self.ionization_metrics:
                    xy_to_metric = {(x, y): getattr(r, ionization_metric) for (x, y), r in xy_to_result.items()}

                    z_mesh = np.empty_like(x_mesh)
                    try:
                        for ii, x_value in enumerate(x):
                            for jj, y_value in enumerate(y):
                                z_mesh[ii, jj] = xy_to_metric[x_value, y_value]
                    except KeyError:  # alignment problem, job is probably not actually a heatmap over these parameters
                        continue

                    str_for_filename, str_for_plot = self._get_plot_parameter_strings(plot_parameters, plot_parameter_values)
                    filename = '__'.join((
                        self.name,
                        '2d',
                        ionization_metric,
                        str_for_filename,
                        f'{x_parameter}_vs_{y_parameter}',
                    ))
                    title = self.name + '\n' + str_for_plot

                    for log_x, log_y, log_z in itertools.product((True, False), repeat = 3):
                        if any((log_x and x_parameter == 'phase',
                                log_y and y_parameter == 'phase',  # skip log phase plots
                                log_x and np.any(x_mesh <= 0),
                                log_y and np.any(y_mesh <= 0))):
                            continue

                        log_str = format_log_str(
                            (log_x, log_y, log_z),
                            ('X', 'Y', 'Z'),
                        )

                        si.vis.xyz_plot(
                            filename + log_str,
                            x_mesh, y_mesh, z_mesh,
                            x_label = fr'${PARAMETER_TO_SYMBOL[x_parameter]}$',
                            y_label = fr'${PARAMETER_TO_SYMBOL[y_parameter]}$',
                            z_label = format_ionization_metric_name(ionization_metric),
                            x_unit = PARAMETER_TO_UNIT_NAME[x_parameter],
                            y_unit = PARAMETER_TO_UNIT_NAME[y_parameter],
                            x_log_axis = log_x,
                            y_log_axis = log_y,
                            z_log_axis = log_z,
                            z_upper_limit = 1,
                            title = title,
                            target_dir = self.summaries_dir,
                        )

    def make_modulation_depth_heatmaps(self):
        phase_set = self.parameter_set('phase')
        if 0 not in phase_set and u.pi / 2 not in phase_set:
            return

        scan_parameters = [s for s in self.scan_parameters if s != 'phase']  # no scans over phase
        for x_parameter, y_parameter, plot_parameters in self._select_two_parameters(parameters = scan_parameters):
            try:
                x_parameter_set, y_parameter_set = self.parameter_set(x_parameter), self.parameter_set(y_parameter)
            except AttributeError:
                continue

            if any((len(x_parameter_set) < 10,
                    len(y_parameter_set) < 10)):  # not dense enough
                continue

            plot_parameter_to_set = self._get_parameters_to_parameter_sets(plot_parameters)
            plot_parameters = tuple(plot_parameter_to_set.keys())
            plot_parameter_value_groups = (plot_parameter_values for plot_parameter_values in itertools.product(*plot_parameter_to_set.values()))

            x, y = np.array(sorted(x_parameter_set)), np.array(sorted(y_parameter_set))
            x_mesh, y_mesh = np.meshgrid(x, y, indexing = 'ij')

            for plot_parameter_values in plot_parameter_value_groups:
                results = self.select_by_kwargs(**dict(zip(plot_parameters, plot_parameter_values)))
                xy_to_result_cos = {(getattr(r, x_parameter), getattr(r, y_parameter)): r for r in results if r.phase == 0}
                xy_to_result_sin = {(getattr(r, x_parameter), getattr(r, y_parameter)): r for r in results if r.phase == u.pi / 2}

                for ionization_metric in self.ionization_metrics:
                    xy_to_metric_cos = {(x, y): getattr(r, ionization_metric) for (x, y), r in xy_to_result_cos.items()}
                    xy_to_metric_sin = {(x, y): getattr(r, ionization_metric) for (x, y), r in xy_to_result_sin.items()}

                    z_mesh = np.empty_like(x_mesh)
                    try:
                        for ii, x_value in enumerate(x):
                            for jj, y_value in enumerate(y):
                                z_mesh[ii, jj] = modulation_depth(xy_to_metric_cos[x_value, y_value], xy_to_metric_sin[x_value, y_value])
                    except KeyError:  # alignment problem, job is probably not actually a heatmap over these parameters
                        continue

                    str_for_filename, str_for_plot = self._get_plot_parameter_strings(plot_parameters, plot_parameter_values)
                    filename = '__'.join((
                        self.name,
                        '2d',
                        f'mod_depth_from_{ionization_metric}',
                        str_for_filename,
                        f'{x_parameter}_vs_{y_parameter}',
                    ))
                    title = self.name + '\n' + str_for_plot

                    for log_x, log_y, log_z in itertools.product((True, False), repeat = 3):
                        if any((log_x and x_parameter == 'phase',
                                log_y and y_parameter == 'phase',  # skip log phase plots
                                log_x and np.any(x_mesh <= 0),
                                log_y and np.any(y_mesh <= 0))):
                            continue

                        log_str = format_log_str(
                            (log_x, log_y, log_z),
                            ('X', 'Y', 'Z'),
                        )

                        si.vis.xyz_plot(
                            filename + log_str,
                            x_mesh, y_mesh, z_mesh,
                            x_label = fr'${PARAMETER_TO_SYMBOL[x_parameter]}$',
                            y_label = fr'${PARAMETER_TO_SYMBOL[y_parameter]}$',
                            z_label = f'Modulation Depth\n({format_ionization_metric_name(ionization_metric)})',
                            x_unit = PARAMETER_TO_UNIT_NAME[x_parameter],
                            y_unit = PARAMETER_TO_UNIT_NAME[y_parameter],
                            x_log_axis = log_x,
                            y_log_axis = log_y,
                            z_log_axis = log_z,
                            z_lower_limit = -1,
                            z_upper_limit = 1,
                            title = title,
                            colormap = plt.get_cmap('RdBu_r'),
                            target_dir = self.summaries_dir,
                        )


class PulseSimulationResult(clu.SimulationResult):
    def __init__(self, sim, job_processor):
        super().__init__(sim, job_processor)

        self.electric_potential = copy(sim.spec.electric_potential)
        self.pulse_type = copy(sim.spec.pulse_type)

        for attr in jobutils.POTENTIAL_ATTRS:
            try:
                setattr(self, attr, copy((getattr(sim.spec, attr))))
            except AttributeError as e:
                logger.debug(f'Failed to copy pulse attribute {attr} from {sim}')


class MeshSimulationResult(PulseSimulationResult):
    def __init__(self, sim, job_processor):
        super().__init__(sim, job_processor)

        self.time_steps = copy(sim.time_steps)

        state_overlaps = sim.data.state_overlaps_vs_time

        self.final_norm = copy(sim.data.norm_vs_time[-1])
        self.final_initial_state_overlap = copy(state_overlaps[sim.spec.initial_state][-1])
        self.final_bound_state_overlap = copy(sum(state_overlaps[s][-1] for s in sim.bound_states))
        self.final_free_state_overlap = copy(sum(state_overlaps[s][-1] for s in sim.free_states))

        try:
            self.final_internal_energy_expectation_value = copy(sim.internal_energy_expectation_value_vs_time[-1])
            self.final_total_energy_expectation_value = copy(sim.total_energy_expectation_value_vs_time[-1])
        except AttributeError:
            pass

        try:
            self.final_radial_position_expectation_value = copy(sim.radial_position_expectation_value_vs_time[-1])
        except AttributeError:
            pass

        try:
            self.final_electric_dipole_moment_expectation_value = copy(sim.electric_dipole_moment_expectation_value_vs_time[-1])
        except AttributeError:
            pass

        self.electric_potential = copy(sim.spec.electric_potential)

        if len(sim.data_times) > 2:
            self.make_wavefunction_plots(sim)

    def make_wavefunction_plots(self, sim):
        plot_kwargs = dict(
            target_dir = self.plots_dir,
            plot_name = 'name',
            show_title = True,
        )

        grouped_states, group_labels = sim.group_free_states_by_continuous_attr(
            'energy',
            divisions = 12,
            cutoff_value = 100 * u.eV,
            attr_unit = 'eV'
        )
        sim.plot_wavefunction_vs_time(
            **plot_kwargs,
            name_postfix = f'__energy__{sim.file_name}',
            grouped_free_states = grouped_states,
            group_free_states_labels = group_labels
        )

        try:
            grouped_states, group_labels = sim.group_free_states_by_discrete_attr('l', cutoff_value = 10)
            sim.plot_wavefunction_vs_time(
                **plot_kwargs,
                name_postfix = f'__l__{sim.file_name}',
                grouped_free_states = grouped_states,
                group_free_states_labels = group_labels
            )
        except AttributeError:  # free states must not have l
            pass


class PulseJobProcessor(PulseParameterScanMixin, clu.JobProcessor):
    scan_parameters = ['pulse_width', 'fluence', 'phase', 'amplitude', 'number_of_cycles']


class MeshJobProcessor(PulseJobProcessor):
    simulation_type = mesh.MeshSimulation
    simulation_result_type = MeshSimulationResult

    ionization_metrics = ['final_norm', 'final_initial_state_overlap', 'final_bound_state_overlap']

    def make_velocity_plot(self):
        sim_numbers = [result.file_name for result in self.data.values() if result is not None]
        velocity = np.array([result.spacetime_points / result.running_time for result in self.data.values() if result is not None])

        si.vis.xy_plot(
            f'{self.name}__velocity',
            sim_numbers,
            velocity,
            line_kwargs = [dict(linestyle = '', marker = '.')],
            y_unit = 1,
            x_label = 'Simulation Number', y_label = 'Space-Time Points / Second',
            title = f'{self.name} Velocity',
            target_dir = self.summaries_dir
        )

        logger.debug(f'Generated velocity plot for job {self.name}')


class IDESimulationResult(PulseSimulationResult):
    def __init__(self, sim, job_processor):
        super().__init__(sim, job_processor)

        self.final_bound_state_overlap = np.abs(sim.b[-1]) ** 2

        if len(sim.data_times) > 2:
            self.make_b2_plots(sim)

    def make_b2_plots(self, sim):
        plot_kwargs = dict(
            target_dir = self.plots_dir,
            plot_name = 'name',
            show_title = True,
            name_postfix = f'__{sim.file_name}',
        )

        sim.plot_b2_vs_time(**plot_kwargs)
        sim.plot_b2_vs_time(**plot_kwargs, log = True)

    @property
    def final_initial_state_overlap(self):
        return self.final_bound_state_overlap

    @property
    def final_norm(self):
        return self.final_bound_state_overlap


class IDEJobProcessor(PulseJobProcessor):
    simulation_type = ide.IntegroDifferentialEquationSimulation
    simulation_result_type = IDESimulationResult

    ionization_metrics = ['final_bound_state_overlap']


class MeshConvergenceSimulationResult(MeshSimulationResult):
    def __init__(self, sim, job_processor):
        super().__init__(sim, job_processor)

        self.r_points = copy(sim.spec.r_points)
        self.r_bound = copy(sim.spec.r_bound)
        self.delta_r = self.r_bound / self.r_points
        self.delta_t = copy(sim.spec.time_step)


class MeshConvergenceJobProcessor(MeshJobProcessor):
    simulation_result_type = MeshConvergenceSimulationResult

    scan_parameters = ['delta_r', 'delta_t']

    def make_summary_plots(self):
        super().make_summary_plots()

        if len(self.unprocessed_sim_names) == 0:
            logger.info(f'Generating relative pulse parameter scans for job {self.name}')
            self.make_pulse_parameter_scans_1d_relative()
            self.make_pulse_parameter_scans_2d_relative()

    def make_scans(self):
        for ionization_metric in self.ionization_metrics:
            ionization_metric_name = ionization_metric.replace('_', ' ').title()

            for plot_parameter, scan_parameter in itertools.permutations(self.scan_parameters):
                plot_parameter_name, scan_parameter_name = plot_parameter.replace('_', ' ').title(), scan_parameter.replace('_', ' ').title()
                plot_parameter_unit, scan_parameter_unit = PARAMETER_TO_UNIT_NAME[plot_parameter], PARAMETER_TO_UNIT_NAME[scan_parameter]
                plot_parameter_set, scan_parameter_set = self.parameter_set(plot_parameter), self.parameter_set(scan_parameter)

                for plot_parameter_value in plot_parameter_set:
                    plot_name = f'{ionization_metric}__{plot_parameter}={u.uround(plot_parameter_value, plot_parameter_unit, 3)}{plot_parameter_unit}'

                    selector = {
                        plot_parameter: plot_parameter_value,
                    }
                    results = sorted(self.select_by_kwargs(**selector), key = lambda result: getattr(result, scan_parameter))

                    line = np.array([getattr(result, ionization_metric) for result in results])

                    x = np.array([getattr(result, scan_parameter) for result in results])

                    for log_x, log_y in itertools.product((False, True), repeat = 2):
                        if not log_y:
                            y_upper_limit = 1
                            y_lower_limit = 0
                        else:
                            y_upper_limit = None
                            y_lower_limit = None

                        log_str = ''
                        if any((log_x, log_y)):
                            log_str = '__log'
                            if log_x:
                                log_str += 'X'
                            if log_y:
                                log_str += 'Y'

                        si.vis.xy_plot(
                            f'1d__{plot_name}{log_str}',
                            x,
                            line,
                            title = f"{plot_parameter_name}$\, = {u.uround(plot_parameter_value, plot_parameter_unit, 3)} \, {u.UNIT_NAME_TO_LATEX[plot_parameter_unit]}$",
                            x_label = scan_parameter_name,
                            x_unit = scan_parameter_unit,
                            x_log_axis = log_x,
                            y_label = ionization_metric_name,
                            y_lower_limit = y_lower_limit,
                            y_upper_limit = y_upper_limit,
                            y_log_axis = log_y,
                            legend_on_right = True,
                            target_dir = self.summaries_dir,
                        )

    def make_pulse_parameter_scans_1d_relative(self):
        for ionization_metric in self.ionization_metrics:
            ionization_metric_name = ionization_metric.replace('_', ' ').title()

            for plot_parameter, scan_parameter in itertools.permutations(self.scan_parameters):
                plot_parameter_name, scan_parameter_name = plot_parameter.replace('_', ' ').title(), scan_parameter.replace('_', ' ').title()
                plot_parameter_unit, scan_parameter_unit = PARAMETER_TO_UNIT_NAME[plot_parameter], PARAMETER_TO_UNIT_NAME[scan_parameter]
                plot_parameter_set, scan_parameter_set = self.parameter_set(plot_parameter), self.parameter_set(scan_parameter)

                for plot_parameter_value in plot_parameter_set:
                    plot_name = f'{ionization_metric}__{plot_parameter}={u.uround(plot_parameter_value, plot_parameter_unit, 3)}{plot_parameter_unit}__rel'

                    selector = {
                        plot_parameter: plot_parameter_value,
                    }
                    results = sorted(self.select_by_kwargs(**selector), key = lambda result: getattr(result, scan_parameter))

                    line = np.array([np.abs(getattr(result, ionization_metric) - getattr(results[0], ionization_metric)) for result in results])
                    line = ma.masked_less_equal(line, 0)

                    x = np.array([getattr(result, scan_parameter) for result in results])

                    for log_x, log_y in itertools.product((False, True), repeat = 2):
                        log_str = ''
                        if any((log_x, log_y)):
                            log_str = '__log'
                            if log_x:
                                log_str += 'X'
                            if log_y:
                                log_str += 'Y'

                        si.vis.xy_plot(
                            f'1d__{plot_name}{log_str}',
                            x,
                            line,
                            title = f"{plot_parameter_name}$\, = {u.uround(plot_parameter_value, plot_parameter_unit, 3)} \, {u.UNIT_NAME_TO_LATEX[plot_parameter_unit]}$ (Diff from Best)",
                            x_label = scan_parameter_name,
                            x_unit = scan_parameter_unit,
                            y_label = ionization_metric_name,
                            x_log_axis = log_x,
                            y_log_axis = log_y,
                            legend_on_right = True,
                            target_dir = self.summaries_dir,
                        )

    def make_heatmaps(self):
        for ionization_metric in self.ionization_metrics:
            ionization_metric_name = ionization_metric.replace('_', ' ').title()

            for x_parameter, y_parameter in itertools.combinations(self.scan_parameters, r = 2):
                x_parameter_name, y_parameter_name = x_parameter.replace('_', ' ').title(), y_parameter.replace('_', ' ').title()
                x_parameter_unit, y_parameter_unit = PARAMETER_TO_UNIT_NAME[x_parameter], PARAMETER_TO_UNIT_NAME[y_parameter]
                x_parameter_set, y_parameter_set = self.parameter_set(x_parameter), self.parameter_set(y_parameter)

                plot_name = ionization_metric

                x = np.array(sorted(x_parameter_set))
                y = np.array(sorted(y_parameter_set))

                x_mesh, y_mesh = np.meshgrid(x, y, indexing = 'ij')

                xy_to_metric = {(getattr(r, x_parameter), getattr(r, y_parameter)): getattr(r, ionization_metric) for r in self.data.values()}
                z_mesh = np.zeros(x_mesh.shape) * np.NaN

                for ii, x_value in enumerate(x):
                    for jj, y_value in enumerate(y):
                        z_mesh[ii, jj] = xy_to_metric[(x_value, y_value)]

                for log_x, log_y, log_z in itertools.product((True, False), repeat = 3):
                    if log_z:
                        z_lower_limit = np.nanmin(z_mesh)
                        z_upper_limit = np.nanmax(z_mesh)
                    else:
                        z_lower_limit = 0
                        z_upper_limit = 1

                    log_str = ''
                    if any((log_x, log_y, log_z)):
                        log_str = '__log'
                        if log_x:
                            log_str += 'X'
                        if log_y:
                            log_str += 'Y'
                        if log_z:
                            log_str += 'Z'

                    si.vis.xyz_plot(
                        f'2d__{plot_name}{log_str}',
                        x_mesh, y_mesh, z_mesh,
                        x_unit = x_parameter_unit,
                        y_unit = y_parameter_unit,
                        x_label = x_parameter_name,
                        y_label = y_parameter_name,
                        z_label = ionization_metric_name,
                        x_log_axis = log_x,
                        y_log_axis = log_y,
                        z_log_axis = log_z,
                        z_lower_limit = z_lower_limit,
                        z_upper_limit = z_upper_limit,
                        target_dir = self.summaries_dir,
                    )

    def make_pulse_parameter_scans_2d_relative(self):
        for ionization_metric in self.ionization_metrics:
            ionization_metric_name = ionization_metric.replace('_', ' ').title()

            for x_parameter, y_parameter in itertools.combinations(self.scan_parameters, r = 2):
                x_parameter_name, y_parameter_name = x_parameter.replace('_', ' ').title(), y_parameter.replace('_', ' ').title()
                x_parameter_unit, y_parameter_unit = PARAMETER_TO_UNIT_NAME[x_parameter], PARAMETER_TO_UNIT_NAME[y_parameter]
                x_parameter_set, y_parameter_set = self.parameter_set(x_parameter), self.parameter_set(y_parameter)

                plot_name = f'{ionization_metric}__{x_parameter}_x_{y_parameter}__rel'

                x = np.array(sorted(x_parameter_set))
                y = np.array(sorted(y_parameter_set))

                x_min = np.nanmin(x)
                y_min = np.nanmin(y)

                x_mesh, y_mesh = np.meshgrid(x, y, indexing = 'ij')

                xy_to_metric = {(getattr(r, x_parameter), getattr(r, y_parameter)): getattr(r, ionization_metric) for r in self.data.values()}
                z_mesh = np.zeros(x_mesh.shape) * np.NaN

                best = xy_to_metric[(x_min, y_min)]

                for ii, x_value in enumerate(x):
                    for jj, y_value in enumerate(y):
                        z_mesh[ii, jj] = np.abs(xy_to_metric[(x_value, y_value)] - best)

                z_mesh = ma.masked_less_equal(z_mesh, 0)

                for log_x, log_y, log_z in itertools.product((True, False), repeat = 3):
                    log_str = ''
                    if any((log_x, log_y, log_z)):
                        log_str = '__log'
                        if log_x:
                            log_str += 'X'
                        if log_y:
                            log_str += 'Y'
                        if log_z:
                            log_str += 'Z'

                    si.vis.xyz_plot(
                        f'2d__{plot_name}{log_str}',
                        x_mesh,
                        y_mesh,
                        z_mesh,
                        x_unit = x_parameter_unit,
                        y_unit = y_parameter_unit,
                        x_label = x_parameter_name,
                        y_label = y_parameter_name,
                        z_label = ionization_metric_name + ' (Diff from Best)',
                        x_log_axis = log_x,
                        y_log_axis = log_y,
                        z_log_axis = log_z,
                        z_lower_limit = None,
                        z_upper_limit = None,
                        target_dir = self.summaries_dir,
                    )
