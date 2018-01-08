import itertools
import logging
from copy import copy

import numpy as np
import numpy.ma as ma

import simulacra as si
import simulacra.cluster as clu
import simulacra.units as u

from . import core, ide, jobutils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

PARAMETER_TO_SYMBOL = {
    'pulse_width': r'\tau',
    'fluence': r'H',
    'amplitude': r'\mathcal{E}_0',
    'phase': r'\varphi',
    'delta_r': r'\Delta r',
    'delta_t': r'\Delta t',
}

PARAMETER_TO_UNIT_NAME = {
    'pulse_width': 'asec',
    'fluence': 'Jcm2',
    'amplitude': 'atomic_electric_field',
    'phase': 'pi',
    'delta_r': 'bohr_radius',
    'delta_t': 'asec',
}


class PulseParameterScanMixin:
    def make_summary_plots(self):
        super().make_summary_plots()

        if len(self.unprocessed_sim_names) == 0:
            logger.info(f'Generating pulse parameter scans for job {self.name}')
            self.make_pulse_parameter_scans_1d()
            self.make_pulse_parameter_scans_2d()

    def make_pulse_parameter_scans_1d(self):
        for ionization_metric in self.ionization_metrics:
            ionization_metric_name = ionization_metric.replace('_', ' ').title()

            for plot_parameter, line_parameter, scan_parameter in itertools.permutations(self.scan_parameters, r = 3):
                plot_parameter_unit, line_parameter_unit, scan_parameter_unit = PARAMETER_TO_UNIT_NAME[plot_parameter], PARAMETER_TO_UNIT_NAME[line_parameter], PARAMETER_TO_UNIT_NAME[scan_parameter]
                plot_parameter_set, line_parameter_set, scan_parameter_set = self.parameter_set(plot_parameter), self.parameter_set(line_parameter), self.parameter_set(scan_parameter)

                if any((len(scan_parameter_set) < 10,
                        len(line_parameter_set) > 8,
                        len(plot_parameter_set) > 10)):  # skip
                    logger.debug(f'Skipped plotting {scan_parameter} scan grouped by {line_parameter} at constant {plot_parameter} for job {self.name} because the scan would not be dense enough, or would have too many lines')
                    continue

                for plot_parameter_value in plot_parameter_set:
                    plot_name = f'{ionization_metric}__{plot_parameter}={u.uround(plot_parameter_value, plot_parameter_unit, 3)}{plot_parameter_unit}__scanning_{scan_parameter}_grouped_by_{line_parameter}'

                    lines = []
                    line_labels = []

                    for line_parameter_value in sorted(line_parameter_set):
                        selector = {
                            plot_parameter: plot_parameter_value,
                            line_parameter: line_parameter_value,
                        }
                        results = sorted(self.select_by_kwargs(**selector), key = lambda result: getattr(result, scan_parameter))

                        lines.append(np.array([getattr(result, ionization_metric) for result in results]))

                        label = fr"${PARAMETER_TO_SYMBOL[line_parameter]} \, = {u.uround(line_parameter_value, line_parameter_unit)} \, {u.UNIT_NAME_TO_LATEX[line_parameter_unit]}$"
                        line_labels.append(label)

                    x = np.array([getattr(result, scan_parameter) for result in results])

                    s_x = sorted(x)
                    if s_x[0] == s_x[-1]:
                        continue

                    for log_x, log_y in itertools.product((True, False), repeat = 2):
                        if scan_parameter == 'phase' and log_x:
                            continue

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
                            f'{self.name}__1d__{plot_name}{log_str}',
                            x,
                            *lines,
                            line_labels = line_labels,
                            x_label = fr'${PARAMETER_TO_SYMBOL[scan_parameter]}$',
                            x_unit = scan_parameter_unit,
                            x_log_axis = log_x,
                            y_label = ionization_metric_name,
                            y_lower_limit = y_lower_limit,
                            y_upper_limit = y_upper_limit,
                            y_log_axis = log_y,
                            title = self.name + '\n' + fr"${PARAMETER_TO_SYMBOL[plot_parameter]} \, = {u.uround(plot_parameter_value, plot_parameter_unit)} \, {u.UNIT_NAME_TO_LATEX[plot_parameter_unit]}$",
                            legend_on_right = True if len(line_labels) > 5 else False,
                            target_dir = self.summaries_dir,
                        )

    def make_pulse_parameter_scans_2d(self):
        for ionization_metric in self.ionization_metrics:
            ionization_metric_name = ionization_metric.replace('_', ' ').title()

            for plot_parameter in self.scan_parameters:
                for x_parameter, y_parameter in itertools.combinations((p for p in self.scan_parameters if p != plot_parameter), r = 2):
                    plot_parameter_unit, x_parameter_unit, y_parameter_unit = PARAMETER_TO_UNIT_NAME[plot_parameter], PARAMETER_TO_UNIT_NAME[x_parameter], PARAMETER_TO_UNIT_NAME[y_parameter]
                    plot_parameter_set, x_parameter_set, y_parameter_set = self.parameter_set(plot_parameter), self.parameter_set(x_parameter), self.parameter_set(y_parameter)

                    if any((len(x_parameter_set) < 10,
                            len(y_parameter_set) < 10)):  # skip
                        logger.debug(f'Skipped plotting {x_parameter} vs {y_parameter} at constant {plot_parameter} for job {self.name} because it would not be dense enough')
                        continue

                    x, y = np.array(sorted(x_parameter_set)), np.array(sorted(y_parameter_set))
                    x_mesh, y_mesh = np.meshgrid(x, y, indexing = 'ij')

                    for plot_parameter_value in plot_parameter_set:
                        plot_name = f'{ionization_metric}__{plot_parameter}={u.uround(plot_parameter_value, plot_parameter_unit, 3)}{plot_parameter_unit}__{x_parameter}_vs_{y_parameter}'

                        results = self.select_by_kwargs(**{plot_parameter: plot_parameter_value})

                        xy_to_metric = {(getattr(r, x_parameter), getattr(r, y_parameter)): getattr(r, ionization_metric) for r in results}
                        z_mesh = np.empty_like(x_mesh)

                        try:
                            for ii, x_value in enumerate(x):
                                for jj, y_value in enumerate(y):
                                    z_mesh[ii, jj] = xy_to_metric[x_value, y_value]
                        except KeyError:
                            logger.debug(f'Skipped plotting {x_parameter} vs {y_parameter} at constant {plot_parameter} for job {self.name} due to alignment error (job is probably not a heatmap)')
                            continue

                        for log_x, log_y, log_z in itertools.product((True, False), repeat = 3):
                            if (x_parameter == 'phase' and log_x) or (y_parameter == 'phase' and log_y):  # skip log phase plots
                                continue

                            log_str = ''
                            if any((log_x, log_y, log_z)):
                                log_str = '__log'
                                if log_x:
                                    log_str += 'X'
                                if log_y:
                                    log_str += 'Y'
                                if log_z:
                                    log_str += 'Z'

                            if log_x and not np.all(x_mesh > 0):
                                continue
                            if log_y and not np.all(y_mesh > 0):
                                continue

                            si.vis.xyz_plot(
                                f'{self.name}__2d__{plot_name}{log_str}',
                                x_mesh, y_mesh, z_mesh,
                                x_label = fr'${PARAMETER_TO_SYMBOL[x_parameter]}$',
                                y_label = fr'${PARAMETER_TO_SYMBOL[y_parameter]}$',
                                z_label = ionization_metric_name,
                                x_unit = x_parameter_unit,
                                y_unit = y_parameter_unit,
                                x_log_axis = log_x,
                                y_log_axis = log_y,
                                z_log_axis = log_z,
                                z_upper_limit = 1,
                                title = self.name + '\n' + fr'${PARAMETER_TO_SYMBOL[plot_parameter]} \, = {u.uround(plot_parameter_value, plot_parameter_unit, 3)} \, {u.UNIT_NAME_TO_LATEX[plot_parameter_unit]}$',
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

        state_overlaps = sim.state_overlaps_vs_time

        self.final_norm = copy(sim.norm_vs_time[-1])
        self.final_initial_state_overlap = copy(state_overlaps[sim.spec.initial_state][-1])
        self.final_bound_state_overlap = copy(sum(state_overlaps[s][-1] for s in sim.bound_states))
        self.final_free_state_overlap = copy(sum(state_overlaps[s][-1] for s in sim.free_states))

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
    scan_parameters = ['pulse_width', 'fluence', 'phase', 'amplitude']


class MeshJobProcessor(PulseJobProcessor):
    simulation_type = core.ElectricFieldSimulation
    simulation_result_type = MeshSimulationResult

    ionization_metrics = ['final_norm', 'final_initial_state_overlap', 'final_bound_state_overlap']


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

    def make_pulse_parameter_scans_1d(self):
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

    def make_pulse_parameter_scans_2d(self):
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
