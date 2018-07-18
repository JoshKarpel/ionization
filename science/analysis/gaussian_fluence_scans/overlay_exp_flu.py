#!/usr/bin/env python

import logging
import os
import itertools

import numpy as np

import simulacra as si
import simulacra.cluster as clu
import simulacra.units as u

import ionization as ion
import ionization.cluster as iclu

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

LOGMAN = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.DEBUG)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)


def ionization_from_fluence(fluence, gamma):
    return np.exp(-2 * gamma * fluence / (u.epsilon_0 * u.c * (u.atomic_electric_field ** 2) * u.atomic_time))


def make_scans_with_overlay(job_processor, gamma = 1):
    for line_parameter, scan_parameter, plot_parameters in job_processor._select_two_parameters():
        if scan_parameter != 'fluence' or line_parameter == 'phase':
            continue

        try:
            line_parameter_set, scan_parameter_set = job_processor.parameter_set(line_parameter), job_processor.parameter_set(scan_parameter)
        except AttributeError:  # parameter is not tracked
            continue

        if any((len(scan_parameter_set) < 10,  # not enough x data, or too many lines
                len(line_parameter_set) > 8,)):
            continue

        plot_parameter_to_set = job_processor._get_parameters_to_parameter_sets(plot_parameters)
        if len(plot_parameter_to_set) == 0:  # this sometimes happens on heatmap jobs?
            continue

        plot_parameters = tuple(plot_parameter_to_set.keys())
        plot_parameter_value_groups = (plot_parameter_values for plot_parameter_values in itertools.product(*plot_parameter_to_set.values()))

        for plot_parameter_values in plot_parameter_value_groups:
            base_selector = dict(zip(plot_parameters, plot_parameter_values))
            ionization_metric = 'final_initial_state_overlap'
            x_data, y_data, line_labels = job_processor._get_lines(line_parameter, scan_parameter, ionization_metric, base_selector)
            if any(x[0] == x[-1] for x in x_data):
                continue

            str_for_filename, str_for_plot = job_processor._get_plot_parameter_strings(plot_parameters, plot_parameter_values)
            filename = '__'.join((
                job_processor.name,
                '1d',
                ionization_metric,
                str_for_filename,
                f'scanning_{scan_parameter}_grouped_by_{line_parameter}',
            ))
            title = job_processor.name + '\n' + str_for_plot

            length = len(x_data)

            x_data.append(x_data[-1])
            y_data.append(ionization_from_fluence(x_data[-1], gamma))
            line_labels.append(f'$\gamma = {gamma}$')

            colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02']
            line_kwargs = [{'color': colors[i]} for i in range(length)]
            line_kwargs.append({'color': 'black', 'linestyle': '--'})
            # line_kwargs += [{'color': colors[i], 'linestyle': '--'} for i in range(length)]

            for log_x, log_y in itertools.product((True, False), repeat = 2):
                if any((log_x and scan_parameter == 'phase',
                        log_x and any(np.any(x <= 0) for x in x_data),
                        log_y and any(np.any(y <= 0) for y in y_data),)):
                    continue

                if log_y:
                    y_lower_limit = 1e-15
                else:
                    y_lower_limit = 0

                if log_x or not log_y:
                    continue

                log_str = iclu.format_log_str(
                    (log_x, log_y),
                    ('X', 'Y'),
                )

                si.vis.xxyy_plot(
                    filename + log_str,
                    x_data,
                    y_data,
                    line_labels = line_labels,
                    line_kwargs = line_kwargs,
                    x_label = fr'${iclu.PARAMETER_TO_SYMBOL[scan_parameter]}$',
                    x_unit = iclu.PARAMETER_TO_UNIT_NAME[scan_parameter],
                    x_log_axis = log_x,
                    y_label = iclu.format_ionization_metric_name(ionization_metric),
                    y_lower_limit = y_lower_limit,
                    y_upper_limit = 1,
                    y_log_axis = log_y,
                    title = title,
                    legend_on_right = True if len(line_labels) > 5 else False,
                    target_dir = os.path.join(OUT_DIR, job_processor.name, f'gamma={gamma}'),
                    img_format = 'png',
                )


if __name__ == '__main__':
    with LOGMAN as logger:
        jp_names = (
            'gaussian_fluence_scan_2.job',
            'gaussian_amplitude_scan.job',
        )

        for jp_name in jp_names:
            jp = clu.JobProcessor.load(jp_name)
            for gamma in (1, 1.16, 1.2):
                make_scans_with_overlay(jp, gamma)
