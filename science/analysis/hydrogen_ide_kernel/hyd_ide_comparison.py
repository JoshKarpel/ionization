import logging
import os

import numpy as np

import simulacra as si
from simulacra.units import *

import ionization as ion
import ionization.cluster as clu

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

LOGMAN = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.DEBUG)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)

def version_1():
    jp_hyd = clu.PulseJobProcessor.load('job_processors/hyd__pw_scan_v2__50-1000as_3flus_3phis__sinc.job')
    jp_ide = clu.PulseJobProcessor.load('job_processors/ide__K_hyd__pw_scan__fast.job')

    print(jp_hyd)
    print(jp_ide)

    phases = sorted(jp_ide.parameter_set('phase'))
    fluences = sorted(jp_ide.parameter_set('fluence'))
    ide_alpha = .6

    styles = ['-', ':', '--']
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']

    phase_to_style = dict(zip(phases, styles))
    fluence_to_color = dict(zip(fluences, colors))
    color_patches = [mpatches.Patch(color = color, label = fr'$ H = {uround(fluence, Jcm2)} \, \mathrm{{J/cm^2}} $')
                     for fluence, color in fluence_to_color.items()]

    phases_latex = [r'0', r'\pi / 4', r'\pi / 2']
    style_patches = [mlines.Line2D([], [], color = 'black', linestyle = style, linewidth = 3, label = fr'$ \varphi = {phase_latex} $')
                     for phase, style, phase_latex in zip(phases, styles, phases_latex)]

    legend_handles = color_patches + style_patches

    results_by_phase_and_fluence__hyd = {(phase, fluence): jp_hyd.select_by_kwargs(phase = phase, fluence = fluence)
                                         for phase in phases for fluence in fluences}
    results_by_phase_and_fluence__ide = {(phase, fluence): jp_ide.select_by_kwargs(phase = phase, fluence = fluence)
                                         for phase in phases for fluence in fluences}

    metrics = ['final_initial_state_overlap', 'final_bound_state_overlap']
    extra_line_kwargs = dict(
        linewidth = 1,
        markevery = 10,
    )

    line_kwargs = [{'linestyle': phase_to_style[phase], 'color': fluence_to_color[fluence], 'marker': 'D', 'markersize': 2.5, **extra_line_kwargs}
                   for phase, fluence in results_by_phase_and_fluence__hyd.keys()]
    line_kwargs += [{'linestyle': phase_to_style[phase], 'color': fluence_to_color[fluence], 'marker': '+', 'markersize': 4, **extra_line_kwargs}
                    for phase, fluence in results_by_phase_and_fluence__ide.keys()]

    for log_y in [True, False]:
        postfix = ''
        if log_y:
            postfix += '_logY'

        for metric in metrics:
            si.vis.xxyy_plot(
                f'hyd_ide_comparison__pulse_width_scan__sinc__{metric}' + postfix,
                [
                    *[[r.pulse_width for r in results] for results in results_by_phase_and_fluence__hyd.values()],
                    *[[r.pulse_width for r in results] for results in results_by_phase_and_fluence__ide.values()],
                ],
                [
                    *[[getattr(r, metric) for r in results] for results in results_by_phase_and_fluence__hyd.values()],
                    *[[getattr(r, metric) for r in results] for results in results_by_phase_and_fluence__ide.values()],
                ],
                line_kwargs = line_kwargs,
                title = 'Pulse Width Scan Comparison: Sinc Pulse', title_offset = 1.075,
                x_label = r'Pulse Width $\tau$',
                x_unit = 'asec',
                y_label = metric.replace('final_', '').replace('_', ' ').title(),
                y_log_axis = log_y, y_log_pad = 2,
                legend_on_right = True,
                legend_kwargs = {
                    'handles': legend_handles,
                },
                **PLOT_KWARGS,
                x_upper_limit = 500 * asec,
                y_lower_limit = 1e-5 if log_y else 0,
            )

if __name__ == '__main__':
    with LOGMAN as logger:
        version_1()
