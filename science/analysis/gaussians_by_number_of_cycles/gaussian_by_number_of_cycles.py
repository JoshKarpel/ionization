import logging
import os
import itertools

import numpy as np

import simulacra as si
from simulacra.units import *

import ionization as ion
import ionization.cluster as clu

import matplotlib as mpl
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


def get_n_cycles(result):
    return result.electric_potential[0].number_of_cycles


def scan_plots_absolute():
    jp = clu.PulseJobProcessor.load('job_processors/hyd__pw_scan_gaussian.job')

    n_cycles = sorted(set(get_n_cycles(r) for r in jp.data.values()))
    fluences = sorted(jp.parameter_set('fluence'))
    phases = sorted(jp.parameter_set('phase'))

    results_by_flu_cep = {(cep, flu): jp.select_by_kwargs(fluence = flu, phase = cep)
                          for cep in phases
                          for flu in fluences}

    results_by_cyc = {cyc: {(cep, flu): list(r for r in results_by_flu_cep[cep, flu] if get_n_cycles(r) == cyc)
                            for cep in phases
                            for flu in fluences}
                      for cyc in n_cycles}

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

    extra_line_kwargs = dict(
        linewidth = 1,
    )
    line_kwargs = [{'linestyle': phase_to_style[phase], 'color': fluence_to_color[fluence], **extra_line_kwargs}
                   for phase in phases for fluence in fluences]

    metrics = ['final_initial_state_overlap', 'final_bound_state_overlap']

    for cyc, metric, log_y in itertools.product(n_cycles, metrics, [False, True]):
        postfix = ''
        if log_y:
            postfix += '__logY'

        si.vis.xxyy_plot(
            f'GaussianPulseWidthScan__Nc={cyc}__{metric}' + postfix,
            [
                *[[r.pulse_width for r in results] for results in results_by_cyc[cyc].values()]
            ],
            [
                *[[getattr(r, metric) for r in results] for results in results_by_cyc[cyc].values()]
            ],
            line_kwargs = line_kwargs,
            title = fr'Pulse Width Scan Comparison: Gaussian Pulse, $N_c = {cyc}$', title_offset = 1.075,
            x_label = r'Pulse Width $\tau$',
            x_unit = 'asec',
            y_label = metric.replace('final_', '').replace('_', ' ').title(),
            y_log_axis = log_y, y_log_pad = 2,
            legend_on_right = True,
            legend_kwargs = {
                'handles': legend_handles,
            },
            y_lower_limit = None if log_y else 0,
            **PLOT_KWARGS,
        )


def scan_plots_modulation_depth():
    raise NotImplementedError


def pulse_comparison(pulse_width = 200 * asec, fluence = 1 * Jcm2, phase = 0, n_cycles = (2, 3, 4)):
    pulses = [ion.GaussianPulse.from_number_of_cycles(pulse_width = pulse_width, fluence = fluence, phase = phase, number_of_cycles = n) for n in n_cycles]

    times = np.linspace(-4 * pulse_width, 4 * pulse_width, 1e4)

    si.vis.xy_plot(
        f'pulse_comparison__pw={uround(pulse_width, asec)}as_flu={uround(fluence, Jcm2)}_cep={uround(phase, pi)}pi',
        times,
        *[pulse.get_electric_field_amplitude(times) for pulse in pulses],
        line_labels = [rf'$N_c = {n}$' for n in n_cycles],
        x_label = r'Time $t$',
        x_unit = 'asec',
        y_label = rf'Electric Field $ {ion.LATEX_EFIELD}(t) $',
        y_unit = 'atomic_electric_field',
        title = rf'Gaussian Pulse Comparison',
        **PLOT_KWARGS,
    )


if __name__ == '__main__':
    with LOGMAN as logger:
        scan_plots_absolute()
        pulse_comparison()
