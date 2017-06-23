import functools
import logging
import os
import sys
from copy import deepcopy

import matplotlib
import numpy as np
import scipy.optimize as optimize
from tqdm import tqdm

matplotlib.use('pgf')

import simulacra as si
import ionization as ion
import ionization.integrodiff as ide
from simulacra.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

logman = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.INFO)

pgf_with_latex = {  # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
    "text.usetex": True,  # use LaTeX to write all text
    "text.latex.unicode": True,
    "font.family": "serif",
    "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    # "axes.labelsize": 11,  # LaTeX default is 10pt font.
    # "font.size": 11,
    # "legend.fontsize": 10,  # Make the legend/label fonts a little smaller
    # "xtick.labelsize": 9,
    # "ytick.labelsize": 9,
    # "figure.figsize": si.vis._get_fig_dims(0.95),  # default fig size of 0.95 \textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",  # use utf8 fonts because your computer can handle it :)
        r"\usepackage[T1]{fontenc}",  # plots will be generated using this preamble
    ]
}
matplotlib.rcParams.update(pgf_with_latex)

import matplotlib.pyplot as plt

PPT_WIDTH = 13 + (1 / 3)
PPT_HEIGHT = 7.5

PPT_ASPECT_RATIO = PPT_HEIGHT / PPT_WIDTH
PPT_WIDTH_PTS = 960

FIGMAN_KWARGS = dict(
    fig_width_pts = PPT_WIDTH_PTS,
    aspect_ratio = PPT_ASPECT_RATIO,
)

PLOT_KWARGS = dict(
    img_format = 'png',
    fig_dpi_scale = 3,
    target_dir = OUT_DIR,
)

BIG_SINE_WAVE = ion.SineWave.from_photon_energy(20 * eV, 1 * atomic_electric_field)


def run(spec):
    sim = spec.to_simulation()
    sim.run_simulation()
    return sim


def save_figure(filename):
    # si.vis.save_current_figure(filename, target_dir = OUT_DIR, img_format = 'pdf')
    # si.vis.save_current_figure(filename, target_dir = OUT_DIR, img_format = 'pgf')
    si.vis.save_current_figure(filename, target_dir = OUT_DIR, img_format = 'png')


def get_func_name():
    return sys._getframe(1).f_code.co_name


def title_bg():
    sim = ion.CylindricalSliceSpecification(
        'cylindrical_slice_mesh',
        electric_potential = BIG_SINE_WAVE,
        time_initial = 0, time_final = 200 * asec,
        z_bound = 40 * bohr_radius,
    ).to_simulation()
    sim.run_simulation(progress_bar = True)

    sim.mesh.plot_g(
        plot_limit = 25 * bohr_radius,
        tick_label_size = 20,
        axis_label_size = 35,
        title = '',
        grid_kwargs = {'linewidth': 2},
        **FIGMAN_KWARGS,
        **PLOT_KWARGS
    )
    sim.mesh.plot_g2(
        plot_limit = 25 * bohr_radius,
        tick_label_size = 20,
        axis_label_size = 35,
        title = '',
        show_colorbar = False,
        grid_kwargs = {'linewidth': 2},
        **FIGMAN_KWARGS,
        **PLOT_KWARGS
    )

    for which in ('g', 'g2'):
        with si.vis.FigureManager(
                f'title_bg__{which}',
                tight_layout = False,
                **FIGMAN_KWARGS,
                **PLOT_KWARGS,
        ) as figman:
            fig = figman.fig
            ax = fig.add_axes([0, 0, 1, 1])

            getattr(sim.mesh, f'attach_{which}_to_axis')(ax, plot_limit = 20 * bohr_radius)

            ax.axis('off')


def spherical_harmonic_mesh():
    sim = ion.SphericalHarmonicSpecification(
        'spherical_harmonic_mesh',
        electric_potential = BIG_SINE_WAVE,
        time_initial = 0, time_final = 200 * asec,
        r_points = 500,
        r_bound = 60 * bohr_radius,
        l_bound = 40,
    ).to_simulation()
    sim.run_simulation(progress_bar = True)

    sim.mesh.plot_g_repr(
        plot_limit = 25 * bohr_radius,
        tick_label_size = 20,
        axis_label_size = 35,
        title = '',
        grid_kwargs = {'linewidth': 2},
        **FIGMAN_KWARGS,
        **PLOT_KWARGS,
    )
    sim.mesh.plot_g(
        plot_limit = 25 * bohr_radius,
        tick_label_size = 20,
        title_size = 30,
        grid_kwargs = {'linewidth': 2},
        **FIGMAN_KWARGS,
        **PLOT_KWARGS
    )
    sim.mesh.plot_g2(
        plot_limit = 25 * bohr_radius,
        tick_label_size = 20,
        title_size = 30,
        show_colorbar = False,
        grid_kwargs = {'linewidth': 2},
        **FIGMAN_KWARGS,
        **PLOT_KWARGS
    )


def efield_and_afield():
    times = np.linspace(-1000 * asec, 1000 * asec, 1e5)

    bsw = deepcopy(BIG_SINE_WAVE)
    bsw.window = ion.SymmetricExponentialTimeWindow(window_time = 800 * asec, window_width = 30 * asec)

    si.vis.xy_plot(
        'e_and_a_fields',
        times,
        bsw.get_electric_field_amplitude(times) / atomic_electric_field,
        proton_charge * bsw.get_vector_potential_amplitude_numeric_cumulative(times) / atomic_momentum,
        line_labels = [f'$ {ion.LATEX_EFIELD}(t) $', f'$ e \, {ion.LATEX_AFIELD}(t) $'],
        line_kwargs = [{'linewidth': 3}, {'linewidth': 3, 'linestyle': '--'}],
        x_label = r'$t$', x_unit = 'asec',
        y_label = r'Amplitude (a.u.)',
        y_lower_limit = -1.6, y_upper_limit = 1,
        title = r'Windowed Sine Wave Pulse w/ $E = 10 \, \mathrm{eV}$',
        font_size_axis_labels = 35,
        font_size_tick_labels = 20,
        font_size_legend = 25,
        font_size_title = 35,
        **FIGMAN_KWARGS,
        **PLOT_KWARGS,
    )

    si.vis.xy_plot(
        'e_field',
        times,
        bsw.get_electric_field_amplitude(times) / atomic_electric_field,
        line_labels = [f'$ {ion.LATEX_EFIELD}(t) $'],
        line_kwargs = [{'linewidth': 3}],
        x_label = r'$t$', x_unit = 'asec',
        y_label = r'Amplitude (a.u.)',
        y_lower_limit = -1.6, y_upper_limit = 1,
        title = r'Windowed Sine Wave Pulse w/ $E = 10 \, \mathrm{eV}$',
        font_size_axis_labels = 35,
        font_size_tick_labels = 20,
        font_size_legend = 25,
        font_size_title = 35,
        **FIGMAN_KWARGS,
        **PLOT_KWARGS,
    )


def sinc_pulse():
    pw = 200 * asec
    flu = 1 * Jcm2

    times = np.linspace(-10 * pw, 10 * pw, 1e5)

    pulse = ion.SincPulse(pulse_width = pw, fluence = flu)
    pulse.window = ion.SymmetricExponentialTimeWindow(window_time = pw * 8, window_width = .2 * pw)

    pulse = ion.DC_correct_electric_potential(pulse, times)

    si.vis.xy_plot(
        'sinc_pulse',
        times,
        pulse.get_electric_field_amplitude(times) / atomic_electric_field,
        proton_charge * pulse.get_vector_potential_amplitude_numeric_cumulative(times) / atomic_momentum,
        line_labels = [f'$ {ion.LATEX_EFIELD}(t) $', f'$ e \, {ion.LATEX_AFIELD}(t) $'],
        line_kwargs = [{'linewidth': 3}, {'linewidth': 3, 'linestyle': '--'}],
        x_label = r'$t$', x_unit = 'asec',
        title = fr'Windowed Sinc Pulse w/ $\tau = {uround(pw, asec, 0)} \, \mathrm{{as}}, \, H = {uround(flu, Jcm2)} \, \mathrm{{J/cm^2}} $',
        font_size_axis_labels = 35,
        font_size_tick_labels = 20,
        font_size_legend = 25,
        font_size_title = 35,
        **FIGMAN_KWARGS,
        **PLOT_KWARGS,
    )


def gaussian_pulse():
    pw = 200 * asec
    flu = 1 * Jcm2

    times = np.linspace(-10 * pw, 10 * pw, 1e5)

    pulse = ion.GaussianPulse(pulse_width = pw, fluence = flu)
    pulse.window = ion.SymmetricExponentialTimeWindow(window_time = pw * 5, window_width = .2 * pw)

    pulse = ion.DC_correct_electric_potential(pulse, times)

    si.vis.xy_plot(
        'gaussian_pulse',
        times,
        pulse.get_electric_field_amplitude(times) / atomic_electric_field,
        proton_charge * pulse.get_vector_potential_amplitude_numeric_cumulative(times) / atomic_momentum,
        line_labels = [f'$ {ion.LATEX_EFIELD}(t) $', f'$ e \, {ion.LATEX_AFIELD}(t) $'],
        line_kwargs = [{'linewidth': 3}, {'linewidth': 3, 'linestyle': '--'}],
        x_label = r'$t$', x_unit = 'asec',
        y_label = r'Amplitude (a.u.)',
        title = fr'Gaussian Pulse w/ $\tau = {uround(pw, asec, 0)} \, \mathrm{{as}}, \, H = {uround(flu, Jcm2)} \, \mathrm{{J/cm^2}} $',
        font_size_axis_labels = 35,
        font_size_tick_labels = 20,
        font_size_legend = 25,
        font_size_title = 35,
        **FIGMAN_KWARGS,
        **PLOT_KWARGS,
    )


def pulse_cep_movie(pulse_type = ion.GaussianPulse, prefix = 'Gaussian Pulse'):
    pw = 200 * asec
    flu = 1 * Jcm2
    phases = np.linspace(0, twopi, 600)
    # phase_frames = list(range(len(phases)))

    times = np.linspace(-10 * pw, 10 * pw, 1000)

    @si.utils.memoize
    def get_pulse_by_phase(phase):
        pulse = pulse_type(pulse_width = pw,
                           fluence = flu,
                           phase = phase,
                           window = ion.SymmetricExponentialTimeWindow(window_time = pw * 8,
                                                                       window_width = .2 * pw))

        return ion.DC_correct_electric_potential(pulse, times)

    def efield(times, phase):
        return get_pulse_by_phase(phase).get_electric_field_amplitude(times) / atomic_electric_field

    def efield_intensity(times, phase):
        return epsilon_0 * c * (np.abs(efield(times, phase) * atomic_electric_field) ** 2)

    def afield(times, phase):
        return get_pulse_by_phase(phase).get_vector_potential_amplitude_numeric_cumulative(times) * proton_charge / atomic_momentum

    si.vis.xyt_plot(
        f'cep_movie__{pulse_type.__name__}_efield',
        times,
        phases,
        efield, afield,
        line_labels = [fr'$ {ion.LATEX_EFIELD}(t) $', fr'$ e \, {ion.LATEX_AFIELD}(t) $'],
        line_kwargs = [None, {'linestyle': '--'}],
        x_label = r'$ t $', x_unit = 'asec',
        y_label = r'Amplitude (a.u.)',
        t_fmt_string = r'$ \varphi = {}\pi \; {} $', t_unit = 'rad',
        title = fr'{prefix} w/ $\tau = {uround(pw, asec, 0)} \, \mathrm{{as}}, \, H = {uround(flu, Jcm2)} \, \mathrm{{J/cm^2}} $',
        progress_bar = True,
        length = 10,
        target_dir = OUT_DIR,
    )

    si.vis.xyt_plot(
        f'cep_movie__{pulse_type.__name__}_intensity',
        times,
        phases,
        efield_intensity,
        x_label = r'$ t $', x_unit = 'asec',
        y_label = r'$ P(t) $', y_unit = 'atomic_intensity',
        t_fmt_string = r'$ \varphi = {}\pi \; {} $', t_unit = 'rad',
        title = fr'{prefix} w/ $\tau = {uround(pw, asec, 0)} \, \mathrm{{as}}, \, H = {uround(flu, Jcm2)} \, \mathrm{{J/cm^2}} $',
        progress_bar = True,
        length = 10,
        target_dir = OUT_DIR,
    )


if __name__ == '__main__':
    with logman as logger:
        figures = [
            sinc_pulse,
            gaussian_pulse,
            efield_and_afield,
            title_bg,
            spherical_harmonic_mesh,
            functools.partial(pulse_cep_movie, pulse_type = ion.GaussianPulse, prefix = 'Gaussian Pulse'),
            functools.partial(pulse_cep_movie, pulse_type = ion.SincPulse, prefix = 'Sinc Pulse'),
        ]

        for fig in tqdm(figures):
            fig()
