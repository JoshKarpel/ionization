import functools
import logging
import itertools
import os
import sys
from copy import deepcopy
import collections

import matplotlib
import numpy as np
import numpy.fft as nfft
import scipy.optimize as optimize
from tqdm import tqdm

matplotlib.use('pgf')

import simulacra as si
import simulacra.cluster as clu
import ionization as ion
import ionization.integrodiff as ide
import ionization.cluster as iclu
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
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

FULL_PAGE_KWARGS = dict(
    fig_width = si.vis.PPT_WIDESCREEN_WIDTH,
    fig_height = si.vis.PPT_WIDESCREEN_HEIGHT,
    img_format = 'png',
    fig_dpi_scale = 6,
)

QUARTER_PAGE_KWARGS = dict(
    fig_width = 13 / 2,
    fig_height = 6.5 / 2,
    # img_format = 'svg',
    img_format = 'png',
    fig_dpi_scale = 6,
)

STILL_FIGMAN_KWARGS_SVG = dict(
    fig_width = si.vis.PPT_WIDESCREEN_WIDTH,
    fig_height = si.vis.PPT_WIDESCREEN_HEIGHT,
    # img_format = 'svg',
    img_format = 'png',
    fig_dpi_scale = 6,
)

ANIMATED_FIGURE_KWARGS = dict(
    fig_width = si.vis.PPT_WIDESCREEN_WIDTH,
    fig_height = si.vis.PPT_WIDESCREEN_HEIGHT,
    fig_dpi_scale = 2,
)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
)

BETTER_GRID_KWARGS = {
    **si.vis.GRID_KWARGS,
    'linewidth': 1,
}

BIG_LINEWIDTH = 4

BIG_FONTS = dict(
    font_size_axis_labels = 35,
    font_size_tick_labels = 25,
    font_size_legend = 30,
    font_size_title = 35,
)

BIG_SINE_WAVE = ion.SineWave.from_photon_energy(20 * eV, 1 * atomic_electric_field)


def run(spec):
    sim = spec.to_simulation()
    sim.run_simulation()
    return sim


def save_figure(filename):
    # si.vis.save_current_figure(filename, target_dir = OUT_DIR, img_format = 'pdf')
    # si.vis.save_current_figure(filename, target_dir = OUT_DIR, img_format = 'pgf')
    si.vis.save_current_figure(filename, **PLOT_KWARGS)


def get_func_name():
    return sys._getframe(1).f_code.co_name


def title_bg():
    sim = ion.CylindricalSliceSpecification(
        'cylindrical_slice_mesh',
        electric_potential = BIG_SINE_WAVE,
        time_initial = 0, time_final = 200 * asec,
        z_bound = 40 * bohr_radius,
        rho_bound = 40 * bohr_radius,
        z_points = 40 * 10 * 2,
        rho_points = 40 * 10,
    ).to_simulation()
    sim.run_simulation(progress_bar = True)

    sim.mesh.plot_g(
        plot_limit = 25 * bohr_radius,
        tick_label_size = 20,
        axis_label_size = 35,
        show_title = False,
        grid_kwargs = {'linewidth': 2},
        **FULL_PAGE_KWARGS,
        **PLOT_KWARGS
    )
    sim.mesh.plot_g2(
        plot_limit = 25 * bohr_radius,
        tick_label_size = 20,
        axis_label_size = 35,
        show_title = False,
        show_colorbar = False,
        grid_kwargs = {'linewidth': 2},
        **FULL_PAGE_KWARGS,
        **PLOT_KWARGS
    )

    for which in ('g', 'g2'):
        with si.vis.FigureManager(f'title_bg__{which}',
                                  tight_layout = False,
                                  **FULL_PAGE_KWARGS,
                                  **PLOT_KWARGS) as figman:
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
        **FULL_PAGE_KWARGS,
        **PLOT_KWARGS,
    )
    sim.mesh.plot_g(
        plot_limit = 25 * bohr_radius,
        tick_label_size = 20,
        title_size = 30,
        grid_kwargs = {'linewidth': 2},
        **FULL_PAGE_KWARGS,
        **PLOT_KWARGS
    )
    sim.mesh.plot_g2(
        plot_limit = 25 * bohr_radius,
        tick_label_size = 20,
        title_size = 30,
        show_colorbar = False,
        grid_kwargs = {'linewidth': 2},
        **FULL_PAGE_KWARGS,
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
        line_kwargs = [{'linewidth': BIG_LINEWIDTH}, {'linewidth': BIG_LINEWIDTH, 'linestyle': '--'}],
        x_label = r'Time $t$', x_unit = 'asec',
        y_label = r'Amplitude (a.u.)',
        y_lower_limit = -1.6, y_upper_limit = 1,
        title = r'Windowed Sine Wave Pulse w/ $E = 10 \, \mathrm{eV}$',
        font_size_axis_labels = 35,
        font_size_tick_labels = 20,
        font_size_legend = 25,
        font_size_title = 35,
        legend_kwargs = {'loc': 'lower left'},
        grid_kwargs = BETTER_GRID_KWARGS,
        **STILL_FIGMAN_KWARGS_SVG,
        **PLOT_KWARGS,
    )

    si.vis.xy_plot(
        'e_field',
        times,
        bsw.get_electric_field_amplitude(times) / atomic_electric_field,
        line_labels = [f'$ {ion.LATEX_EFIELD}(t) $'],
        line_kwargs = [{'linewidth': BIG_LINEWIDTH}],
        x_label = r'Time $t$', x_unit = 'asec',
        y_label = r'Amplitude (a.u.)',
        y_lower_limit = -1.6, y_upper_limit = 1,
        title = r'Windowed Sine Wave Pulse w/ $E = 10 \, \mathrm{eV}$',
        font_size_axis_labels = 35,
        font_size_tick_labels = 20,
        font_size_legend = 25,
        font_size_title = 35,
        legend_kwargs = {'loc': 'lower left'},
        grid_kwargs = BETTER_GRID_KWARGS,
        **STILL_FIGMAN_KWARGS_SVG,
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
        line_kwargs = [{'linewidth': BIG_LINEWIDTH}, {'linewidth': BIG_LINEWIDTH, 'linestyle': '--'}],
        x_label = r'Time $t$', x_unit = 'asec',
        title = fr'Windowed Sinc Pulse w/ $\tau = {uround(pw, asec, 0)} \, \mathrm{{as}}, \, H = {uround(flu, Jcm2)} \, \mathrm{{J/cm^2}} $',
        font_size_axis_labels = 35,
        font_size_tick_labels = 20,
        font_size_legend = 25,
        font_size_title = 35,
        grid_kwargs = BETTER_GRID_KWARGS,
        **FULL_PAGE_KWARGS,
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
        line_kwargs = [{'linewidth': BIG_LINEWIDTH}, {'linewidth': BIG_LINEWIDTH, 'linestyle': '--'}],
        x_label = r'Time $t$', x_unit = 'asec',
        y_label = r'Amplitude (a.u.)',
        title = fr'Gaussian Pulse w/ $\tau = {uround(pw, asec, 0)} \, \mathrm{{as}}, \, H = {uround(flu, Jcm2)} \, \mathrm{{J/cm^2}} $',
        font_size_axis_labels = 35,
        font_size_tick_labels = 20,
        font_size_legend = 25,
        font_size_title = 35,
        grid_kwargs = BETTER_GRID_KWARGS,
        **STILL_FIGMAN_KWARGS_SVG,
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

    def efield_envelope(times, phase):
        pulse = get_pulse_by_phase(phase)[0]
        return pulse.get_electric_field_envelope(times) * pulse.amplitude_time / atomic_electric_field

    def efield_envelope_negated(times, phase):
        pulse = get_pulse_by_phase(phase)[0]
        return -pulse.get_electric_field_envelope(times) * pulse.amplitude_time / atomic_electric_field

    def efield_intensity(times, phase):
        return epsilon_0 * c * (np.abs(efield(times, phase) * atomic_electric_field) ** 2)

    def afield(times, phase):
        return get_pulse_by_phase(phase).get_vector_potential_amplitude_numeric_cumulative(times) * proton_charge / atomic_momentum

    si.vis.xyt_plot(
        f'cep_movie__{pulse_type.__name__}_efield',
        times,
        phases,
        # efield_envelope,
        # efield_envelope_negated,
        efield,
        afield,
        line_labels = [
            # None,
            # fr'$ {ion.LATEX_EFIELD} Envelope $',
            fr'$ {ion.LATEX_EFIELD}(t) $',
            fr'$ e \, {ion.LATEX_AFIELD}(t) $'
        ],
        line_kwargs = [
            # {'linestyle': '-', 'linewidth': BIG_LINEWIDTH, 'color': 'black'},
            # {'linestyle': '-', 'linewidth': BIG_LINEWIDTH, 'color': 'black'},
            {'linewidth': BIG_LINEWIDTH},
            {'linestyle': '--', 'linewidth': BIG_LINEWIDTH},
        ],
        x_label = r'Time $t$', x_unit = 'asec',
        y_label = r'Amplitude (a.u.)',
        t_fmt_string = r'$ \varphi = {} \; {} $', t_unit = 'rad',
        title = fr'{prefix} w/ $\tau = {uround(pw, asec, 0)} \, \mathrm{{as}}, \, H = {uround(flu, Jcm2)} \, \mathrm{{J/cm^2}} $',
        progress_bar = True,
        title_offset = 1.1,
        length = 10,
        **BIG_FONTS,
        t_text_kwargs = {'fontsize': 25},
        grid_kwargs = BETTER_GRID_KWARGS,
        **ANIMATED_FIGURE_KWARGS,
        **PLOT_KWARGS,
    )

    si.vis.xyt_plot(
        f'cep_movie__{pulse_type.__name__}_intensity',
        times,
        phases,
        efield_intensity,
        line_kwargs = [{'linewidth': BIG_LINEWIDTH}],
        x_label = r'Time $t$', x_unit = 'asec',
        y_label = r'Intensity $ P(t) $', y_unit = 'atomic_intensity',
        t_fmt_string = r'$ \varphi = {} \; {} $', t_unit = 'rad',
        title = fr'{prefix} w/ $\tau = {uround(pw, asec, 0)} \, \mathrm{{as}}, \, H = {uround(flu, Jcm2)} \, \mathrm{{J/cm^2}} $',
        progress_bar = True,
        title_offset = 1.1,
        length = 10,
        **BIG_FONTS,
        t_text_kwargs = {'fontsize': 25},
        grid_kwargs = BETTER_GRID_KWARGS,
        **ANIMATED_FIGURE_KWARGS,
        **PLOT_KWARGS,
    )


def pulse_cep_movie_zoom(pulse_type = ion.GaussianPulse, prefix = 'Gaussian Pulse'):
    pw = 200 * asec
    flu = 1 * Jcm2
    phases = np.linspace(0, twopi, 600)
    # phase_frames = list(range(len(phases)))

    times = np.linspace(-5 * pw, 5 * pw, 1000)

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

    # si.vis.xyt_plot(
    #     f'cep_movie__{pulse_type.__name__}_efield',
    #     times,
    #     phases,
    #     efield, afield,
    #     line_labels = [fr'$ {ion.LATEX_EFIELD}(t) $', fr'$ q \, {ion.LATEX_AFIELD}(t) $'],
    #     line_kwargs = [{'linewidth': BIG_LINEWIDTH}, {'linestyle': '--', 'linewidth': BIG_LINEWIDTH}],
    #     x_label = r'Time', x_unit = 'asec',
    #     y_label = r'Amplitude (a.u.)',
    #     t_fmt_string = r'$ \varphi = {} \; {} $', t_unit = 'rad',
    #     title = fr'{prefix} w/ $\tau = {uround(pw, asec, 0)} \, \mathrm{{as}}, \, H = {uround(flu, Jcm2)} \, \mathrm{{J/cm^2}} $',
    #     progress_bar = True,
    #     length = 10,
    #     font_size_axis_labels = 35,
    #     font_size_tick_labels = 20,
    #     font_size_legend = 25,
    #     font_size_title = 35,
    #     t_text_kwargs = {'fontsize': 25},
    #     **ANIMATED_FIGURE_KWARGS,
    #     **PLOT_KWARGS,
    # )

    si.vis.xyt_plot(
        f'cep_movie__{pulse_type.__name__}_intensity__zoom',
        times,
        phases,
        efield_intensity,
        line_kwargs = [{'linewidth': BIG_LINEWIDTH}],
        x_label = r'Time $t$', x_unit = 'asec',
        y_label = r'Intensity $P(t)$', y_unit = 'atomic_intensity',
        t_fmt_string = r'$ \varphi = {} \; {} $', t_unit = 'rad',
        title = fr'{prefix} w/ $\tau = {uround(pw, asec, 0)} \, \mathrm{{as}}, \, H = {uround(flu, Jcm2)} \, \mathrm{{J/cm^2}} $',
        progress_bar = True,
        title_offset = 1.1,
        length = 10,
        font_size_axis_labels = 35,
        font_size_tick_labels = 20,
        font_size_legend = 25,
        font_size_title = 35,
        t_text_kwargs = {'fontsize': 25},
        grid_kwargs = BETTER_GRID_KWARGS,
        **ANIMATED_FIGURE_KWARGS,
        **PLOT_KWARGS,
    )


def tunneling_ionization_animation():
    z = np.linspace(-5, 50, 1000) * bohr_radius
    amplitudes = np.hstack([0, np.geomspace(0.001, .3, 599)]) * atomic_electric_field
    amplitudes = np.hstack([amplitudes, amplitudes[::-1]])

    coul_pot = -coulomb_constant * (proton_charge ** 2) / np.abs(z)
    elec_pot = -proton_charge * amplitudes[0] * z

    fm = si.vis.xy_plot(
        get_func_name(),
        z,
        coul_pot + elec_pot,
        coul_pot,
        elec_pot,
        line_labels = [r'$ V_{\mathrm{Coul}} + V_{\mathrm{Field}} $', r'$ V_{\mathrm{Coul}} $', r'$ V_{\mathrm{Field}} $'],
        line_kwargs = [{'animated': True, 'linewidth': BIG_LINEWIDTH}, {'linestyle': '--', 'linewidth': BIG_LINEWIDTH}, {'linestyle': '--', 'animated': True, 'linewidth': BIG_LINEWIDTH}],
        hlines = [ion.HydrogenBoundState(1, 0).energy], hline_kwargs = [{'linestyle': ':', 'color': 'black'}],
        y_lower_limit = -2 * hartree,
        y_upper_limit = 0,
        y_unit = 'eV',
        y_label = '$ V(z) $',
        x_unit = 'bohr_radius',
        x_label = r'$ z $',
        font_size_axis_labels = 35,
        font_size_tick_labels = 20,
        font_size_legend = 25,
        font_size_title = 35,
        grid_kwargs = BETTER_GRID_KWARGS,
        **ANIMATED_FIGURE_KWARGS,
        **PLOT_KWARGS,
        close_after_exit = False,
        save_on_exit = False,
    )

    ax = fm.fig.get_axes()[0]
    y1 = ion.HydrogenBoundState(1, 0).energy
    y2 = (coul_pot + elec_pot)
    fb = ax.fill_between(
        z / bohr_radius,
        y1 / eV,
        y2 / eV,
        where = y1 > y2,
        interpolate = True,
        facecolor = 'black',
        alpha = 0.5,
        animated = True,
    )

    def update(amp):
        nonlocal fb

        elec_pot = -proton_charge * amp * z
        y2 = (coul_pot + elec_pot)

        fm.elements['lines'][0].set_ydata(y2 / eV)
        fm.elements['lines'][2].set_ydata(elec_pot / eV)

        fb.remove()
        fb = ax.fill_between(
            z / bohr_radius,
            y1 / eV,
            y2 / eV,
            where = y1 > y2,
            interpolate = True,
            facecolor = 'black',
            alpha = 0.5,
        )

        fm.fig.draw_artist(fb)

    si.vis.animate(fm, update, amplitudes, artists = [fm.elements['lines'][0], fm.elements['lines'][2]], length = 20)


def tunneling_ionization_animation__pulse():
    z = np.linspace(-50, 50, 1000) * bohr_radius
    pulse_width = 200 * asec
    pulse = ion.SincPulse(pulse_width = pulse_width, fluence = 1 * Jcm2)
    times = np.linspace(-10 * pulse_width, 10 * pulse_width, 1200)
    amplitudes = pulse.get_electric_field_amplitude(times)

    coul_pot = -coulomb_constant * (proton_charge ** 2) / np.abs(z)
    elec_pot = -proton_charge * amplitudes[0] * z

    fm = si.vis.xy_plot(
        get_func_name(),
        z,
        coul_pot + elec_pot,
        coul_pot,
        elec_pot,
        line_labels = [r'$ V_{\mathrm{Coul}} + V_{\mathrm{Field}} $', r'$ V_{\mathrm{Coul}} $', r'$ V_{\mathrm{Field}} $'],
        line_kwargs = [{'animated': True}, {'linestyle': '--'}, {'linestyle': '--', 'animated': True}],
        hlines = [ion.HydrogenBoundState(1, 0).energy], hline_kwargs = [{'linestyle': ':', 'color': 'black'}],
        y_lower_limit = -2 * hartree,
        y_upper_limit = 0,
        y_unit = 'eV',
        y_label = '$ V(z) $',
        x_unit = 'bohr_radius',
        x_label = r'$ z $',
        grid_kwargs = BETTER_GRID_KWARGS,
        **ANIMATED_FIGURE_KWARGS,
        **PLOT_KWARGS,
        close_after_exit = False,
        save_on_exit = False,
    )

    ax = fm.fig.get_axes()[0]
    y1 = ion.HydrogenBoundState(1, 0).energy
    y2 = (coul_pot + elec_pot)
    fb = ax.fill_between(
        z / bohr_radius,
        y1 / eV,
        y2 / eV,
        where = y1 > y2,
        interpolate = True,
        facecolor = 'black',
        alpha = 0.5,
        animated = True,
    )

    def update(amp):
        nonlocal fb

        elec_pot = -proton_charge * amp * z
        y2 = (coul_pot + elec_pot)

        fm.elements['lines'][0].set_ydata(y2 / eV)
        fm.elements['lines'][2].set_ydata(elec_pot / eV)

        fb.remove()
        fb = ax.fill_between(
            z / bohr_radius,
            y1 / eV,
            y2 / eV,
            where = y1 > y2,
            interpolate = True,
            facecolor = 'black',
            alpha = 0.5,
        )

        fm.fig.draw_artist(fb)

    si.vis.animate(fm, update, amplitudes, artists = [fm.elements['lines'][0], fm.elements['lines'][2]], length = 20)


def length_ide_kernel_gaussian():
    dt = np.linspace(0, 3, 1000)
    tau = .5
    y = (1 + 1j * (dt / tau)) ** (-3 / 2)

    si.vis.xy_plot(
        get_func_name(),
        dt,
        np.abs(y),
        np.real(y),
        np.imag(y),
        line_labels = [r"$\left| K(t-t') \right|$",
                       r"$\mathrm{Re} \left\lbrace K(t-t') \right\rbrace$",
                       r"$\mathrm{Im} \left\lbrace K(t-t') \right\rbrace$"],
        line_kwargs = [{'color': 'black', 'linewidth': BIG_LINEWIDTH},
                       {'color': 'C0', 'linewidth': BIG_LINEWIDTH},
                       {'color': 'C1', 'linewidth': BIG_LINEWIDTH}],
        x_label = r"$t-t'$ ($\mathrm{\tau_{\alpha}}$)", x_unit = tau,
        font_size_axis_labels = 35,
        font_size_tick_labels = 20,
        font_size_legend = 25,
        font_size_title = 35,
        grid_kwargs = BETTER_GRID_KWARGS,
        **STILL_FIGMAN_KWARGS_SVG,
        **PLOT_KWARGS,
    )
    # with si.vis.FigureManager(get_func_name(), **FIGMAN_KWARGS, **PLOT_KWARGS) as figman:
    #     fig = figman.fig
    #     ax = fig.add_subplot(111)
    #
    #     dt = np.linspace(-10, 10, 1000)
    #     tau = .5
    #     y = (1 + 1j * (dt / tau)) ** (-3 / 2)

    # ax.plot(dt, np.abs(y), color = 'black', label = r"$\left| K(t-t') \right|$")
    # ax.plot(dt, np.real(y), color = 'C0', label = r"$  \mathrm{Re} \left\lbrace K(t-t') \right\rbrace  $")
    # ax.plot(dt, np.imag(y), color = 'C1', label = r"$  \mathrm{Im} \left\lbrace K(t-t') \right\rbrace   $")
    #
    # ax.set_xlabel(r"$   t-t'  $")
    # ax.set_ylabel(r"$   K(t-t') $")
    # # ax.yaxis.set_label_coords(-., .5)
    #
    # ax.set_xticks([0, tau, -tau, 2 * tau, -2 * tau])
    # ax.set_xticklabels(
    #     [r'$0$',
    #      r'$\tau_{\alpha}$',
    #      r'$-\tau_{\alpha}$',
    #      r'$2\tau_{\alpha}$',
    #      r'$-2\tau_{\alpha}$',
    #      ]
    # )
    #
    # ax.set_yticks([0, 1, -1, .5, -.5, 1 / np.sqrt(2)])
    # ax.set_yticklabels(
    #     [r'$0$',
    #      r'$1$',
    #      r'$-1$',
    #      r'$1/2$',
    #      r'$-1/2$',
    #      r'$1/\sqrt{2}$',
    #      ]
    # )
    #
    # ax.set_xlim(-3, 3)
    # ax.set_ylim(-.75, 1.4)
    #
    # ax.grid(True, **si.vis.GRID_KWARGS)
    #
    # ax.legend(loc = 'upper right', framealpha = 1)


time_field_unit = atomic_electric_field * atomic_time
kick = collections.namedtuple('kick', ['time', 'time_field_product'])


def decompose_pulse_into_kicks__amplitude(electric_potential, times):
    efield_vs_time = electric_potential.get_electric_field_amplitude(times)
    signs = np.sign(efield_vs_time)

    # state machine
    kicks = []
    current_sign = signs[0]
    efield_accumulator = 0
    start_time = times[0]
    prev_time = times[0]
    last_time = times[-1]
    max_field = 0
    max_field_time = 0
    for efield, sign, time in zip(efield_vs_time, signs, times):
        if sign == current_sign and time != last_time:
            # efield_accumulator += (efield ** 2) * (time - prev_time)
            efield_accumulator += efield * (time - prev_time)
            if max_field < np.abs(efield):
                max_field = np.abs(efield)
                max_field_time = time
        else:
            # time_diff = time - start_time
            # kick_time = (time + start_time) / 2

            kicks.append(kick(time = max_field_time, time_field_product = efield_accumulator))
            # kicks.append(kick(time = kick_time, time_field_product = current_sign * np.sqrt(efield_accumulator)))

            # reset
            current_sign = sign
            start_time = time
            efield_accumulator = 0
            max_field = 0
        prev_time = time

    return kicks


def delta_kick_decomposition_plot():
    pulse_width = 200 * asec
    timess = [np.linspace(-5 * pulse_width, 5 * pulse_width, 1000), np.linspace(-5 * pulse_width, 5 * pulse_width, 1000)]

    pulses = [ion.GaussianPulse(), ion.GaussianPulse(phase = pi / 2)]
    names = [r'Gaussian Pulse, $\varphi = 0$', r'Gaussian Pulse, $\varphi = \pi / 2$']

    for pulse, name, times in zip(pulses, names, timess):
        kicks = decompose_pulse_into_kicks__amplitude(pulse, times)
        kick_times = np.array(list(k.time for k in kicks))
        # kick_products = np.array(list(k.time_field_product for k in kicks))

        si.vis.xxyy_plot(
            f'decomposition__{name.replace("$", "").replace(" ", "").lower()}_original',
            [times, kick_times, *si.utils.grouper(np.repeat(kick_times, 2), 2)],
            [
                pulse.get_electric_field_amplitude(times),
                pulse.get_electric_field_amplitude(kick_times),
                *zip(np.zeros_like(kick_times), pulse.get_electric_field_amplitude(kick_times)),
            ],
            line_labels = ['Original', 'Decomposed'],
            x_label = r'Time $t$', x_unit = 'asec',
            y_label = fr'${ion.LATEX_EFIELD}(t)$', y_unit = 'atomic_electric_field',
            line_kwargs = [{'linewidth': BIG_LINEWIDTH},
                           {'linestyle': '', 'marker': 'o', },
                           *[{'color': 'C1', 'linestyle': '-', 'linewidth': BIG_LINEWIDTH * .75} for _ in kick_times]],
            font_size_axis_labels = 22,
            font_size_tick_labels = 14,
            font_size_legend = 16,
            font_size_title = 22,
            title = f'{name}',
            grid_kwargs = BETTER_GRID_KWARGS,
            **{**FULL_PAGE_KWARGS, **QUARTER_PAGE_KWARGS},
            **PLOT_KWARGS,
        )

        # si.vis.xy_plot(
        #     f'decomposition__{name}_decomposed',
        #     kick_times,
        #     kick_products,
        #     line_kwargs = [{'linestyle': ':', 'marker': 'o'}],
        #     x_label = r'Time $t$', x_unit = 'asec',
        #     y_label = fr'$\eta_i$ (a.u.$\,\times\,$a.u.)', y_unit = time_field_unit,
        #     font_size_axis_labels = 25,
        #     font_size_tick_labels = 15,
        #     font_size_legend = 20,
        #     font_size_title = 25,
        #     title = f'Decomposed {name} Pulse',
        #     grid_kwargs = BETTER_GRID_KWARGS,
        #     x_lower_limit = times[0], x_upper_limit = times[-1],
        #     **{**FULL_SLIDE_FIGMAN_KWARGS, **quarter_page_kwargs},
        #     **PLOT_KWARGS,
        # )


def recursive_kicks(kicks, *, abs_prefactor, kernel_func, bound_state_frequency):
    abs_prefactor = np.abs(abs_prefactor)
    bound_state_frequency = np.abs(bound_state_frequency)

    @si.utils.memoize
    def time_diff(i, j):
        return kicks[i].time - kicks[j].time

    @si.utils.memoize
    def b(i, j):
        return abs_prefactor * kicks[i].time_field_product * kicks[j].time_field_product

    @si.utils.memoize
    def a(n):
        # print(f'calling a({n})')
        if n < 0:
            return 1
        else:
            first_term = np.exp(-1j * np.abs(bound_state_frequency) * time_diff(n, n - 1)) * a(n - 1) * (1 - b(n, n))
            second_term = sum(a(i) * b(n, i) * kernel_func(time_diff(n, i)) for i in range(n))  # all but current kick
            # print(second_term)

            return first_term - second_term

    return np.array(list(a(i) for i in range(len(kicks))))


def make_sine_wave_kicks(number_of_periods, period, eta):
    kicks = []
    for n in range(number_of_periods):
        kicks.append(kick(time = (2 * n) * period / 2, time_field_product = eta))
        kicks.append(kick(time = ((2 * n) + 1) * period / 2, time_field_product = -eta))

    return kicks


def delta_kicks_eta_plot():
    etas = np.array([.2, .4, .5]) * time_field_unit

    number_of_periods = 10

    test_width = 1 * bohr_radius
    test_charge = 1 * electron_charge
    test_mass = 1 * electron_mass
    # potential_depth = 36.831335 * eV

    prefactor = ide.gaussian_prefactor_LEN(test_width, test_charge)
    tau_alpha = ide.gaussian_tau_alpha_LEN(test_width, test_mass)
    kernel = functools.partial(ide.gaussian_kernel_LEN, tau_alpha = tau_alpha)
    omega_alpha = 1 / (2 * tau_alpha)

    periods = np.linspace(0, 3, 500) * tau_alpha
    periods = periods[1:]

    def curve(periods, eta):
        results = []
        for period in periods:
            kicks = make_sine_wave_kicks(number_of_periods, period, eta)

            results.append(
                recursive_kicks(
                    kicks,
                    abs_prefactor = np.abs(prefactor),
                    kernel_func = kernel,
                    bound_state_frequency = omega_alpha
                )[-1])

        return np.abs(results) ** 2

    si.vis.xy_plot(
        f'bound_state_amplitude_vs_sine_period__eta__{etas}',
        periods,
        *[curve(periods, eta) for eta in etas],
        line_labels = [fr'$\eta = {uround(eta, time_field_unit)}$ a.u.' for eta in etas],
        line_kwargs = [{'linewidth': BIG_LINEWIDTH}, {'linewidth': BIG_LINEWIDTH}, {'linewidth': BIG_LINEWIDTH}, ],
        x_label = r'Sine Wave Period Time $t$ ($ \tau_{\alpha} $)', x_unit = tau_alpha,
        y_label = r'$\left|\left\langle a | a \right\rangle\right|^2$',
        # vlines = [tau_alpha / 2, tau_alpha], vline_kwargs = [{'linestyle': ':', 'color': 'black'}, {'linestyle': ':', 'color': 'black'}],
        y_log_axis = True,
        y_log_pad = 1,
        legend_kwargs = {'loc': 'upper right'},
        # y_lower_limit = 1e-9, y_upper_limit = 1,
        font_size_axis_labels = 35,
        font_size_tick_labels = 20,
        font_size_legend = 25,
        font_size_title = 35,
        grid_kwargs = BETTER_GRID_KWARGS,
        **PLOT_KWARGS,
        **STILL_FIGMAN_KWARGS_SVG,
    )


def pulse_ffts():
    t_bound = 500
    freq_plot_limit = 6000 * THz

    pw = 200 * asec
    flu = 1 * Jcm2
    phase = pi / 2

    times = np.linspace(-t_bound * pw, t_bound * pw, 2 ** 14)
    dt = np.abs(times[1] - times[0])

    pulses = [
        ion.SincPulse.from_omega_min(pulse_width = pw, fluence = flu, phase = phase,
                                     window = ion.SymmetricExponentialTimeWindow(window_time = 30 * pw, window_width = .2 * pw)),
        ion.GaussianPulse.from_omega_min(pulse_width = pw, fluence = flu, phase = phase,
                                         window = ion.SymmetricExponentialTimeWindow(window_time = 5 * pw, window_width = .2 * pw)),
    ]

    names = [
        'Sinc',
        'Gaussian',
    ]

    pulses = list(ion.DC_correct_electric_potential(pulse, times) for pulse in pulses)
    fields = tuple(pulse.get_electric_field_amplitude(times) for pulse in pulses)

    freqs = nfft.fftshift(nfft.fftfreq(len(times), dt))
    df = np.abs(freqs[1] - freqs[0])

    ffts = tuple(nfft.fftshift(nfft.fft(nfft.fftshift(field), norm = 'ortho') / df) for field in fields)

    si.vis.xy_plot(
        'pulse_power_spectra',
        freqs,
        *(epsilon_0 * c * (np.abs(f) ** 2) / len(times) for f in ffts),
        line_labels = names,
        line_kwargs = [{'linewidth': BIG_LINEWIDTH}, {'linewidth': BIG_LINEWIDTH}],
        x_label = r'Frequency $ f $', x_unit = 'THz',
        x_lower_limit = 0, x_upper_limit = freq_plot_limit,
        y_label = fr'$ \left| {ion.LATEX_EFIELD}(f) \right|^2 $  ($\mathrm{{J / cm^2 / THz}}$)',
        y_unit = Jcm2 / THz,
        title = fr'Power Spectral Density for $\tau = {uround(pw, asec)} \, \mathrm{{as}}, \, H = {uround(flu, Jcm2)} \, \mathrm{{J/cm^2}}$',
        vlines = [2530 * THz], vline_kwargs = [{'linewidth': BIG_LINEWIDTH, 'linestyle': '--', 'color': 'black'}],
        **BIG_FONTS,
        grid_kwargs = BETTER_GRID_KWARGS,
        **STILL_FIGMAN_KWARGS_SVG,
        **PLOT_KWARGS,
    )


def multicycle_sine_cosine_comparison(pulse_type, omega_min, postfix):
    alpha = .7

    pulse_width = 1000 * asec
    fluence = 1 * Jcm2

    times = np.linspace(-4 * pulse_width, 4 * pulse_width, 1000)

    cos_pulse = pulse_type.from_omega_min(pulse_width = pulse_width, fluence = fluence, phase = 0, omega_min = omega_min)
    sin_pulse_aligned = pulse_type.from_omega_min(pulse_width = pulse_width, fluence = fluence, phase = pi / 2, omega_min = omega_min)

    shift = (pi / 2) / cos_pulse.omega_carrier
    sin_pulse_shifted = pulse_type.from_omega_min(pulse_width = pulse_width, fluence = fluence, phase = pi / 2, pulse_center = shift, omega_min = omega_min)

    ylim = 1.05 * cos_pulse.amplitude_time

    si.vis.xy_plot(
        get_func_name() + f'__{pulse_type.__name__}__aligned_envelopes__omega_min={uround(omega_min, twopi * THz)}THz',
        times,
        cos_pulse.get_electric_field_amplitude(times),
        sin_pulse_aligned.get_electric_field_amplitude(times),
        sin_pulse_aligned.get_electric_field_envelope(times) * sin_pulse_aligned.amplitude_time,
        cos_pulse.get_electric_field_envelope(times) * cos_pulse.amplitude_time,
        line_labels = [r"$\varphi = 0$", r"$\varphi = \pi / 2$"],
        line_kwargs = [
            {'color': 'C0', 'linewidth': BIG_LINEWIDTH},
            {'color': 'C1', 'linewidth': BIG_LINEWIDTH},
            {'color': 'C0', 'linewidth': BIG_LINEWIDTH - 1, 'linestyle': '--', 'alpha': alpha},
            {'color': 'C1', 'linewidth': BIG_LINEWIDTH - 1, 'linestyle': '--', 'alpha': alpha},
        ],
        x_label = r"Time $t$", x_unit = 'asec',
        y_label = fr'Electric Field Amplitude ${ion.LATEX_EFIELD}(t)$', y_unit = 'atomic_electric_field',
        title = 'Aligned Envelopes' + postfix,
        # font_size_axis_labels = 22,
        # font_size_tick_labels = 14,
        # font_size_legend = 16,
        # font_size_title = 22,
        font_size_axis_labels = 35,
        font_size_tick_labels = 20,
        font_size_legend = 25,
        font_size_title = 35,
        grid_kwargs = BETTER_GRID_KWARGS,
        y_lower_limit = -ylim,
        y_upper_limit = ylim,
        title_offset = 1.2,
        **FULL_PAGE_KWARGS,
        # **QUARTER_PAGE_KWARGS,
        **PLOT_KWARGS,
    )

    si.vis.xy_plot(
        get_func_name() + f'__{pulse_type.__name__}__aligned_zeros__omega_min={uround(omega_min, twopi * THz)}THz',
        times,
        cos_pulse.get_electric_field_amplitude(times),
        sin_pulse_shifted.get_electric_field_amplitude(times),
        cos_pulse.get_electric_field_envelope(times) * cos_pulse.amplitude_time,
        sin_pulse_shifted.get_electric_field_envelope(times) * sin_pulse_aligned.amplitude_time,
        line_labels = [r"$\varphi = 0$", r"$\varphi = \pi / 2$"],
        line_kwargs = [
            {'color': 'C0', 'linewidth': BIG_LINEWIDTH},
            {'color': 'C1', 'linewidth': BIG_LINEWIDTH},
            {'color': 'C0', 'linewidth': BIG_LINEWIDTH - 1, 'linestyle': '--', 'alpha': alpha},
            {'color': 'C1', 'linewidth': BIG_LINEWIDTH - 1, 'linestyle': '--', 'alpha': alpha},
        ],
        x_label = r"Time $t$", x_unit = 'asec',
        y_label = fr'Electric Field Amplitude ${ion.LATEX_EFIELD}(t)$', y_unit = 'atomic_electric_field',
        title = 'Aligned Zeros' + postfix,
        # font_size_axis_labels = 22,
        # font_size_tick_labels = 14,
        # font_size_legend = 16,
        # font_size_title = 22,
        **BIG_FONTS,
        grid_kwargs = BETTER_GRID_KWARGS,
        y_lower_limit = -ylim,
        y_upper_limit = ylim,
        **FULL_PAGE_KWARGS,
        # **QUARTER_PAGE_KWARGS,
        **PLOT_KWARGS,
    )


def pulse_width_scan__hyd__sinc():
    jp = clu.JobProcessor.load('job_processors/hyd__pw_scan_v2__50-1000as_3flus_3phis__sinc.job')

    phases = sorted(jp.parameter_set('phase'))
    fluences = sorted(jp.parameter_set('fluence'))

    styles = ['-', ':', '--']
    colors = ['C0', 'C1', 'C2']

    phase_to_style = dict(zip(phases, styles))
    fluence_to_color = dict(zip(fluences, colors))
    color_patches = [mpatches.Patch(color = color, label = fr'$ H = {uround(fluence, Jcm2)} \, \mathrm{{J/cm^2}} $')
                     for fluence, color in fluence_to_color.items()]

    phases_latex = [r'0', r'\pi / 4', r'\pi / 2']
    style_patches = [mlines.Line2D([], [], color = 'black', linestyle = style, linewidth = 3, label = fr'$ \varphi = {phase_latex} $')
                     for phase, style, phase_latex in zip(phases, styles, phases_latex)]

    legend_handles = color_patches + style_patches

    results_by_phase_and_fluence = {(phase, fluence): jp.select_by_kwargs(phase = phase, fluence = fluence)
                                    for phase in phases for fluence in fluences}

    metrics = ['final_initial_state_overlap']
    extra_line_kwargs = dict(
        linewidth = 3,
    )

    for log_x, log_y in itertools.product([True, False], repeat = 2):
        postfix = ''
        if any([log_x, log_y]):
            postfix += '__log'
        if log_x:
            postfix += 'X'
        if log_y:
            postfix += 'Y'

        for metric in metrics:
            si.vis.xxyy_plot(
                f'pulse_width_scan__sinc__hyd__{metric}' + postfix,
                [
                    *[[r.pulse_width for r in results] for results in results_by_phase_and_fluence.values()]
                ],
                [
                    *[[getattr(r, metric) for r in results] for results in results_by_phase_and_fluence.values()]
                ],
                line_kwargs = [{'linestyle': phase_to_style[phase], 'color': fluence_to_color[fluence], **extra_line_kwargs} for phase, fluence in results_by_phase_and_fluence.keys()],
                title = 'Pulse Width Scan: Sinc Pulse', title_offset = 1.075,
                x_label = r'Pulse Width $\tau$',
                x_unit = 'asec',
                y_label = metric.replace('_', ' ').title(),
                y_log_axis = log_y, y_log_pad = 2,
                x_log_axis = log_x,
                legend_kwargs = {
                    'loc': 'upper right',
                    'bbox_to_anchor': (.99, .875),
                    'handles': legend_handles,
                },
                grid_kwargs = BETTER_GRID_KWARGS,
                font_size_axis_labels = 35,
                font_size_tick_labels = 20,
                font_size_legend = 20,
                font_size_title = 35,
                **FULL_PAGE_KWARGS,
                **PLOT_KWARGS,
            )

            # si.vis.xxyy_plot(
            #     f'pulse_width_scan__sinc__hyd__zoom__{metric}' + postfix,
            #     [
            #         *[[r.pulse_width for r in results] for results in results_by_phase_and_fluence.values()]
            #     ],
            #     [
            #         *[[getattr(r, metric) for r in results] for results in results_by_phase_and_fluence.values()]
            #     ],
            #     line_kwargs = [{'linestyle': phase_to_style[phase], 'color': fluence_to_color[fluence], **extra_line_kwargs} for phase, fluence in results_by_phase_and_fluence.keys()],
            #     title = 'Pulse Width Scan: Sinc Pulse', title_offset = 1.075,
            #     x_label = r'Pulse Width $\tau$',
            #     x_unit = 'asec',
            #     y_label = metric.replace('_', ' ').title(),
            #     y_log_axis = log_y, y_log_pad = 2,
            #     x_log_axis = log_x,
            #     legend_kwargs = {
            #         'loc': 'upper right',
            #         'bbox_to_anchor': (.99, .875),
            #         'handles': legend_handles,
            #     },
            #     x_lower_limit = 80 * asec,
            #     x_upper_limit = 100 * asec,
            #     grid_kwargs = BETTER_GRID_KWARGS,
            #     font_size_axis_labels = 35,
            #     font_size_tick_labels = 20,
            #     font_size_legend = 20,
            #     font_size_title = 35,
            #     **FULL_PAGE_KWARGS,
            #     **PLOT_KWARGS,
            # )


def pulse_width_scan__ide__sinc():
    jp = clu.JobProcessor.load('job_processors/ide__pw_scan_50-1000as_3flus_3phis__sinc__fast_fixed.job')
    # jp = clu.JobProcessor.load('job_processors/ide__pw_scan__different_energy__fast.job')

    phases = sorted(jp.parameter_set('phase'))
    # fluences = sorted(jp.parameter_set('fluence'))
    fluences = [.1 * Jcm2, 1 * Jcm2, 10 * Jcm2]

    styles = ['-', ':', '--']
    colors = ['C0', 'C1', 'C2']

    phase_to_style = dict(zip(phases, styles))
    fluence_to_color = dict(zip(fluences, colors))
    color_patches = [mpatches.Patch(color = color, label = fr'$ H = {uround(fluence, Jcm2)} \, \mathrm{{J/cm^2}} $')
                     for fluence, color in fluence_to_color.items()]

    phases_latex = [r'0', r'\pi / 4', r'\pi / 2']
    style_patches = [mlines.Line2D([], [], color = 'black', linestyle = style, linewidth = 3, label = fr'$ \varphi = {phase_latex} $')
                     for phase, style, phase_latex in zip(phases, styles, phases_latex)]

    legend_handles = color_patches + style_patches

    results_by_phase_and_fluence = {(phase, fluence): jp.select_by_kwargs(phase = phase, fluence = fluence)
                                    for phase in phases for fluence in fluences}

    metrics = ['final_initial_state_overlap', 'final_bound_state_overlap']
    extra_line_kwargs = dict(
        linewidth = 3,
    )

    for log_x, log_y in itertools.product([True, False], repeat = 2):
        postfix = ''
        if any([log_x, log_y]):
            postfix += '__log'
        if log_x:
            postfix += 'X'
        if log_y:
            postfix += 'Y'

        for metric in metrics:
            si.vis.xxyy_plot(
                f'pulse_width_scan__sinc__ide__{metric}' + postfix,
                [
                    *[[r.pulse_width for r in results] for results in results_by_phase_and_fluence.values()]
                ],
                [
                    *[[getattr(r, metric) for r in results] for results in results_by_phase_and_fluence.values()]
                ],
                line_kwargs = [{'linestyle': phase_to_style[phase], 'color': fluence_to_color[fluence], **extra_line_kwargs} for phase, fluence in results_by_phase_and_fluence.keys()],
                title = 'Pulse Width Scan: Sinc Pulse', title_offset = 1.075,
                x_label = r'Pulse Width $\tau$',
                x_unit = 'asec',
                y_label = metric.replace('_', ' ').title(),
                y_log_axis = log_y, y_log_pad = 2,
                x_log_axis = log_x,
                legend_kwargs = {
                    'loc': 'upper right',
                    'bbox_to_anchor': (.99, .875),
                    'handles': legend_handles,
                },
                grid_kwargs = BETTER_GRID_KWARGS,
                font_size_axis_labels = 35,
                font_size_tick_labels = 20,
                font_size_legend = 20,
                font_size_title = 35,
                x_upper_limit = 350 * asec,
                **FULL_PAGE_KWARGS,
                **PLOT_KWARGS,
            )

            # si.vis.xxyy_plot(
            #     f'pulse_width_scan__sinc__hyd__zoom__{metric}' + postfix,
            #     [
            #         *[[r.pulse_width for r in results] for results in results_by_phase_and_fluence.values()]
            #     ],
            #     [
            #         *[[getattr(r, metric) for r in results] for results in results_by_phase_and_fluence.values()]
            #     ],
            #     line_kwargs = [{'linestyle': phase_to_style[phase], 'color': fluence_to_color[fluence], **extra_line_kwargs} for phase, fluence in results_by_phase_and_fluence.keys()],
            #     title = 'Pulse Width Scan: Sinc Pulse', title_offset = 1.075,
            #     x_label = r'Pulse Width $\tau$',
            #     x_unit = 'asec',
            #     y_label = metric.replace('_', ' ').title(),
            #     y_log_axis = log_y, y_log_pad = 2,
            #     x_log_axis = log_x,
            #     legend_kwargs = {
            #         'loc': 'upper right',
            #         'bbox_to_anchor': (.99, .875),
            #         'handles': legend_handles,
            #     },
            #     x_lower_limit = 80 * asec,
            #     x_upper_limit = 100 * asec,
            #     grid_kwargs = BETTER_GRID_KWARGS,
            #     font_size_axis_labels = 35,
            #     font_size_tick_labels = 20,
            #     font_size_legend = 20,
            #     font_size_title = 35,
            #     **FULL_PAGE_KWARGS,
            #     **PLOT_KWARGS,
            # )


def fluence_scan__hyd__sinc():
    jp = clu.JobProcessor.load('job_processors/hyd__flu_scan_v2__5pws_.01-30jcm2_3phis__sinc.job')

    phases = sorted(jp.parameter_set('phase'))
    pulse_widths = sorted(jp.parameter_set('pulse_width'))[::2]

    styles = ['-', ':', '--']
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']

    phase_to_style = dict(zip(phases, styles))
    pulse_width_to_color = dict(zip(pulse_widths, colors))
    color_patches = [mpatches.Patch(color = color, label = fr'$ \tau = {uround(pulse_width, asec)} \, \mathrm{{as}} $')
                     for pulse_width, color in pulse_width_to_color.items()]

    phases_latex = [r'0', r'\pi / 4', r'\pi / 2']
    style_patches = [mlines.Line2D([], [], color = 'black', linestyle = style, linewidth = 3, label = fr'$ \varphi = {phase_latex} $')
                     for phase, style, phase_latex in zip(phases, styles, phases_latex)]

    legend_handles = color_patches + style_patches

    results_by_phase_and_pulse_width = {(phase, pulse_width): jp.select_by_kwargs(phase = phase, pulse_width = pulse_width)
                                        for phase in phases for pulse_width in pulse_widths}

    metrics = ['final_initial_state_overlap', 'final_bound_state_overlap']
    extra_line_kwargs = dict(
        linewidth = 3,
    )

    for log_x, log_y in itertools.product([True, False], repeat = 2):
        postfix = ''
        if any([log_x, log_y]):
            postfix += '__log'
        if log_x:
            postfix += 'X'
        if log_y:
            postfix += 'Y'

        for metric in metrics:
            si.vis.xxyy_plot(
                f'fluence_scan__sinc__hyd__{metric}' + postfix,
                [
                    *[[r.fluence for r in results] for results in results_by_phase_and_pulse_width.values()]
                ],
                [
                    *[[getattr(r, metric) for r in results] for results in results_by_phase_and_pulse_width.values()]
                ],
                line_kwargs = [{'linestyle': phase_to_style[phase], 'color': pulse_width_to_color[pulse_width], **extra_line_kwargs} for phase, pulse_width in results_by_phase_and_pulse_width.keys()],
                title = 'Fluence Scan: Sinc Pulse', title_offset = 1.075,
                x_label = r'Fluence $H$',
                x_unit = 'Jcm2',
                y_label = metric.replace('_', ' ').title(),
                y_log_axis = log_y, y_log_pad = 2,
                x_log_axis = log_x,
                legend_kwargs = {
                    'loc': 'best',
                    # 'bbox_to_anchor': (.99, .875),
                    'handles': legend_handles,
                },
                grid_kwargs = BETTER_GRID_KWARGS,
                font_size_axis_labels = 35,
                font_size_tick_labels = 20,
                font_size_legend = 20,
                font_size_title = 35,
                x_upper_limit = 15 * Jcm2,
                **FULL_PAGE_KWARGS,
                **PLOT_KWARGS,
            )


def pulse_width_scan__hyd__gaussian():
    jp = clu.JobProcessor.load('job_processors/hyd__pw_scan_v2__50-1000as_3flus_3phis__gaussian__fixed_bounds.job')

    phases = sorted(jp.parameter_set('phase'))
    fluences = sorted(jp.parameter_set('fluence'))

    styles = ['-', ':', '--']
    colors = ['C0', 'C1', 'C2']

    phase_to_style = dict(zip(phases, styles))
    fluence_to_color = dict(zip(fluences, colors))
    color_patches = [mpatches.Patch(color = color, label = fr'$ H = {uround(fluence, Jcm2)} \, \mathrm{{J/cm^2}} $')
                     for fluence, color in fluence_to_color.items()]

    phases_latex = [r'0', r'\pi / 4', r'\pi / 2']
    style_patches = [mlines.Line2D([], [], color = 'black', linestyle = style, linewidth = 3, label = fr'$ \varphi = {phase_latex} $')
                     for phase, style, phase_latex in zip(phases, styles, phases_latex)]

    legend_handles = color_patches + style_patches

    results_by_phase_and_fluence = {(phase, fluence): jp.select_by_kwargs(phase = phase, fluence = fluence)
                                    for phase in phases for fluence in fluences}

    metrics = ['final_initial_state_overlap', 'final_bound_state_overlap']
    extra_line_kwargs = dict(
        linewidth = 3,
    )

    for log_x, log_y in itertools.product([True, False], repeat = 2):
        postfix = ''
        if any([log_x, log_y]):
            postfix += '__log'
        if log_x:
            postfix += 'X'
        if log_y:
            postfix += 'Y'

        for metric in metrics:
            si.vis.xxyy_plot(
                f'pulse_width_scan__gaussian__hyd__{metric}' + postfix,
                [
                    *[[r.pulse_width for r in results] for results in results_by_phase_and_fluence.values()]
                ],
                [
                    *[[getattr(r, metric) for r in results] for results in results_by_phase_and_fluence.values()]
                ],
                line_kwargs = [{'linestyle': phase_to_style[phase], 'color': fluence_to_color[fluence], **extra_line_kwargs} for phase, fluence in results_by_phase_and_fluence.keys()],
                title = 'Pulse Width Scan: Gaussian Pulse', title_offset = 1.075,
                x_label = r'Pulse Width $\tau$',
                x_unit = 'asec',
                y_label = metric.replace('_', ' ').title(),
                y_log_axis = log_y, y_log_pad = 2,
                x_log_axis = log_x,
                legend_kwargs = {
                    'loc': 'upper right',
                    'bbox_to_anchor': (.99, .875),
                    'handles': legend_handles,
                },
                grid_kwargs = BETTER_GRID_KWARGS,
                font_size_axis_labels = 35,
                font_size_tick_labels = 20,
                font_size_legend = 20,
                font_size_title = 35,
                **FULL_PAGE_KWARGS,
                **PLOT_KWARGS,
            )

            # si.vis.xxyy_plot(
            #     f'pulse_width_scan__gaussian__hyd__zoom__{metric}' + postfix,
            #     [
            #         *[[r.pulse_width for r in results] for results in results_by_phase_and_fluence.values()]
            #     ],
            #     [
            #         *[[getattr(r, metric) for r in results] for results in results_by_phase_and_fluence.values()]
            #     ],
            #     line_kwargs = [{'linestyle': phase_to_style[phase], 'color': fluence_to_color[fluence], **extra_line_kwargs} for phase, fluence in results_by_phase_and_fluence.keys()],
            #     title = 'Pulse Width Scan: Gaussian Pulse', title_offset = 1.075,
            #     x_label = r'Pulse Width $\tau$',
            #     x_unit = 'asec',
            #     y_label = metric.replace('_', ' ').title(),
            #     y_log_axis = log_y, y_log_pad = 2,
            #     x_log_axis = log_x,
            #     legend_kwargs = {
            #         'loc': 'upper right',
            #         'bbox_to_anchor': (.99, .875),
            #         'handles': legend_handles,
            #     },
            #     x_lower_limit = 80 * asec,
            #     x_upper_limit = 100 * asec,
            #     grid_kwargs = BETTER_GRID_KWARGS,
            #     font_size_axis_labels = 35,
            #     font_size_tick_labels = 20,
            #     font_size_legend = 20,
            #     font_size_title = 35,
            #     **FULL_PAGE_KWARGS,
            #     **PLOT_KWARGS,
            #     )


def fluence_scan__hyd__gaussian():
    jp = clu.JobProcessor.load('job_processors/hyd__flu_scan_v2__5pws_.01-30jcm2_3phis__gaussian__fixed_bounds.job')

    phases = sorted(jp.parameter_set('phase'))
    pulse_widths = sorted(jp.parameter_set('pulse_width'))[::2]

    styles = ['-', ':', '--']
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']

    phase_to_style = dict(zip(phases, styles))
    pulse_width_to_color = dict(zip(pulse_widths, colors))
    color_patches = [mpatches.Patch(color = color, label = fr'$ \tau = {uround(pulse_width, asec)} \, \mathrm{{as}} $')
                     for pulse_width, color in pulse_width_to_color.items()]

    phases_latex = [r'0', r'\pi / 4', r'\pi / 2']
    style_patches = [mlines.Line2D([], [], color = 'black', linestyle = style, linewidth = 3, label = fr'$ \varphi = {phase_latex} $')
                     for phase, style, phase_latex in zip(phases, styles, phases_latex)]

    legend_handles = color_patches + style_patches

    results_by_phase_and_pulse_width = {(phase, pulse_width): jp.select_by_kwargs(phase = phase, pulse_width = pulse_width)
                                        for phase in phases for pulse_width in pulse_widths}

    metrics = ['final_initial_state_overlap', 'final_bound_state_overlap']
    extra_line_kwargs = dict(
        linewidth = BIG_LINEWIDTH,
    )

    for log_x, log_y in itertools.product([True, False], repeat = 2):
        postfix = ''
        if any([log_x, log_y]):
            postfix += '__log'
        if log_x:
            postfix += 'X'
        if log_y:
            postfix += 'Y'

        for metric in metrics:
            si.vis.xxyy_plot(
                f'fluence_scan__gaussian__hyd__{metric}' + postfix,
                [
                    *[[r.fluence for r in results] for results in results_by_phase_and_pulse_width.values()]
                ],
                [
                    *[[getattr(r, metric) for r in results] for results in results_by_phase_and_pulse_width.values()]
                ],
                line_kwargs = [{'linestyle': phase_to_style[phase], 'color': pulse_width_to_color[pulse_width], **extra_line_kwargs} for phase, pulse_width in results_by_phase_and_pulse_width.keys()],
                title = 'Fluence Scan: Gaussian Pulse', title_offset = 1.075,
                x_label = r'Fluence $H$',
                x_unit = 'Jcm2',
                y_label = metric.replace('_', ' ').title(),
                y_log_axis = log_y, y_log_pad = 2,
                x_log_axis = log_x,
                legend_kwargs = {
                    'loc': 'best',
                    # 'bbox_to_anchor': (.99, .875),
                    'handles': legend_handles,
                },
                grid_kwargs = BETTER_GRID_KWARGS,
                font_size_axis_labels = 35,
                font_size_tick_labels = 20,
                font_size_legend = 20,
                font_size_title = 35,
                x_upper_limit = 15 * Jcm2,
                **FULL_PAGE_KWARGS,
                **PLOT_KWARGS,
            )


# def field_properties_vs_phase():
#     t_bound = 35
#     p_bound = 30
#
#     pulse_types = (
#         ion.SincPulse,
#         # ion.GaussianPulse,
#         # ion.SechPulse,
#     )
#
#     labels = [
#         'Sinc',
#         'Gaussian',
#     ]
#     line_kwargs = [
#         {'linewidth': BIG_LINEWIDTH},
#         {'linewidth': BIG_LINEWIDTH},
#     ]
#
#     common = dict(
#         x_label = r'Carrier-Envelope Phase $ \varphi $',
#         x_unit = 'rad',
#     )
#
#     pw = 200 * asec
#     flu = 1 * Jcm2
#     phases = np.linspace(0, pi, 1e3)
#
#     times = np.linspace(-t_bound * pw, t_bound * pw, 2 ** 14)
#     dt = np.abs(times[1] - times[0])
#
#     window = ion.SymmetricExponentialTimeWindow(window_time = p_bound * pw, window_width = .2 * pw)
#
#     max_electric_field = {pulse_type: np.empty_like(phases) for pulse_type in pulse_types}
#     avg_electric_field = {pulse_type: np.empty_like(phases) for pulse_type in pulse_types}
#     rms_electric_field = {pulse_type: np.empty_like(phases) for pulse_type in pulse_types}
#
#     max_vector_potential = {pulse_type: np.empty_like(phases) for pulse_type in pulse_types}
#     avg_vector_potential = {pulse_type: np.empty_like(phases) for pulse_type in pulse_types}
#     rms_vector_potential = {pulse_type: np.empty_like(phases) for pulse_type in pulse_types}
#
#     max_intensity = {pulse_type: np.empty_like(phases) for pulse_type in pulse_types}
#     avg_intensity = {pulse_type: np.empty_like(phases) for pulse_type in pulse_types}
#     rms_intensity = {pulse_type: np.empty_like(phases) for pulse_type in pulse_types}
#
#     for pulse_type in pulse_types:
#         for ii, phase in enumerate(tqdm(phases)):
#             pulse = pulse_type.from_omega_min(pulse_width = pw, fluence = flu, phase = phase,
#                                               window = window)
#             pulse = ion.DC_correct_electric_potential(pulse, times)
#
#             ef = pulse.get_electric_field_amplitude(times)
#             max_electric_field[pulse_type][ii] = np.max(np.abs(ef))
#             avg_electric_field[pulse_type][ii] = np.mean(np.abs(ef))
#             rms_electric_field[pulse_type][ii] = np.sqrt(np.mean(ef ** 2))
#
#             vp = pulse.get_vector_potential_amplitude_numeric_cumulative(times)
#             max_vector_potential[pulse_type][ii] = np.max(np.abs(vp))
#             avg_vector_potential[pulse_type][ii] = np.mean(np.abs(vp))
#             rms_vector_potential[pulse_type][ii] = np.sqrt(np.mean(vp ** 2))
#
#             intensity = epsilon_0 * c * (np.abs(ef) ** 2)
#             max_intensity[pulse_type][ii] = np.max(np.abs(intensity))
#             avg_intensity[pulse_type][ii] = np.mean(np.abs(intensity))
#             rms_intensity[pulse_type][ii] = np.sqrt(np.mean(intensity ** 2))
#
#     # MAX ABS
#
#     si.vis.xy_plot(
#         'max_electric_field_vs_phase',
#         phases,
#         *(max_ef for pulse, max_ef in max_electric_field.items()),
#         line_labels = labels,
#         line_kwargs = line_kwargs,
#         x_label = r'Carrier-Envelope Phase $ \varphi $', x_unit = 'rad',
#         y_label = rf'$ \max \; \left|{ion.LATEX_EFIELD}_{{\varphi}}(t)\right| $', y_unit = 'atomic_electric_field',
#         grid_kwargs = BETTER_GRID_KWARGS,
#         **BIG_FONTS,
#         **FULL_PAGE_KWARGS,
#         **PLOT_KWARGS,
#     )
#
#     si.vis.xy_plot(
#         'max_electric_field_vs_phase__rel',
#         phases,
#         *(max_ef / max_electric_field[pulse][0] for pulse, max_ef in max_electric_field.items()),
#         line_labels = labels,
#         line_kwargs = line_kwargs,
#         y_label = rf'$ \max \; \left|{ion.LATEX_EFIELD}_{{\varphi}}(t)\right| $ (Normalized)',
#         grid_kwargs = BETTER_GRID_KWARGS,
#         **common,
#         **BIG_FONTS,
#         **FULL_PAGE_KWARGS,
#         **PLOT_KWARGS,
#     )
#
#     si.vis.xy_plot(
#         'max_vector_potential_vs_phase',
#         phases,
#         *(proton_charge * max_vp for pulse, max_vp in max_vector_potential.items()),
#         line_labels = labels,
#         line_kwargs = line_kwargs,
#         x_label = r'Carrier-Envelope Phase $ \varphi $', x_unit = 'rad',
#         y_label = rf'$ \max \; e \, \left|{ion.LATEX_AFIELD}_{{\varphi}}(t)\right| $', y_unit = 'atomic_momentum',
#         grid_kwargs = BETTER_GRID_KWARGS,
#         **BIG_FONTS,
#         **FULL_PAGE_KWARGS,
#         **PLOT_KWARGS,
#     )
#
#     si.vis.xy_plot(
#         'max_vector_potential_vs_phase__rel',
#         phases,
#         *(max_vp / max_vector_potential[pulse][0] for pulse, max_vp in max_vector_potential.items()),
#         line_labels = labels,
#         line_kwargs = line_kwargs,
#         x_label = r'Carrier-Envelope Phase $ \varphi $', x_unit = 'rad',
#         y_label = rf'$ \max \; e \, \left|{ion.LATEX_AFIELD}_{{\varphi}}(t)\right| $ (Normalized)',
#         grid_kwargs = BETTER_GRID_KWARGS,
#         **BIG_FONTS,
#         **FULL_PAGE_KWARGS,
#         **PLOT_KWARGS,
#     )
#
#     si.vis.xy_plot(
#         'max_intensity_vs_phase',
#         phases,
#         *(max_int for pulse, max_int in max_intensity.items()),
#         line_labels = labels,
#         line_kwargs = line_kwargs,
#         x_label = r'Carrier-Envelope Phase $ \varphi $', x_unit = 'rad',
#         y_label = rf'$ \max \; P(t) $', y_unit = 'atomic_intensity',
#         grid_kwargs = BETTER_GRID_KWARGS,
#         **BIG_FONTS,
#         **FULL_PAGE_KWARGS,
#         **PLOT_KWARGS,
#     )
#
#     si.vis.xy_plot(
#         'max_intensity_vs_phase__rel',
#         phases,
#         *(max_int / max_intensity[pulse][0] for pulse, max_int in max_intensity.items()),
#         line_labels = labels,
#         line_kwargs = line_kwargs,
#         x_label = r'Carrier-Envelope Phase $ \varphi $', x_unit = 'rad',
#         y_label = rf'$ \max \; P(t) $ (Normalized)',
#         grid_kwargs = BETTER_GRID_KWARGS,
#         **BIG_FONTS,
#         **FULL_PAGE_KWARGS,
#         **PLOT_KWARGS,
#     )
#
#     # # AVG ABS
#     #
#     # si.vis.xy_plot(
#     #     'avg_abs_vector_potential_vs_phase',
#     #     phases,
#     #     *(proton_charge * avg_abs_vp for pulse, avg_abs_vp in avg_abs_vector_potential.items()),
#     #     line_labels = labels,
#     #     line_kwargs = line_kwargs,
#     #     x_label = r'$ \varphi $', x_unit = 'rad',
#     #     y_label = rf'$ \mathrm{{avg}}_t \; \left|{ion.LATEX_AFIELD}_{{\varphi}}(t)\right| $', y_unit = 'atomic_momentum',
#     #     grid_kwargs = BETTER_GRID_KWARGS,
#     #     font_size_axis_labels = 35,
#     #     font_size_tick_labels = 20,
#     #     font_size_legend = 20,
#     #     font_size_title = 35,
#     #     **FULL_PAGE_KWARGS,
#     #     **PLOT_KWARGS,
#     # )
#     #
#     # si.vis.xy_plot(
#     #     'avg_abs_vector_potential_vs_phase__rel',
#     #     phases,
#     #     *(avg_abs_vp / avg_abs_vector_potential[pulse][0] for pulse, avg_abs_vp in avg_abs_vector_potential.items()),
#     #     line_labels = labels,
#     #     line_kwargs = line_kwargs,
#     #     x_label = r'$ \varphi $', x_unit = 'rad',
#     #     y_label = rf'$ \mathrm{{avg}}_t \; \left|{ion.LATEX_AFIELD}_{{\varphi}}(t)\right| / \mathrm{{avg}}_t \left|{ion.LATEX_AFIELD}_{{\varphi = 0}}(t)\right| $',
#     #     grid_kwargs = BETTER_GRID_KWARGS,
#     #     font_size_axis_labels = 35,
#     #     font_size_tick_labels = 20,
#     #     font_size_legend = 20,
#     #     font_size_title = 35,
#     #     **FULL_PAGE_KWARGS,
#     #     **PLOT_KWARGS,
#     # )
#
#     # RMS
#
#     si.vis.xy_plot(
#         'rms_electric_field_vs_phase',
#         phases,
#         *(rms_ef for pulse, rms_ef in rms_electric_field.items()),
#         line_labels = labels,
#         line_kwargs = line_kwargs,
#         x_label = r'$ \varphi $', x_unit = 'rad',
#         y_label = rf'$ \mathrm{{RMS}} \; {ion.LATEX_EFIELD}_{{\varphi}}(t) $', y_unit = 'atomic_electric_field',
#         grid_kwargs = BETTER_GRID_KWARGS,
#         **BIG_FONTS,
#         **FULL_PAGE_KWARGS,
#         **PLOT_KWARGS,
#     )
#
#     si.vis.xy_plot(
#         'rms_electric_field_vs_phase__rel',
#         phases,
#         *(rms_ef / rms_electric_field[pulse][0] for pulse, rms_ef in rms_electric_field.items()),
#         line_labels = labels,
#         line_kwargs = line_kwargs,
#         x_label = r'$ \varphi $', x_unit = 'rad',
#         y_label = rf'$ \mathrm{{RMS}} \; {ion.LATEX_EFIELD}_{{\varphi}}(t) $ (Normalized)',
#         grid_kwargs = BETTER_GRID_KWARGS,
#         **BIG_FONTS,
#         **FULL_PAGE_KWARGS,
#         **PLOT_KWARGS,
#     )
#
#     si.vis.xy_plot(
#         'rms_vector_potential_vs_phase',
#         phases,
#         *(proton_charge * rms_vp for pulse, rms_vp in rms_vector_potential.items()),
#         line_labels = labels,
#         line_kwargs = line_kwargs,
#         x_label = r'Carrier-Envelope Phase $ \varphi $', x_unit = 'rad',
#         y_label = rf'$ \mathrm{{RMS}} \; e \, {ion.LATEX_AFIELD}_{{\varphi}}(t) $', y_unit = 'atomic_momentum',
#         grid_kwargs = BETTER_GRID_KWARGS,
#         **BIG_FONTS,
#         **FULL_PAGE_KWARGS,
#         **PLOT_KWARGS,
#     )
#
#     si.vis.xy_plot(
#         'rms_vector_potential_vs_phase__rel',
#         phases,
#         *(rms_vp / rms_vector_potential[pulse][0] for pulse, rms_vp in rms_vector_potential.items()),
#         line_labels = labels,
#         line_kwargs = line_kwargs,
#         y_label = rf'$ \mathrm{{RMS}} \; e \, {ion.LATEX_AFIELD}_{{\varphi}}(t) $ (Normalized)',
#         **common,
#         grid_kwargs = BETTER_GRID_KWARGS,
#         **BIG_FONTS,
#         **FULL_PAGE_KWARGS,
#         **PLOT_KWARGS,
#     )
#
#     si.vis.xy_plot(
#         'rms_intensity_vs_phase',
#         phases,
#         *(rms_int for pulse, rms_int in rms_intensity.items()),
#         line_labels = labels,
#         line_kwargs = line_kwargs,
#         y_label = rf'$ \mathrm{{RMS}} \; P(t) $', y_unit = 'atomic_intensity',
#         grid_kwargs = BETTER_GRID_KWARGS,
#         **common,
#         **BIG_FONTS,
#         **FULL_PAGE_KWARGS,
#         **PLOT_KWARGS,
#     )
#
#     si.vis.xy_plot(
#         'rms_intensity_vs_phase__rel',
#         phases,
#         *(rms_int / rms_intensity[pulse][0] for pulse, rms_int in rms_intensity.items()),
#         line_labels = labels,
#         line_kwargs = line_kwargs,
#         y_label = rf'$ \mathrm{{RMS}} \; P(t) $ (Normalized)',
#         grid_kwargs = BETTER_GRID_KWARGS,
#         **common,
#         **BIG_FONTS,
#         **FULL_PAGE_KWARGS,
#         **PLOT_KWARGS,
#     )


def field_properties_vs_phase_v2():
    t_bound = 35
    p_bound = 30

    common = dict(
        x_label = r'Carrier-Envelope Phase $ \varphi $',
        x_unit = 'rad',
        line_kwargs = [{'linewidth': BIG_LINEWIDTH}],
    )

    pw = 200 * asec
    flu = 1 * Jcm2
    phases = np.linspace(0, pi, 1e3)

    times = np.linspace(-t_bound * pw, t_bound * pw, 10000)

    window = ion.SymmetricExponentialTimeWindow(window_time = p_bound * pw, window_width = .2 * pw)

    max_electric_field = np.empty_like(phases)
    max_vector_potential = np.empty_like(phases)
    max_intensity = np.empty_like(phases)

    for ii, phase in enumerate(tqdm(phases)):
        pulse = ion.SincPulse.from_omega_min(pulse_width = pw, fluence = flu, phase = phase,
                                             window = window)
        pulse = ion.DC_correct_electric_potential(pulse, times)

        ef = pulse.get_electric_field_amplitude(times)
        max_electric_field[ii] = np.max(np.abs(ef))

        vp = pulse.get_vector_potential_amplitude_numeric_cumulative(times)
        max_vector_potential[ii] = np.max(np.abs(vp))

        intensity = epsilon_0 * c * (np.abs(ef) ** 2)
        max_intensity[ii] = np.max(np.abs(intensity))

    # MAX ABS

    si.vis.xy_plot(
        'max_electric_field_vs_phase',
        phases,
        max_electric_field,
        y_label = rf'$ \max \; \left|{ion.LATEX_EFIELD}_{{\varphi}}(t)\right| $', y_unit = 'atomic_electric_field',
        title = 'Max. Electric Field vs. CEP',
        grid_kwargs = BETTER_GRID_KWARGS,
        **common,
        **BIG_FONTS,
        **FULL_PAGE_KWARGS,
        **PLOT_KWARGS,
    )

    si.vis.xy_plot(
        'max_electric_field_vs_phase__rel',
        phases,
        max_electric_field / max_electric_field[0],
        y_label = rf'$ \max \; \left|{ion.LATEX_EFIELD}_{{\varphi}}(t)\right| $ (Normalized)',
        title = 'Max. Electric Field vs. CEP',
        grid_kwargs = BETTER_GRID_KWARGS,
        **common,
        **BIG_FONTS,
        **FULL_PAGE_KWARGS,
        **PLOT_KWARGS,
    )

    si.vis.xy_plot(
        'max_vector_potential_vs_phase',
        phases,
        proton_charge * max_vector_potential,
        y_label = rf'$ \max \; e \, \left|{ion.LATEX_AFIELD}_{{\varphi}}(t)\right| $', y_unit = 'atomic_momentum',
        title = 'Max. Vector Potential vs. CEP',
        grid_kwargs = BETTER_GRID_KWARGS,
        **common,
        **BIG_FONTS,
        **FULL_PAGE_KWARGS,
        **PLOT_KWARGS,
    )

    si.vis.xy_plot(
        'max_vector_potential_vs_phase__rel',
        phases,
        max_vector_potential / max_vector_potential[0],
        y_label = rf'$ \max \; e \, \left|{ion.LATEX_AFIELD}_{{\varphi}}(t)\right| $ (Normalized)',
        title = 'Max. Vector Potential vs. CEP',
        grid_kwargs = BETTER_GRID_KWARGS,
        **common,
        **BIG_FONTS,
        **FULL_PAGE_KWARGS,
        **PLOT_KWARGS,
    )

    si.vis.xy_plot(
        'max_intensity_vs_phase',
        phases,
        max_intensity,
        y_label = rf'$ \max \; P(t) $', y_unit = 'atomic_intensity',
        title = 'Max. Intensity vs. CEP',
        grid_kwargs = BETTER_GRID_KWARGS,
        **common,
        **BIG_FONTS,
        **FULL_PAGE_KWARGS,
        **PLOT_KWARGS,
    )

    si.vis.xy_plot(
        'max_intensity_vs_phase__rel',
        phases,
        max_intensity / max_intensity[0],
        y_label = rf'$ \max \; P(t) $ (Normalized)',
        title = 'Max. Intensity vs. CEP',
        grid_kwargs = BETTER_GRID_KWARGS,
        **common,
        **BIG_FONTS,
        **FULL_PAGE_KWARGS,
        **PLOT_KWARGS,
    )


def instantaneous_tunneling_rate(electric_field_amplitude, ionization_potential = -rydberg):
    # f = np.abs(electric_field_amplitude / atomic_electric_field)
    #
    # return (4 / f) * (electron_mass_reduced * (proton_charge ** 4) / (hbar ** 3)) * np.exp(-(2 / 3) / f)

    amplitude_scaled = np.abs(electric_field_amplitude / atomic_electric_field)
    potential_scaled = np.abs(ionization_potential / hartree)

    f = amplitude_scaled / ((2 * potential_scaled) ** 1.5)

    return (4 / f) * np.exp(-2 / (3 * f)) / atomic_time


def instantaneous_tunneling_rate_plot():
    amplitudes = np.linspace(0, .1, 1e4) * atomic_electric_field
    tunneling_rates = instantaneous_tunneling_rate(amplitudes, -rydberg)

    si.vis.xy_plot(
        f'tunneling_rate_vs_field_amplitude',
        amplitudes,
        tunneling_rates * asec,
        line_kwargs = [{'linewidth': BIG_LINEWIDTH}],
        x_label = fr'Electric Field Amplitude ${ion.LATEX_EFIELD}$', x_unit = 'atomic_electric_field',
        y_label = r'Tunneling Rate $\Gamma$ ($\mathrm{as}^{-1}$)',
        title = 'Tunneling Ionization Rate',
        grid_kwargs = BETTER_GRID_KWARGS,
        **BIG_FONTS,
        **FULL_PAGE_KWARGS,
        **PLOT_KWARGS,
    )


def tunneling_ionization():
    z = np.linspace(-5, 50, 1000) * bohr_radius
    amplitude = .02 * atomic_electric_field

    coul_pot = -coulomb_constant * (proton_charge ** 2) / np.abs(z)
    elec_pot = -proton_charge * amplitude * z

    fm = si.vis.xy_plot(
        get_func_name() + f'__amp={uround(amplitude, atomic_electric_field, 5)}aef',
        z,
        coul_pot + elec_pot,
        coul_pot,
        elec_pot,
        line_labels = [r'$ V_{\mathrm{Coul}} + V_{\mathrm{Field}} $',
                       r'$ V_{\mathrm{Coul}} $',
                       r'$ V_{\mathrm{Field}} $'],
        line_kwargs = [{'linewidth': BIG_LINEWIDTH},
                       {'linewidth': BIG_LINEWIDTH, 'linestyle': '--'},
                       {'linewidth': BIG_LINEWIDTH, 'linestyle': '--'}],
        hlines = [ion.HydrogenBoundState(1, 0).energy],
        hline_kwargs = [{'linewidth': BIG_LINEWIDTH, 'linestyle': ':', 'color': 'black'}],
        y_lower_limit = -2 * hartree,
        y_upper_limit = 0 * hartree,
        y_unit = 'eV',
        y_label = 'Potential Energy $ V(z) $',
        x_unit = 'bohr_radius',
        x_label = r'Distance $ z $',
        title = r'Effective Potential Along $z$-axis',
        grid_kwargs = BETTER_GRID_KWARGS,
        **BIG_FONTS,
        **FULL_PAGE_KWARGS,
        **PLOT_KWARGS,
        close_after_exit = False,
        save_on_exit = False,
    )

    ax = fm.fig.get_axes()[0]
    y1 = ion.HydrogenBoundState(1, 0).energy
    y2 = (coul_pot + elec_pot)
    ax.fill_between(
        z / bohr_radius,
        y1 / eV,
        y2 / eV,
        where = y1 > y2,
        # facecolor = 'none',
        # edgecolor = 'purple',
        interpolate = True,
        # hatch = 'X'
        facecolor = 'black',
        alpha = 0.5,
    )

    fm.save()
    fm.cleanup()


def richardson_colormap():
    colormap = plt.get_cmap('richardson')
    norm = si.vis.RichardsonNormalization()
    shading = 'gouraud'

    r = np.linspace(0, 10, 200)
    theta = np.linspace(0, twopi, 200)

    theta_mesh, r_mesh = np.meshgrid(r, theta, indexing = 'ij')
    plot_mesh = .25 * r_mesh * np.exp(1j * theta_mesh)

    fig = si.vis.get_figure(fig_width = 6, aspect_ratio = 1, fig_dpi_scale = 6)
    axis = fig.add_axes([.05, .05, .9, .9], projection = 'polar')

    color_mesh = axis.pcolormesh(theta_mesh,
                                 r_mesh,
                                 plot_mesh,
                                 shading = shading,
                                 cmap = colormap,
                                 norm = norm)

    axis.set_theta_zero_location('E')
    axis.set_theta_direction('counterclockwise')
    axis.set_rlabel_position(80)

    axis.grid(True, color = 'black', linewidth = 2, alpha = 0.4, )  # change grid color to make it show up against the colormesh
    angle_labels = [r'0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$']  # \u00b0 is unicode degree symbol
    axis.set_thetagrids(np.arange(0, 359, 45), frac = 1.2, labels = angle_labels)

    axis.tick_params(axis = 'both', which = 'major', labelsize = 30)  # increase size of tick labels
    # axis.tick_params(axis = 'y', which = 'major', colors = 'black', pad = 3)
    axis.set_yticklabels([])

    axis.axis('tight')

    si.vis.save_current_figure(get_func_name(), img_format = 'png', target_dir = OUT_DIR, transparent = False)


def cep__ionization_scan__sinc():
    jp_names = [
        'job_processors/hyd__cep_scan__200as_1jcm2__fast.job',
        'job_processors/hyd__cep_scan__800as_1jcm2__fast.job',
    ]
    pws = [200, 800]

    for jp_name, pw in zip(jp_names, pws):
        jp = clu.JobProcessor.load(jp_name)

        metrics = ['final_initial_state_overlap', 'final_bound_state_overlap']
        styles = ['-', '--']

        labels = [
            r'$ \left| \left\langle \Psi_{\mathrm{final}} | \Psi_{\mathrm{initial}} \right\rangle \right|^2 $',
            r'$ \left| \left\langle \Psi_{\mathrm{final}} | \Psi_{\mathrm{bound}} \right\rangle \right|^2 $',
        ]

        metric_to_style = dict(zip(metrics, styles))

        results = jp.data.values()

        extra_line_kwargs = dict(
            linewidth = BIG_LINEWIDTH,
        )

        si.vis.xxyy_plot(
            f'cep_scan__sinc__hyd__pw={pw}as',
            [
                *[[r.phase for r in results] for metric in metrics]
            ],
            [
                *[[getattr(r, metric) for r in results] for metric in metrics]
            ],
            line_labels = labels,
            line_kwargs = [{'linestyle': metric_to_style[metric], **extra_line_kwargs} for metric in metrics],
            title = fr'CEP Scan: Sinc Pulse w/ $\tau = {pw} \, \mathrm{{as}}, \; H = 1 \, \mathrm{{J/cm^2}}$',
            # title_offset = 1.075,
            x_label = r'Carrier-Envelope Phase $\varphi$',
            x_unit = 'rad',
            y_label = 'Final State Projection',
            grid_kwargs = BETTER_GRID_KWARGS,
            **BIG_FONTS,
            **FULL_PAGE_KWARGS,
            **PLOT_KWARGS,
        )


def get_max_or_rms_electric_field(pulse, times, which = 'max'):
    ef = pulse.get_electric_field_amplitude(times)

    if which == 'max':
        return np.max(np.abs(ef))
    elif which == 'rms':
        return np.sqrt(np.mean(ef ** 2))


def get_max_or_rms_vector_potential(pulse, times, which = 'max'):
    vp = pulse.get_vector_potential_amplitude_numeric_cumulative(times)

    if which == 'max':
        return np.max(np.abs(vp))
    elif which == 'rms':
        return np.sqrt(np.mean(vp ** 2))


def add_arrow(line, position = None, index = None, direction = 'right', size = 15, color = None, arrowprops = None):
    """
    add an arrow to a line.

    from https://stackoverflow.com/questions/34017866/arrow-on-a-line-plot-with-matplotlib

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    if arrowprops is None:
        arrowprops = {}
    arrowprops = {**dict(arrowstyle = "->", color = color), **arrowprops}

    x_data = line.get_xdata()
    y_data = line.get_ydata()

    if index is None:
        if position is None:
            position = x_data[len(x_data) // 2]
        # find closest index
        start_ind = np.argmin(np.abs(x_data - position))
    else:
        start_ind = index

    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate(
        '',
        xytext = (x_data[start_ind], y_data[start_ind]),
        xy = (x_data[end_ind], y_data[end_ind]),
        arrowprops = arrowprops,
        size = size
    )


def ionization_vs_field_properties():
    jp_names = [
        'job_processors/hyd__cep_scan__200as_1jcm2__fast.job',
        'job_processors/hyd__cep_scan__800as_1jcm2__fast.job',
    ]
    pws = [200, 800]

    for jp_name, pw in zip(jp_names, pws):
        jp = clu.JobProcessor.load(jp_name)

        metrics = ['final_initial_state_overlap', 'final_bound_state_overlap']
        styles = ['-', '--']

        labels = [
            r'$ \left| \left\langle \Psi_{\mathrm{final}} | \Psi_{\mathrm{initial}} \right\rangle \right|^2 $',
            r'$ \left| \left\langle \Psi_{\mathrm{final}} | \Psi_{\mathrm{bound}} \right\rangle \right|^2 $',
        ]

        metric_to_style = dict(zip(metrics, styles))

        results = jp.data.values()

        extra_line_kwargs = dict(
            linewidth = BIG_LINEWIDTH,
        )

        tb = (pw * asec) * 35
        times = np.linspace(-tb, tb, 1e4)

        # MAX AFIELD

        fm = si.vis.xxyy_plot(
            f'vp_scan__sinc__hyd__max_vp__pw={pw}as',
            [
                *[[proton_charge * get_max_or_rms_vector_potential(r.electric_potential, times, which = 'max') for r in results] for metric in metrics]
            ],
            [
                *[[getattr(r, metric) for r in results] for metric in metrics]
            ],
            line_labels = labels,
            line_kwargs = [{'linestyle': metric_to_style[metric], **extra_line_kwargs} for metric in metrics],
            title = fr'VP Scan: Sinc Pulse w/ $\tau = {pw} \, \mathrm{{as}}, \; H = 1 \, \mathrm{{J/cm^2}}$',
            # title_offset = 1.075,
            x_label = rf'$ \max \; e \, \left| {ion.LATEX_AFIELD}_{{\varphi}}(t) \right| $',
            x_unit = 'atomic_momentum',
            y_label = 'Final State Projection',
            grid_kwargs = BETTER_GRID_KWARGS,
            **BIG_FONTS,
            **FULL_PAGE_KWARGS,
            **PLOT_KWARGS,
            save_after_exit = False,
            close_after_exit = False,
        )

        second_line = fm.elements['lines'][-1]
        props = dict(
            linewidth = 5,
        )
        arrow_size = 40

        for index in [10, 20, 30, 70, 80, 90]:
            add_arrow(second_line, size = arrow_size, index = index, arrowprops = props)

        fm.save()
        fm.cleanup()

        ## MAX EFIELD

        fm = si.vis.xxyy_plot(
            f'ef_scan__sinc__hyd__max_ef__pw={pw}as',
            [
                *[[get_max_or_rms_electric_field(r.electric_potential, times, which = 'max') for r in results] for metric in metrics]
            ],
            [
                *[[getattr(r, metric) for r in results] for metric in metrics]
            ],
            line_labels = labels,
            line_kwargs = [{'linestyle': metric_to_style[metric], **extra_line_kwargs} for metric in metrics],
            title = fr'EF Scan: Sinc Pulse w/ $\tau = {pw} \, \mathrm{{as}}, \; H = 1 \, \mathrm{{J/cm^2}}$',
            # title_offset = 1.075,
            x_label = rf'$ \max \; \left| {ion.LATEX_EFIELD}_{{\varphi}}(t) \right| $',
            x_unit = 'atomic_electric_field',
            y_label = 'Final State Projection',
            grid_kwargs = BETTER_GRID_KWARGS,
            **BIG_FONTS,
            **FULL_PAGE_KWARGS,
            **PLOT_KWARGS,
            save_after_exit = False,
            close_after_exit = False,
        )

        second_line = fm.elements['lines'][-1]
        props = dict(
            linewidth = 5,
        )
        arrow_size = 40

        for index in [10, 20, 30, 70, 80, 90]:
            add_arrow(second_line, size = arrow_size, index = index, arrowprops = props)

        fm.save()
        fm.cleanup()

        # # RMS
        # fm = si.vis.xxyy_plot(
        #     f'vp_scan__sinc__hyd__rms_vp',
        #     [
        #         *[[proton_charge * get_max_or_rms_vector_potential(r.electric_potential, times, which = 'rms') for r in results] for metric in metrics]
        #     ],
        #     [
        #         *[[getattr(r, metric) for r in results] for metric in metrics]
        #     ],
        #     line_labels = labels,
        #     line_kwargs = [{'linestyle': metric_to_style[metric], **extra_line_kwargs} for metric in metrics],
        #     title = r'VP Scan: Sinc Pulse w/ $\tau = 200 \, \mathrm{as}, \; H = 1 \, \mathrm{J/cm^2}$',
        #     # title_offset = 1.075,
        #     x_label = rf'$ \mathrm{{RMS}}_t \; e \, {ion.LATEX_AFIELD}_{{\varphi}}(t) $',
        #     x_unit = 'atomic_momentum',
        #     # y_label = 'Ionization Metric',
        #     grid_kwargs = BETTER_GRID_KWARGS,
        #     **BIG_FONTS,
        #     **FULL_PAGE_KWARGS,
        #     **PLOT_KWARGS,
        #     save_after_exit = False,
        #     close_after_exit = False,
        # )
        #
        # second_line = fm.elements['lines'][-1]
        #
        # for index in [10, 20, 30, 70, 80, 90]:
        #     add_arrow(second_line, size = arrow_size, index = index, arrowprops = props)
        #
        # fm.save()


def get_omega_carrier(sim_result):
    return sim_result.electric_potential[0].omega_carrier


def omega_min_scan():
    jp = clu.JobProcessor.load('job_processors/hyd__omega_min_scan__gaussian__fixed_bounds.job')

    phases = jp.parameter_set('phase')

    colors = ['C0', 'C2', 'C1']

    selectors = dict(
        pulse_width = 200 * asec,
        fluence = 1 * Jcm2,
    )

    si.vis.xxyy_plot(
        'omega_min_scan__gaussian',
        [
            *[[get_omega_carrier(r) / twopi for r in jp.select_by_kwargs(phase = phase, **selectors)] for phase in phases]
        ],
        [
            *[[r.final_bound_state_overlap for r in jp.select_by_kwargs(phase = phase, **selectors)] for phase in phases],
        ],
        line_labels = [r'$ \varphi = 0 $', r'$ \varphi = \pi / 4 $', r'$ \varphi = \pi / 2 $'],
        line_kwargs = [{'linewidth': BIG_LINEWIDTH, 'color': color} for color in colors],
        grid_kwargs = BETTER_GRID_KWARGS,
        x_label = '$ f_{\mathrm{carrier}} $', x_unit = 'THz',
        y_label = 'Final Bound State Overlap',
        title = 'Carrier Frequency Scan: Gaussian Pulse',
        title_offset = 1.075,
        **BIG_FONTS,
        **FULL_PAGE_KWARGS,
        **PLOT_KWARGS,
    )


def ide_symmetry():
    pulse_width = 200
    fluence = .3
    phase = pi / 4

    time_bound = 10
    plot_bound = 4

    efields = [ion.SincPulse(pulse_width = pulse_width * asec, fluence = fluence * Jcm2, phase = phase),
               ion.SincPulse(pulse_width = pulse_width * asec, fluence = fluence * Jcm2, phase = pi - phase)]

    test_charge = electron_charge
    test_mass = electron_mass_reduced
    test_width = bohr_radius
    # tau_alpha = 4 * m * (L ** 2) / hbar
    # prefactor = -np.sqrt(pi) * (L ** 2) * ((q / hbar) ** 2)

    spec_kwargs = dict(
        time_initial = -time_bound * pulse_width * asec, time_final = time_bound * pulse_width * asec,
        time_step = 1 * asec,
        prefactor = ide.gaussian_prefactor_LEN(test_width, test_charge),
        kernel = ide.gaussian_kernel_LEN,
        kernel_kwargs = dict(tau_alpha = ide.gaussian_tau_alpha_LEN(test_width, test_mass)),
        evolution_gauge = 'LEN',
        evolution_method = 'RK4',
        electric_potential_DC_correction = True,
    )

    specs = []
    for efield in efields:
        specs.append(ide.IntegroDifferentialEquationSpecification(efield.phase,
                                                                  electric_potential = efield,
                                                                  **spec_kwargs,
                                                                  ))
    results = si.utils.multi_map(run, specs, processes = 2)

    fig = si.vis.get_figure(fig_width = si.vis.PPT_WIDESCREEN_WIDTH, fig_height = si.vis.PPT_WIDESCREEN_HEIGHT, fig_dpi_scale = 6, )

    grid_spec = matplotlib.gridspec.GridSpec(2, 1, height_ratios = [2.5, 1], hspace = 0.07)
    ax_upper = plt.subplot(grid_spec[0])
    ax_lower = plt.subplot(grid_spec[1], sharex = ax_upper)

    # font_size_axis_labels = 35,
    # font_size_tick_labels = 25,
    # font_size_legend = 30,
    # font_size_title = 35,

    for result, cep, color, style in zip(results, (r'$\varphi = \pi / 4$', r'$\varphi = - \pi / 4$'), ('C0', 'C1'), ('-', '-')):
        ax_lower.plot(result.times / asec, result.spec.electric_potential.get_electric_field_amplitude(result.times) / atomic_electric_field, color = color, linewidth = BIG_LINEWIDTH, label = cep,
                      linestyle = style)
        ax_upper.plot(result.times / asec, result.a2, color = color, linewidth = BIG_LINEWIDTH, label = cep, linestyle = style)

    efield_1 = results[0].spec.electric_potential.get_electric_field_amplitude(results[0].times)
    field_max = np.max(efield_1)
    field_min = np.min(efield_1)
    field_range = np.abs(field_max - field_min)

    ax_lower.set_xlabel(r'Time Time $t$ ($\mathrm{as}$)', fontsize = 35)
    ax_lower.set_ylabel(r'$   \mathcal{E}(t) $ ($\mathrm{a.u.}$)   ', fontsize = 35)
    # ax_lower.yaxis.set_label_coords(-.125, .5)

    ax_upper.set_ylabel(r'$ \left| a_{\alpha}(t) \right|^2 $', fontsize = 35)

    # ax_lower.set_xticks([i * pulse_width * asec for i in range(-10, 11)])
    # # ax_lower.set_xticklabels([r'$0$'] + [r'$ {} \tau $'.format(i) for i in range(-10, 11) if i != 0])
    # x_tick_labels = []
    # for i in range(-10, 11):
    #     if i == 0:
    #         x_tick_labels.append(r'$0$')
    #     elif i == 1:
    #         x_tick_labels.append(r'$\tau$')
    #     elif i == -1:
    #         x_tick_labels.append(r'$-\tau$')
    #     else:
    #         x_tick_labels.append(r'$ {} \tau $'.format(i))
    # ax_lower.set_xticklabels(x_tick_labels)

    # ax_lower.set_yticks([0, field_max, field_max / np.sqrt(2), -field_max, -field_max / np.sqrt(2)])
    # ax_lower.set_yticklabels([
    #     r'$0$',
    #     r'$\mathcal{E}_{t} $',
    #     r'$\mathcal{E}_{t} / \sqrt{2} $',
    #     r'$-\mathcal{E}_{t} $',
    #     r'$-\mathcal{E}_{t} / \sqrt{2} $',
    # ])

    ax_upper.tick_params(labelright = True, labelsize = 25)
    ax_lower.tick_params(labelright = True, labelsize = 25)
    ax_upper.xaxis.tick_top()

    ax_lower.grid(True, **BETTER_GRID_KWARGS)
    ax_upper.grid(True, **BETTER_GRID_KWARGS)

    ax_lower.set_xlim(-plot_bound * pulse_width, plot_bound * pulse_width)
    # ax_lower.set_ylim(field_min - (.125 * field_range), field_max + (.125 * field_range))
    ax_upper.set_ylim(.55, 1)

    ax_upper.legend(loc = 'best', fontsize = 30)
    fig.set_tight_layout(True)

    # save_figure(get_func_name())
    si.vis.save_current_figure(get_func_name(), target_dir = OUT_DIR, img_format = 'png')


if __name__ == '__main__':
    with logman as logger:
        figures = [
            tunneling_ionization,
            # ide_symmetry,
            # omega_min_scan,
            # cep__ionization_scan__sinc,
            # ionization_vs_field_properties,
            # richardson_colormap,
            # instantaneous_tunneling_rate_plot,
            # field_properties_vs_phase_v2,
            # pulse_width_scan__hyd__sinc,
            # pulse_width_scan__ide__sinc,
            # pulse_width_scan__hyd__gaussian,
            # fluence_scan__hyd__sinc,
            # fluence_scan__hyd__gaussian,
            # functools.partial(multicycle_sine_cosine_comparison, ion.GaussianPulse, twopi * 30 * THz, ', Few-cycle'),
            # functools.partial(multicycle_sine_cosine_comparison, ion.SincPulse, twopi * 30 * THz, ', Few-cycle'),
            # functools.partial(multicycle_sine_cosine_comparison, ion.GaussianPulse, twopi * 500 * THz, ', Many-cycle'),
            # functools.partial(multicycle_sine_cosine_comparison, ion.SincPulse, twopi * 500 * THz, ', Many-cycle'),
            # functools.partial(multicycle_sine_cosine_comparison, ion.GaussianPulse, twopi * 2000 * THz, ', Many-cycle'),
            # functools.partial(multicycle_sine_cosine_comparison, ion.SincPulse, twopi * 2000 * THz, ', Many-cycle'),
            # pulse_ffts,
            # delta_kicks_eta_plot,
            # delta_kick_decomposition_plot,
            # length_ide_kernel_gaussian,
            # tunneling_ionization_animation__pulse,
            # tunneling_ionization_animation,
            # sinc_pulse,
            # gaussian_pulse,
            # efield_and_afield,
            # title_bg,
            # spherical_harmonic_mesh,
            # functools.partial(pulse_cep_movie, pulse_type = ion.GaussianPulse, prefix = 'Gaussian Pulse'),
            # functools.partial(pulse_cep_movie_zoom, pulse_type = ion.GaussianPulse, prefix = 'Gaussian Pulse'),
            # functools.partial(pulse_cep_movie, pulse_type = ion.SincPulse, prefix = 'Sinc Pulse'),
        ]

        for fig in tqdm(figures):
            fig()
