import functools
import logging
import os
import sys
from copy import deepcopy
import collections

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

FULL_SLIDE_FIGMAN_KWARGS = dict(
    fig_width = si.vis.PPT_WIDESCREEN_WIDTH,
    fig_height = si.vis.PPT_WIDESCREEN_HEIGHT,
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
        **FULL_SLIDE_FIGMAN_KWARGS,
        **PLOT_KWARGS
    )
    sim.mesh.plot_g2(
        plot_limit = 25 * bohr_radius,
        tick_label_size = 20,
        axis_label_size = 35,
        show_title = False,
        show_colorbar = False,
        grid_kwargs = {'linewidth': 2},
        **FULL_SLIDE_FIGMAN_KWARGS,
        **PLOT_KWARGS
    )

    for which in ('g', 'g2'):
        with si.vis.FigureManager(f'title_bg__{which}',
                                  tight_layout = False,
                                  **FULL_SLIDE_FIGMAN_KWARGS,
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
        **FULL_SLIDE_FIGMAN_KWARGS,
        **PLOT_KWARGS,
    )
    sim.mesh.plot_g(
        plot_limit = 25 * bohr_radius,
        tick_label_size = 20,
        title_size = 30,
        grid_kwargs = {'linewidth': 2},
        **FULL_SLIDE_FIGMAN_KWARGS,
        **PLOT_KWARGS
    )
    sim.mesh.plot_g2(
        plot_limit = 25 * bohr_radius,
        tick_label_size = 20,
        title_size = 30,
        show_colorbar = False,
        grid_kwargs = {'linewidth': 2},
        **FULL_SLIDE_FIGMAN_KWARGS,
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
        line_labels = [f'$ {ion.LATEX_EFIELD}(t) $', f'$ q \, {ion.LATEX_AFIELD}(t) $'],
        line_kwargs = [{'linewidth': 3}, {'linewidth': 3, 'linestyle': '--'}],
        x_label = r'$t$', x_unit = 'asec',
        y_label = r'Amplitude (a.u.)',
        y_lower_limit = -1.6, y_upper_limit = 1,
        title = r'Windowed Sine Wave Pulse w/ $E = 10 \, \mathrm{eV}$',
        font_size_axis_labels = 35,
        font_size_tick_labels = 20,
        font_size_legend = 25,
        font_size_title = 35,
        legend_kwargs = {'loc': 'lower left'},
        grid_kwargs = BETTER_GRID_KWARGS,
        **FULL_SLIDE_FIGMAN_KWARGS,
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
        legend_kwargs = {'loc': 'lower left'},
        grid_kwargs = BETTER_GRID_KWARGS,
        **FULL_SLIDE_FIGMAN_KWARGS,
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
        line_labels = [f'$ {ion.LATEX_EFIELD}(t) $', f'$ q \, {ion.LATEX_AFIELD}(t) $'],
        line_kwargs = [{'linewidth': 3}, {'linewidth': 3, 'linestyle': '--'}],
        x_label = r'$t$', x_unit = 'asec',
        title = fr'Windowed Sinc Pulse w/ $\tau = {uround(pw, asec, 0)} \, \mathrm{{as}}, \, H = {uround(flu, Jcm2)} \, \mathrm{{J/cm^2}} $',
        font_size_axis_labels = 35,
        font_size_tick_labels = 20,
        font_size_legend = 25,
        font_size_title = 35,
        grid_kwargs = BETTER_GRID_KWARGS,
        **FULL_SLIDE_FIGMAN_KWARGS,
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
        line_labels = [f'$ {ion.LATEX_EFIELD}(t) $', f'$ q \, {ion.LATEX_AFIELD}(t) $'],
        line_kwargs = [{'linewidth': 3}, {'linewidth': 3, 'linestyle': '--'}],
        x_label = r'$t$', x_unit = 'asec',
        y_label = r'Amplitude (a.u.)',
        title = fr'Gaussian Pulse w/ $\tau = {uround(pw, asec, 0)} \, \mathrm{{as}}, \, H = {uround(flu, Jcm2)} \, \mathrm{{J/cm^2}} $',
        font_size_axis_labels = 35,
        font_size_tick_labels = 20,
        font_size_legend = 25,
        font_size_title = 35,
        grid_kwargs = BETTER_GRID_KWARGS,
        **FULL_SLIDE_FIGMAN_KWARGS,
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
        line_labels = [fr'$ {ion.LATEX_EFIELD}(t) $', fr'$ q \, {ion.LATEX_AFIELD}(t) $'],
        line_kwargs = [{'linewidth': 3}, {'linestyle': '--', 'linewidth': 3}],
        x_label = r'Time', x_unit = 'asec',
        y_label = r'Amplitude (a.u.)',
        t_fmt_string = r'$ \varphi = {} \; {} $', t_unit = 'rad',
        title = fr'{prefix} w/ $\tau = {uround(pw, asec, 0)} \, \mathrm{{as}}, \, H = {uround(flu, Jcm2)} \, \mathrm{{J/cm^2}} $',
        progress_bar = True,
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

    si.vis.xyt_plot(
        f'cep_movie__{pulse_type.__name__}_intensity',
        times,
        phases,
        efield_intensity,
        line_kwargs = [{'linewidth': 3}],
        x_label = r'Time', x_unit = 'asec',
        y_label = r'$ P(t) $', y_unit = 'atomic_intensity',
        t_fmt_string = r'$ \varphi = {} \; {} $', t_unit = 'rad',
        title = fr'{prefix} w/ $\tau = {uround(pw, asec, 0)} \, \mathrm{{as}}, \, H = {uround(flu, Jcm2)} \, \mathrm{{J/cm^2}} $',
        progress_bar = True,
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
    #     line_kwargs = [{'linewidth': 3}, {'linestyle': '--', 'linewidth': 3}],
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
        line_kwargs = [{'linewidth': 3}],
        x_label = r'Time', x_unit = 'asec',
        y_label = r'Intensity', y_unit = 'atomic_intensity',
        t_fmt_string = r'$ \varphi = {} \; {} $', t_unit = 'rad',
        title = fr'{prefix} w/ $\tau = {uround(pw, asec, 0)} \, \mathrm{{as}}, \, H = {uround(flu, Jcm2)} \, \mathrm{{J/cm^2}} $',
        progress_bar = True,
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
        line_kwargs = [{'animated': True, 'linewidth': 3}, {'linestyle': '--', 'linewidth': 3}, {'linestyle': '--', 'animated': True, 'linewidth': 3}],
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
        line_kwargs = [{'color': 'black', 'linewidth': 3},
                       {'color': 'C0', 'linewidth': 3},
                       {'color': 'C1', 'linewidth': 3}],
        x_label = r"$t-t'$ ($\mathrm{\tau_{\alpha}}$)", x_unit = tau,
        font_size_axis_labels = 35,
        font_size_tick_labels = 20,
        font_size_legend = 25,
        font_size_title = 35,
        grid_kwargs = BETTER_GRID_KWARGS,
        **FULL_SLIDE_FIGMAN_KWARGS,
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

    quarter_page_kwargs = dict(
        fig_width = 13 / 2,
        fig_height = 6.5 / 2
    )

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
            x_label = r'$t$', x_unit = 'asec',
            y_label = fr'${ion.LATEX_EFIELD}(t)$', y_unit = 'atomic_electric_field',
            line_kwargs = [{'linewidth': 3}, {'linestyle': '', 'marker': 'o'}, *[{'color': 'C1', 'linestyle': '-'} for _ in kick_times]],
            font_size_axis_labels = 22,
            font_size_tick_labels = 14,
            font_size_legend = 16,
            font_size_title = 22,
            title = f'{name}',
            grid_kwargs = BETTER_GRID_KWARGS,
            **{**FULL_SLIDE_FIGMAN_KWARGS, **quarter_page_kwargs},
            **PLOT_KWARGS,
        )

        # si.vis.xy_plot(
        #     f'decomposition__{name}_decomposed',
        #     kick_times,
        #     kick_products,
        #     line_kwargs = [{'linestyle': ':', 'marker': 'o'}],
        #     x_label = r'$t$', x_unit = 'asec',
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
        line_kwargs = [{'linewidth': 3}, {'linewidth': 3}, {'linewidth': 3},],
        x_label = r'Sine Wave Period $T$ ($ \tau_{\alpha} $)', x_unit = tau_alpha,
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
        **FULL_SLIDE_FIGMAN_KWARGS,
    )


if __name__ == '__main__':
    with logman as logger:
        figures = [
            delta_kicks_eta_plot,
            delta_kick_decomposition_plot,
            length_ide_kernel_gaussian,
            tunneling_ionization_animation__pulse,
            tunneling_ionization_animation,
            sinc_pulse,
            gaussian_pulse,
            efield_and_afield,
            title_bg,
            spherical_harmonic_mesh,
            functools.partial(pulse_cep_movie, pulse_type = ion.GaussianPulse, prefix = 'Gaussian Pulse'),
            functools.partial(pulse_cep_movie_zoom, pulse_type = ion.GaussianPulse, prefix = 'Gaussian Pulse'),
            functools.partial(pulse_cep_movie, pulse_type = ion.SincPulse, prefix = 'Sinc Pulse'),
        ]

        for fig in tqdm(figures):
            fig()
