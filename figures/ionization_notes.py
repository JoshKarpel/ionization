import functools
import itertools
import logging
import os
import sys

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

log = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.INFO)

pgf_with_latex = {  # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
    "text.usetex": True,  # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 11,  # LaTeX default is 10pt font.
    "font.size": 11,
    "legend.fontsize": 10,  # Make the legend/label fonts a little smaller
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.figsize": si.vis._get_fig_dims(0.95),  # default fig size of 0.95 \textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",  # use utf8 fonts because your computer can handle it :)
        r"\usepackage[T1]{fontenc}",  # plots will be generated using this preamble
    ]
}
matplotlib.rcParams.update(pgf_with_latex)

import matplotlib.pyplot as plt


def run(spec):
    sim = spec.to_simulation()
    sim.run_simulation()
    return sim


PLOT_KWARGS_LIST = [
    dict(
        fig_width = si.vis.points_to_inches(498.66),
        target_dir = OUT_DIR,
        img_format = 'pdf',
    ),
    dict(
        fig_width = si.vis.points_to_inches(498.66),
        target_dir = OUT_DIR,
        img_format = 'pgf',
    ),
    dict(
        fig_width = si.vis.points_to_inches(498.66),
        target_dir = OUT_DIR,
        img_format = 'png',
        fig_dpi_scale = 6,
    ),
]


def save_figure(filename):
    for kwargs in PLOT_KWARGS_LIST:
        si.vis.save_current_figure(filename, **kwargs)


def get_func_name():
    return sys._getframe(1).f_code.co_name


grid_kwargs = {
    # 'dashes': [.5, .5],
    'linestyle': '-',
    'color': 'black',
    'linewidth': .25,
    'alpha': 0.4
}


def sinc_pulse_power_spectrum_full():
    fig = si.vis.get_figure('full')
    ax = fig.add_subplot(111)

    lower = .15
    upper = .85
    c = (lower + upper) / 2

    omega = np.linspace(-1, 1, 1000)
    power = np.where(np.abs(omega) < upper, 1, 0) * np.where(np.abs(omega) > lower, 1, 0)

    ax.fill_between(omega, 0, power, alpha = 1, edgecolor = 'black', facecolor = 'darkgray')

    delta_line_y_coord = .75
    plt.annotate(s = '', xy = (lower, delta_line_y_coord), xytext = (upper, delta_line_y_coord), textcoords = 'data', arrowprops = dict(arrowstyle = '<->'))
    plt.text(c + .1 * (upper - c), delta_line_y_coord + .025, r'$\Delta_{\omega}$')

    plt.annotate(s = '', xy = (-lower, delta_line_y_coord), xytext = (-upper, delta_line_y_coord), textcoords = 'data', arrowprops = dict(arrowstyle = '<->'))
    plt.text(-c - .1 * (lower - c), delta_line_y_coord + .025, r'$\Delta_{\omega}$')

    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1.2)

    ax.set_xlabel(r'$   \omega  $')
    ax.set_ylabel(r'$   \left|   \widehat{   \mathcal{E}    }(\omega)  \right|^2   $')
    ax.yaxis.set_label_coords(-.1, .2)

    ax.set_xticks([-upper, -c, -lower, 0, lower, c, upper])
    ax.set_xticklabels([r'$-\omega_{\mathrm{max}}$',
                        r'$-\omega_{\mathrm{c}}$',
                        r'$-\omega_{\mathrm{min}}$',
                        r'$0$',
                        r'$\omega_{\mathrm{min}}$',
                        r'$\omega_{\mathrm{c}}$',
                        r'$\omega_{\mathrm{max}}$'
                        ])
    ax.set_yticks([0, 1])
    ax.set_yticklabels([
        r'$0$',
        r'$\left|   \mathcal{E}_{\omega}      \right|^2$',
    ])

    ax.grid(True, **grid_kwargs)

    save_figure(get_func_name())


def sinc_pulse_power_spectrum_half():
    fig = si.vis.get_figure('full')
    ax = fig.add_subplot(111)

    lower = .15
    upper = .85
    carrier = (lower + upper) / 2

    omega = np.linspace(0, 1, 1000)
    power = np.where(np.abs(omega) < upper, 1, 0) * np.where(np.abs(omega) > lower, 1, 0)

    ax.fill_between(omega, 0, power, alpha = 1, edgecolor = 'black', facecolor = 'darkgray')

    delta_line_y_coord = .75
    plt.annotate(s = '', xy = (lower, delta_line_y_coord), xytext = (upper, delta_line_y_coord), textcoords = 'data', arrowprops = dict(arrowstyle = '<->'))
    plt.text(carrier + .1 * (upper - carrier), delta_line_y_coord + 0.025, r'$\Delta_{\omega}$')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.2)

    ax.set_xlabel(r'$   \omega  $')
    ax.set_ylabel(r'$   \left|   \widehat{   \mathcal{E}    }(\omega)  \right|^2   $')
    ax.yaxis.set_label_coords(-.1, .2)

    ax.set_xticks([0, lower, carrier, upper])
    ax.set_xticklabels([r'$0$',
                        r'$\omega_{\mathrm{min}}$',
                        r'$\omega_{\mathrm{c}}$',
                        r'$\omega_{\mathrm{max}}$'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r'$0$', r'$\left|   \mathcal{E}_{\omega}      \right|^2$'])

    ax.grid(True, **grid_kwargs)

    save_figure(get_func_name())


def sinc_pulse_electric_field(phase = 0):
    fig = si.vis.get_figure('half')
    ax = fig.add_subplot(111)

    omega_min = twopi
    omega_max = 20 * twopi

    omega_c = (omega_min + omega_max) / 2
    delta = omega_max - omega_min

    time = np.linspace(-1, 1, 1000)
    field = si.math.sinc(delta * time / 2) * np.cos(omega_c * time + phase * pi)

    ax.plot(time, field, color = 'black', linewidth = 1.5)

    ax.set_xlim(-.18, .18)
    ax.set_ylim(-1.2, 1.2)

    ax.set_xlabel(r'$   t  $')
    ax.set_ylabel(r'$   \mathcal{E}(t)   $')
    ax.yaxis.set_label_coords(-.125, .5)

    d = twopi / delta
    ax.set_xticks([0, -3 * d, -2 * d, -d, d, 2 * d, 3 * d])
    ax.set_xticklabels([r'$0$',
                        r'$ -3 \frac{2\pi}{\Delta_{\omega}}  $',
                        # r'$ -2 \frac{2\pi}{\Delta_{\omega}}   $',
                        r'',
                        r'$ - \frac{2\pi}{\Delta_{\omega}}   $',
                        r'$  \frac{2\pi}{\Delta_{\omega}}   $',
                        # r'$  2 \frac{2\pi}{\Delta_{\omega}}   $',
                        r'',
                        r'$  3 \frac{2\pi}{\Delta_{\omega}}   $',
                        ])

    ax.set_yticks([0, 1, 1 / np.sqrt(2), -1, -1 / np.sqrt(2)])
    ax.set_yticklabels([
        r'$0$',
        r'$\mathcal{E}_{t} $',
        r'$\mathcal{E}_{t} / \sqrt{2} $',
        # r'$ \frac{ \mathcal{E}_{t} }{ \sqrt{2} } $',
        r'$-\mathcal{E}_{t} $',
        r'$-\mathcal{E}_{t} / \sqrt{2} $',
        # r'$ -\frac{ \mathcal{E}_{t} }{ \sqrt{2} } $',
    ])

    ax.grid(True, **grid_kwargs)

    save_figure(get_func_name() + '_phase={}'.format(phase))


def gaussian_pulse_power_spectrum_half():
    # TODO: er, be caereful, is delta for the power spectrum or the amplitude spectrum?
    fig = si.vis.get_figure('full')
    ax = fig.add_subplot(111)

    carrier = .6
    sigma = .1
    delta = 2 * np.sqrt(2 * np.log(2)) * sigma

    omega = np.linspace(0, 1, 1000)
    power = np.exp(-.5 * (((omega - carrier) / sigma) ** 2)) / np.sqrt(twopi) / sigma
    max_power = np.max(power)
    power /= max_power

    ax.fill_between(omega, 0, power, alpha = 1, edgecolor = 'black', facecolor = 'darkgray')

    delta_line_y_coord = .3
    plt.annotate(s = '', xy = (carrier - delta / 2, delta_line_y_coord), xytext = (carrier + delta / 2, delta_line_y_coord), textcoords = 'data', arrowprops = dict(arrowstyle = '<->'))
    plt.text(carrier + sigma / 5, delta_line_y_coord - .1, r'$\Delta_{\omega}$')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.2)

    ax.set_xlabel(r'$   \omega  $')
    ax.set_ylabel(r'$   \left|   \widehat{   \mathcal{E}    }(\omega)  \right|^2   $')
    ax.yaxis.set_label_coords(-.1, .2)

    ax.set_xticks([0, carrier, carrier - delta / 2, carrier + delta / 2])
    ax.set_xticklabels([r'$0$',
                        r'$  \omega_{\mathrm{c}}  $',
                        r'$  \omega_{\mathrm{c}} - \frac{\Delta}{2}   $',
                        r'$  \omega_{\mathrm{c}} + \frac{\Delta}{2}   $',
                        ])
    ax.set_yticks([0, .5, 1])
    ax.set_yticklabels([
        r'$0$',
        r'$\frac{1}{2}   \left|   \mathcal{E}_{\omega}    \right|^2$',
        r'$\left|   \mathcal{E}_{\omega}      \right|^2$',
    ])

    ax.grid(True, **grid_kwargs)

    save_figure(get_func_name())


def finite_square_well():
    fig = si.vis.get_figure('full')
    ax = fig.add_subplot(111)

    a_over_two = .5
    depth = -.5

    x = np.linspace(-1, 1, 1000)
    well = np.where(np.abs(x) < a_over_two, depth, 0)
    ax.plot(x, well, linewidth = 1.5, color = 'black')

    ax.set_xlim(-1, 1)
    ax.set_ylim(-.75, .25)

    ax.set_xlabel(r'$   x  $')
    ax.set_ylabel(r'$   V(x)   $')
    ax.yaxis.set_label_coords(-.05, .5)

    ax.set_xticks([0, -a_over_two, a_over_two])
    ax.set_xticklabels([r'$0$',
                        r'$  -\frac{a}{2} $',
                        r'$  \frac{a}{2} $',
                        ])
    ax.set_yticks([0, depth])
    ax.set_yticklabels([
        r'$0$',
        r'$-V_0$',
    ])

    ax.grid(True, **grid_kwargs)

    save_figure(get_func_name())


def finite_square_well_energies():
    fig = si.vis.get_figure('full')
    ax = fig.add_subplot(111)

    z_0 = 6 * pi / 2 + .5 * np.sqrt(1)  # must make it numpy data type so that the optimizer doesn't panic

    z = np.linspace(0, z_0 + 5, 1000)

    tan = np.tan(z)
    cotan = -1 / np.tan(z)
    sqrt = np.sqrt((z_0 / z) ** 2 - 1)
    sqrt[-1] = 0

    tan[tan < 0] = np.NaN
    cotan[cotan < 0] = np.NaN

    ax.plot(z, tan, color = 'C0', label = r'$\tan(z)$')
    ax.plot(z, cotan, color = 'C1', label = r'$-\cot(z)$')
    ax.plot(z, sqrt, color = 'black', label = r'$   \sqrt{  \left( \frac{z_0}{z} \right)^2  -1 }  $')

    ax.set_xlabel(r'$   z  $')
    # ax.set_ylabel(r'$   \sqrt{  \left( \frac{z_0}{z} \right)^2  -1 }  $')
    # ax.yaxis.set_label_coords(-.05, .5)

    ax.set_xticks([z_0] + list(np.arange(0, z_0 + 5, pi / 2)))
    ax.set_xticklabels([r'$z_0$', r'$0$', r'$\frac{\pi}{2}$'] + [r'${} \frac{{\pi}}{{2}}$'.format(n) for n in range(2, int(z_0 + 5))])

    intersections = []

    for n in np.arange(1, z_0 / (pi / 2) + 1.1, 1):
        left_bound = (n - 1) * pi / 2
        right_bound = min(z_0, left_bound + (pi / 2))

        if n % 2 != 0:  # n is odd
            intersection = optimize.brentq(lambda x: np.tan(x) - np.sqrt(((z_0 / x) ** 2) - 1), left_bound, right_bound)
        else:  # n is even
            intersection = optimize.brentq(lambda x: (1 / np.tan(x)) + np.sqrt(((z_0 / x) ** 2) - 1), left_bound, right_bound)

        intersections.append(intersection)

    intersections = sorted(intersections)
    intersections = np.sqrt((z_0 / intersections) ** 2 - 1)
    ax.set_yticks(intersections)
    ax.set_yticklabels([r'$n={}$'.format(n) for n in range(1, len(intersections) + 1)])

    ax.set_xlim(0, round(z_0 + 2))
    ax.set_ylim(0, 8)

    ax.grid(True, **grid_kwargs)

    ax.legend(loc = 'upper right', framealpha = 1)

    save_figure(get_func_name())


def a_alpha_v2_kernel_gaussian():
    fig = si.vis.get_figure('full')
    ax = fig.add_subplot(111)

    dt = np.linspace(-10, 10, 1000)
    tau = .5
    y = (1 + 1j * (dt / tau)) ** (-3 / 2)

    ax.plot(dt, np.abs(y), color = 'black', label = r"$\left| K(t-t') \right|$")
    ax.plot(dt, np.real(y), color = 'C0', label = r"$  \mathrm{Re} \left\lbrace K(t-t') \right\rbrace  $")
    ax.plot(dt, np.imag(y), color = 'C1', label = r"$  \mathrm{Im} \left\lbrace K(t-t') \right\rbrace   $")

    ax.set_xlabel(r"$   t-t'  $")
    ax.set_ylabel(r"$   K(t-t') = \left(1 + i \frac{t-t'}{\tau_{\alpha}}\right)^{-3/2}  $")
    # ax.yaxis.set_label_coords(-., .5)

    ax.set_xticks([0, tau, -tau, 2 * tau, -2 * tau])
    ax.set_xticklabels([r'$0$',
                        r'$\tau_{\alpha}$',
                        r'$-\tau_{\alpha}$',
                        r'$2\tau_{\alpha}$',
                        r'$-2\tau_{\alpha}$',
                        ])

    ax.set_yticks([0, 1, -1, .5, -.5, 1 / np.sqrt(2)])
    ax.set_yticklabels([r'$0$',
                        r'$1$',
                        r'$-1$',
                        r'$1/2$',
                        r'$-1/2$',
                        r'$1/\sqrt{2}$',
                        ])

    ax.set_xlim(-3, 3)
    ax.set_ylim(-.75, 1.4)

    ax.grid(True, **grid_kwargs)

    ax.legend(loc = 'upper right', framealpha = 1)

    save_figure(get_func_name())


def ide_solution_sinc_pulse_cep_symmetry(phase = 0):
    fig = si.vis.get_figure('full')

    grid_spec = matplotlib.gridspec.GridSpec(2, 1, height_ratios = [2.5, 1], hspace = 0.07)  # TODO: switch to fixed axis construction
    ax_upper = plt.subplot(grid_spec[0])
    ax_lower = plt.subplot(grid_spec[1], sharex = ax_upper)
    #
    # omega_min = twopi
    # omega_max = 20 * twopi
    #
    # omega_c = (omega_min + omega_max) / 2
    # delta = omega_max - omega_min
    #
    # time = np.linspace(-1, 1, 1000)
    # field = si.math.sinc(delta * time / 2) * np.cos(omega_c * time + phase * pi)

    pulse_width = 200
    fluence = .3

    plot_bound = 4

    efields = [ion.SincPulse(pulse_width = pulse_width * asec, fluence = fluence * Jcm2, phase = phase * pi),
               ion.SincPulse(pulse_width = pulse_width * asec, fluence = fluence * Jcm2, phase = -phase * pi)]

    q = electron_charge
    m = electron_mass_reduced
    L = bohr_radius
    tau_alpha = 4 * m * (L ** 2) / hbar
    prefactor = -np.sqrt(pi) * (L ** 2) * ((q / hbar) ** 2)

    spec_kwargs = dict(
        time_initial = -20 * pulse_width * asec, time_final = 20 * pulse_width * asec,
        time_step = 1 * asec,
        error_on = 'da/dt', eps = 1e-3,
        minimum_time_step = .1 * asec,
        maximum_time_step = 5 * asec,
        prefactor = prefactor,
        kernel = ide.gaussian_kernel_LEN, kernel_kwargs = dict(tau_alpha = tau_alpha),
        evolution_gauge = 'LEN',
        evolution_method = 'ARK4',
        electric_potential_DC_correction = True,
    )

    specs = []
    for efield in efields:
        specs.append(ide.IntegroDifferentialEquationSpecification(efield.phase,
                                                                  electric_potential = efield,
                                                                  **spec_kwargs,
                                                                  ))
    results = si.utils.multi_map(run, specs, processes = 2)

    for result, cep, color, style in zip(results, (r'$\mathrm{CEP} = \varphi$', r'$\mathrm{CEP} = -\varphi$'), ('C0', 'C1'), ('-', '-')):
        ax_lower.plot(result.times, result.spec.electric_potential.get_electric_field_amplitude(result.times), color = color, linewidth = 1.5, label = cep, linestyle = style)
        ax_upper.plot(result.times, result.a2, color = color, linewidth = 1.5, label = cep, linestyle = style)

    efield_1 = results[0].spec.electric_potential.get_electric_field_amplitude(results[0].times)
    field_max = np.max(efield_1)
    field_min = np.min(efield_1)
    field_range = np.abs(field_max - field_min)

    ax_lower.set_xlabel(r'Time $t$')
    ax_lower.set_ylabel(r'$   \mathcal{E}(t)   $')
    ax_lower.yaxis.set_label_coords(-.125, .5)

    ax_upper.set_ylabel(r'$ \left| a(t) \right|^2 $')

    ax_lower.set_xticks([i * pulse_width * asec for i in range(-10, 11)])
    # ax_lower.set_xticklabels([r'$0$'] + [r'$ {} \tau $'.format(i) for i in range(-10, 11) if i != 0])
    x_tick_labels = []
    for i in range(-10, 11):
        if i == 0:
            x_tick_labels.append(r'$0$')
        elif i == 1:
            x_tick_labels.append(r'$\tau$')
        elif i == -1:
            x_tick_labels.append(r'$-\tau$')
        else:
            x_tick_labels.append(r'$ {} \tau $'.format(i))
    ax_lower.set_xticklabels(x_tick_labels)

    ax_lower.set_yticks([0, field_max, field_max / np.sqrt(2), -field_max, -field_max / np.sqrt(2)])
    ax_lower.set_yticklabels([
        r'$0$',
        r'$\mathcal{E}_{t} $',
        r'$\mathcal{E}_{t} / \sqrt{2} $',
        r'$-\mathcal{E}_{t} $',
        r'$-\mathcal{E}_{t} / \sqrt{2} $',
    ])

    ax_upper.tick_params(labelright = True)
    ax_lower.tick_params(labelright = True)
    ax_upper.xaxis.tick_top()

    ax_lower.grid(True, **grid_kwargs)
    ax_upper.grid(True, **grid_kwargs)

    ax_lower.set_xlim(-plot_bound * pulse_width * asec, plot_bound * pulse_width * asec)
    ax_lower.set_ylim(field_min - (.125 * field_range), field_max + (.125 * field_range))
    ax_upper.set_ylim(0, 1)

    ax_upper.legend(loc = 'best')

    save_figure(get_func_name() + '_phase={}'.format(round(phase, 3)))


def tunneling_ionization(amplitude):
    z = np.linspace(-5, 50, 1000) * bohr_radius

    coul_pot = -coulomb_constant * (proton_charge ** 2) / np.abs(z)
    elec_pot = -proton_charge * amplitude * z

    for img_format in ('pdf', 'png', 'pgf'):
        fm = si.vis.xy_plot(
            get_func_name() + f'__amp={uround(amplitude, atomic_electric_field, 5)}aef',
            z,
            coul_pot + elec_pot,
            coul_pot,
            elec_pot,
            line_labels = [r'$ V_{\mathrm{Coul}} + V_{\mathrm{Field}} $', r'$ V_{\mathrm{Coul}} $', r'$ V_{\mathrm{Field}} $'],
            line_kwargs = [None, {'linestyle': '--'}, {'linestyle': '--'}],
            hlines = [ion.HydrogenBoundState(1, 0).energy], hline_kwargs = [{'linestyle': ':', 'color': 'black'}],
            y_lower_limit = -2 * hartree,
            y_upper_limit = 0 * hartree,
            y_unit = 'eV',
            y_label = '$ V(z) $',
            x_unit = 'bohr_radius',
            x_label = r'$ z $',
            img_format = img_format,
            target_dir = OUT_DIR,
            close_after_exit = False,
            save = False,
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


def photons_at_a_glance():
    energies = np.linspace(0.1, 20, 1e4) * eV
    frequencies = energies / h
    wavelengths = c / frequencies

    data = [energies, frequencies, wavelengths]
    names = ['Energy', 'Frequency', 'Wavelength']
    axis_labels = [r'Energy $E$', r'Frequency $f$', r'Wavelength $\lambda$']
    units = ['eV', 'THz', 'nm']

    for kwargs in PLOT_KWARGS_LIST:
        for (x_data, x_name, x_label, x_unit), (y_data, y_name, y_label, y_unit) in itertools.combinations(zip(data, names, axis_labels, units), 2):
            print(x_name, y_name)
            si.vis.xy_plot(
                f'photons_at_a_glance__{x_name.lower()}_vs_{y_name.lower()}',
                x_data,
                y_data,
                x_label = x_label, y_label = y_label,
                x_unit = x_unit, y_unit = y_unit,
                title = f'Photon {x_name} vs. {y_name}',
                x_lower_limit = 0, y_lower_limit = 0,
                **kwargs
            )


def sine_squared_pulse(phase = 0, number_of_cycles = 1):
    pulse = ion.CosSquaredPulse.from_period(
        amplitude = 1 * atomic_electric_field,
        period = 200 * asec,
        number_of_cycles = number_of_cycles,
        phase = phase,
    )

    t_bound = .5 * number_of_cycles * pulse.period

    times = np.linspace(-t_bound, t_bound, 1e3)

    for plot_kwargs in PLOT_KWARGS_LIST:
        si.vis.xy_plot(
            f'sine_squared_pulse__amp={uround(pulse.amplitude, atomic_electric_field)}aef_N={number_of_cycles}_cep={uround(pulse.phase, pi)}pi',
            times,
            pulse.get_electric_field_envelope(times) * pulse.amplitude,
            pulse.get_electric_field_amplitude(times),
            line_labels = [
                'Envelope',
                'Pulse',
            ],
            line_kwargs = [
                {'linestyle': '--', 'color': 'black', 'alpha': 0.5},
                {'linestyle': '-', 'color': 'black', 'alpha': 1},
            ],
            x_unit = 'asec',
            x_label = r'Time $t$',
            y_unit = 'atomic_electric_field',
            y_label = rf'${ion.LATEX_EFIELD}(t)$',
            title = rf'$ T_c = 200 \, \mathrm{{as}}, \; N = {number_of_cycles}, \; \varphi = {uround(pulse.phase, pi)}\pi $',
            fig_scale = .475,
            **plot_kwargs,
        )


def relative_field_properties_vs_cep():
    pw = 200 * asec
    flu = 1 * Jcm2

    bound = 30
    times = np.linspace(-bound * pw, bound * pw, 1e5)
    print(f'dt = {uround(times[1] - times[0], asec)} as')
    len_times = len(times)

    ceps = np.linspace(0, pi, 5e2)
    field_fraction_vs_cep = np.zeros(len(ceps))
    power_fraction_vs_cep = np.zeros(len(ceps))
    rms_field_vs_cep = np.zeros(len(ceps))
    rms_power_vs_cep = np.zeros(len(ceps))
    avg_abs_field_vs_cep = np.zeros(len(ceps))
    avg_abs_power_vs_cep = np.zeros(len(ceps))

    pot_zero = ion.SincPulse.from_omega_carrier(pulse_width = pw, fluence = flu, phase = 0)

    field_zero = pot_zero.get_electric_field_amplitude(times)
    field_cut = np.max(np.abs(field_zero)) / 2
    power_cut = np.max(np.abs(field_zero)) / np.sqrt(2)

    for ii, cep in enumerate(tqdm(ceps)):
        pot = ion.SincPulse(pulse_width = pw, fluence = flu, phase = cep,
                            window = ion.SymmetricExponentialTimeWindow(
                                window_time = (bound - 2) * pw,
                                window_width = .2 * pw),
                            )
        # pot = ion.DC_correct_electric_potential(pot, times)

        field = pot.get_electric_field_amplitude(times)

        field_fraction_vs_cep[ii] = (np.abs(field) > field_cut).sum() / len_times
        power_fraction_vs_cep[ii] = (np.abs(field) > power_cut).sum() / len_times
        rms_field_vs_cep[ii] = np.std(field)
        rms_power_vs_cep[ii] = np.std(np.abs(field) ** 2)
        avg_abs_field_vs_cep[ii] = np.mean(np.abs(field))
        avg_abs_power_vs_cep[ii] = np.mean(np.abs(field) ** 2)

    for plot_kwargs in PLOT_KWARGS_LIST:
        si.vis.xy_plot(
            f'relative_field_properties_vs_cep',
            ceps,
            field_fraction_vs_cep / field_fraction_vs_cep[0],
            power_fraction_vs_cep / power_fraction_vs_cep[0],
            rms_field_vs_cep / rms_field_vs_cep[0],
            rms_power_vs_cep / rms_power_vs_cep[0],
            avg_abs_field_vs_cep / avg_abs_field_vs_cep[0],
            avg_abs_power_vs_cep / avg_abs_power_vs_cep[0],
            line_labels = [
                rf'$ {ion.LATEX_EFIELD} $ Fraction',
                rf'$ \left|{ion.LATEX_EFIELD}\right|^2 $ Fraction',
                rf'RMS $ {ion.LATEX_EFIELD} $',
                rf'RMS $ \left|{ion.LATEX_EFIELD}\right|^2 $',
                rf'AVG $ \left|{ion.LATEX_EFIELD}\right| $',
                rf'AVG $ \left|{ion.LATEX_EFIELD}\right|^2 $',
            ],
            x_label = r'Carrier-Envelope Phase $ \varphi $', x_unit = 'rad',
            title = fr'Relative $ {ion.LATEX_EFIELD} $ Properties for T = $ {bound}\tau $',
            legend_on_right = True,
            **plot_kwargs
        )


def sine_squared_pulse(phase = 0, number_of_cycles = 1):
    pulse = ion.CosSquaredPulse.from_period(
        amplitude = 1 * atomic_electric_field,
        period = 200 * asec,
        number_of_cycles = number_of_cycles,
        phase = phase,
    )

    t_bound = .5 * number_of_cycles * pulse.period

    times = np.linspace(-t_bound, t_bound, 1e3)

    for plot_kwargs in PLOT_KWARGS_LIST:
        si.vis.xy_plot(
            f'sine_squared_pulse__amp={uround(pulse.amplitude, atomic_electric_field)}aef_N={number_of_cycles}_cep={uround(pulse.phase, pi)}pi',
            times,
            pulse.get_electric_field_envelope(times) * pulse.amplitude,
            pulse.get_electric_field_amplitude(times),
            line_labels = [
                'Envelope',
                'Pulse',
            ],
            line_kwargs = [
                {'linestyle': '--', 'color': 'black', 'alpha': 0.5},
                {'linestyle': '-', 'color': 'black', 'alpha': 1},
            ],
            x_unit = 'asec',
            x_label = r'Time $t$',
            y_unit = 'atomic_electric_field',
            y_label = rf'${ion.LATEX_EFIELD}(t)$',
            title = rf'$ T_c = 200 \, \mathrm{{as}}, \; N = {number_of_cycles}, \; \varphi = {uround(pulse.phase, pi)}\pi $',
            fig_scale = .475,
            **plot_kwargs,
        )


if __name__ == '__main__':
    with log as logger:
        figures = [
            functools.partial(sine_squared_pulse, phase = 0, number_of_cycles = 1),
            functools.partial(sine_squared_pulse, phase = pi / 4, number_of_cycles = 1),
            functools.partial(sine_squared_pulse, phase = pi / 2, number_of_cycles = 1),
            functools.partial(sine_squared_pulse, phase = 0, number_of_cycles = 4),
            functools.partial(sine_squared_pulse, phase = pi / 4, number_of_cycles = 4),
            functools.partial(sine_squared_pulse, phase = pi / 2, number_of_cycles = 4),
            # photons_at_a_glance,
            # functools.partial(tunneling_ionization, (pi * epsilon_0 / (proton_charge ** 3)) * (rydberg ** 2)),
            # functools.partial(tunneling_ionization, 1 * atomic_electric_field),
            # functools.partial(tunneling_ionization, .5 * atomic_electric_field),
            # functools.partial(tunneling_ionization, .1 * atomic_electric_field),
            # functools.partial(tunneling_ionization, .05 * atomic_electric_field),
            # functools.partial(tunneling_ionization, .03 * atomic_electric_field),
            # functools.partial(tunneling_ionization, .025 * atomic_electric_field),
            # functools.partial(tunneling_ionization, .02 * atomic_electric_field),
            # functools.partial(tunneling_ionization, .01 * atomic_electric_field),
            # a_alpha_v2_kernel_gaussian,
            # finite_square_well,
            # finite_square_well_energies,
            # sinc_pulse_power_spectrum_full,
            # sinc_pulse_power_spectrum_half,
            # functools.partial(sinc_pulse_electric_field, phase = 0),
            # functools.partial(sinc_pulse_electric_field, phase = 1 / 4),
            # functools.partial(sinc_pulse_electric_field, phase = 1 / 2),
            # functools.partial(sinc_pulse_electric_field, phase = 1),
            # gaussian_pulse_power_spectrum_half,
            # functools.partial(ide_solution_sinc_pulse_cep_symmetry, phase = 1 / 4),
            # functools.partial(ide_solution_sinc_pulse_cep_symmetry, phase = 1 / 3),
            relative_field_properties_vs_cep,
            # photons_at_a_glance,
            # functools.partial(tunneling_ionization, (pi * epsilon_0 / (proton_charge ** 3)) * (rydberg ** 2)),
            # functools.partial(tunneling_ionization, 1 * atomic_electric_field),
            # functools.partial(tunneling_ionization, .5 * atomic_electric_field),
            # functools.partial(tunneling_ionization, .1 * atomic_electric_field),
            # functools.partial(tunneling_ionization, .05 * atomic_electric_field),
            # functools.partial(tunneling_ionization, .03 * atomic_electric_field),
            # functools.partial(tunneling_ionization, .025 * atomic_electric_field),
            # functools.partial(tunneling_ionization, .02 * atomic_electric_field),
            # functools.partial(tunneling_ionization, .01 * atomic_electric_field),
            # a_alpha_v2_kernel_gaussian,
            # finite_square_well,
            # finite_square_well_energies,
            # sinc_pulse_power_spectrum_full,
            # sinc_pulse_power_spectrum_half,
            # functools.partial(sinc_pulse_electric_field, phase = 0),
            # functools.partial(sinc_pulse_electric_field, phase = 1 / 4),
            # functools.partial(sinc_pulse_electric_field, phase = 1 / 2),
            # functools.partial(sinc_pulse_electric_field, phase = 1),
            # gaussian_pulse_power_spectrum_half,
            # functools.partial(ide_solution_sinc_pulse_cep_symmetry, phase = 1 / 4),
            # functools.partial(ide_solution_sinc_pulse_cep_symmetry, phase = 1 / 3),
            functools.partial(sine_squared_pulse, phase = 0, number_of_cycles = 1),
            functools.partial(sine_squared_pulse, phase = pi / 4, number_of_cycles = 1),
            functools.partial(sine_squared_pulse, phase = pi / 2, number_of_cycles = 1),
            functools.partial(sine_squared_pulse, phase = 0, number_of_cycles = 4),
            functools.partial(sine_squared_pulse, phase = pi / 4, number_of_cycles = 4),
            functools.partial(sine_squared_pulse, phase = pi / 2, number_of_cycles = 4),
            # photons_at_a_glance,
            # functools.partial(tunneling_ionization, (pi * epsilon_0 / (proton_charge ** 3)) * (rydberg ** 2)),
            # functools.partial(tunneling_ionization, 1 * atomic_electric_field),
            # functools.partial(tunneling_ionization, .5 * atomic_electric_field),
            # functools.partial(tunneling_ionization, .1 * atomic_electric_field),
            # functools.partial(tunneling_ionization, .05 * atomic_electric_field),
            # functools.partial(tunneling_ionization, .03 * atomic_electric_field),
            # functools.partial(tunneling_ionization, .025 * atomic_electric_field),
            # functools.partial(tunneling_ionization, .02 * atomic_electric_field),
            # functools.partial(tunneling_ionization, .01 * atomic_electric_field),
            # a_alpha_v2_kernel_gaussian,
            # finite_square_well,
            # finite_square_well_energies,
            # sinc_pulse_power_spectrum_full,
            # sinc_pulse_power_spectrum_half,
            # functools.partial(sinc_pulse_electric_field, phase = 0),
            # functools.partial(sinc_pulse_electric_field, phase = 1 / 4),
            # functools.partial(sinc_pulse_electric_field, phase = 1 / 2),
            # functools.partial(sinc_pulse_electric_field, phase = 1),
            # gaussian_pulse_power_spectrum_half,
            # functools.partial(ide_solution_sinc_pulse_cep_symmetry, phase = 1 / 4),
            # functools.partial(ide_solution_sinc_pulse_cep_symmetry, phase = 1 / 3),
        ]

        for fig in tqdm(figures):
            fig()
