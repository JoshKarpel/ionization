import logging
import os
import functools
import itertools
import collections

from tqdm import tqdm

import numpy as np
import scipy.integrate as integrate

import simulacra as si
from simulacra.units import *

import ionization as ion
import ionization.integrodiff as ide

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)
SIM_LIB = os.path.join(OUT_DIR, 'simlib')

LOGMAN = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.DEBUG)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)


def run(spec):
    with LOGMAN as logger:
        # sim = spec.to_simulation()
        sim = si.utils.find_or_init_sim(spec, search_dir = SIM_LIB)

        logger.info(sim.info())
        if not sim.status == si.STATUS_FIN:
            sim.run_simulation()
            logger.info(sim.info())

        sim.save(target_dir = SIM_LIB)

        sim.plot_wavefunction_vs_time(
            show_vector_potential = False,
            **PLOT_KWARGS,
        )

        return (spec.test_energy, spec.phase), sim


if __name__ == '__main__':
    with LOGMAN as logger:
        gauges = ['LEN']
        tb = 10

        pulse_width = 93 * asec
        fluence = 5 * Jcm2
        phases = [0, pi / 2]

        test_width = 1 * bohr_radius
        test_charge = 1 * electron_charge
        test_mass = 1 * electron_mass
        test_energies = np.array([0, -3.4, -13.6]) * eV

        specs = []
        for gauge in gauges:
            for test_energy, phase in itertools.product(test_energies, phases):
                pulse = ion.SincPulse(pulse_width = pulse_width, fluence = fluence, phase = phase)

                t_bound = tb * pulse_width

                prefactor = ide.gaussian_prefactor_LEN(test_width, test_charge)
                tau_alpha = ide.gaussian_tau_alpha_LEN(test_width, test_mass)
                kernel = functools.partial(ide.gaussian_kernel_LEN, tau_alpha = tau_alpha)

                specs.append(ide.IntegroDifferentialEquationSpecification(
                    f'ide_{gauge}_tb={tb}_E={uround(test_energy, eV)}_pw={uround(pulse_width, asec)}as_flu={uround(fluence, Jcm2)}jcm2_cep={uround(phase, pi)}',
                    test_width = test_width,
                    test_charge = test_charge,
                    test_mass = test_mass,
                    test_energy = test_energy,
                    time_initial = -t_bound,
                    time_final = t_bound,
                    time_step = 1 * asec,
                    electric_potential = pulse,
                    electric_potential_dc_correction = True,
                    phase = phase,
                    prefactor = prefactor,
                    kernel = ide.gaussian_kernel_LEN,
                    kernel_kwargs = {'tau_alpha': tau_alpha},
                    evolution_gauge = gauge,
                    evolution_method = 'RK4',
                    store_data_every = 1,
                ))

        results = si.utils.multi_map(run, specs, processes = 2)
        energy_cep_to_a2 = dict((k, v.a2) for k, v in results)
        times = results[0][1].data_times

        colors = list(f'C{c}' for c in range(10))
        styles = ['-', '--', ':']
        energy_to_color = dict(zip(test_energies, colors))
        phase_to_style = dict(zip(phases, styles))

        line_labels = [
            fr'$E_{{\alpha}} = {uround(energy, eV)} \mathrm{{eV}}, \; \varphi = {uround(phase, pi)}\pi$'
            for energy, phase in energy_cep_to_a2.keys()
        ]
        line_kwargs = [
            {'color': energy_to_color[energy], 'linestyle': phase_to_style[phase]}
            for energy, phase in energy_cep_to_a2.keys()
        ]

        si.vis.xy_plot(
            f'comparison_pw={uround(pulse_width, asec)}as_flu={uround(fluence, Jcm2)}jcm2',
            times,
            *energy_cep_to_a2.values(),
            line_labels = line_labels,
            line_kwargs = line_kwargs,
            x_label = r'Time $t$', x_unit = 'asec',
            y_label = r'$ \left| a_{\alpha}(t) \right|^2 $',
            legend_on_right = True,
            **PLOT_KWARGS,
        )