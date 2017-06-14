import os
import logging

import numpy as np

import simulacra as si
import simulacra.cluster as clu
from simulacra.units import *

import ionization as ion


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

PLOT_KWARGS = dict(
        target_dir = OUT_DIR,
)


def get(potential, attr):
    for p in potential:
        if p.__class__ == ion.SineWave:
            return getattr(p, attr)


if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.DEBUG) as logger:
        jp = clu.JobProcessor.load('volkova__test.job')

        print(jp)

        intensities = sorted(set(get(result.electric_potential, 'intensity') for result in jp.data.values()))

        results_by_photon_energy = {}
        for photon_energy in sorted(set(get(result.electric_potential, 'photon_energy') for result in jp.data.values())):
            results_by_photon_energy[photon_energy] = sorted(jp.select_by_lambda(lambda result: get(result.electric_potential, 'photon_energy') == photon_energy),
                                                             key = lambda result: get(result.electric_potential, 'intensity'))

        si.vis.xy_plot(
                jp.name + '__ionization_probability_vs_intensity',
                intensities,
                *(np.array(list(1 - r.final_bound_state_overlap for r in results)) for results in results_by_photon_energy.values()),
                line_labels = (fr'$ \hbar \omega = {uround(photon_energy, eV, 1)} \, \mathrm{{eV}} $' for photon_energy in results_by_photon_energy.keys()),
                x_label = r'Intensity', x_unit = 'TWcm2', x_log_axis = True,
                y_label = r'Ionization Probability', y_log_axis = True,
                y_upper_limit = 1.0, y_log_pad = 1,
                **PLOT_KWARGS,
        )

        si.vis.xy_plot(
                jp.name + '__keldysh_parameter_vs_intensity',
                intensities,
                *(np.array(list(get(r.electric_potential, 'keldysh_parameter')(-rydberg) for r in results)) for results in results_by_photon_energy.values()),
                line_labels = (fr'$ \hbar \omega = {uround(photon_energy, eV, 1)} \, \mathrm{{eV}} $' for photon_energy in results_by_photon_energy.keys()),
                x_label = r'Intensity', x_unit = 'TWcm2', x_log_axis = True,
                y_label = r'$ \gamma $', y_log_axis = True,
                y_lower_limit = .1, y_upper_limit = 10, y_log_pad = 1,
                **PLOT_KWARGS,
        )