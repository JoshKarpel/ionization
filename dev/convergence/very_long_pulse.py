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
    # img_format = 'png',
    # fig_dpi_scale = 3,
)

if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.INFO) as logger:
        pw = 1 * fsec
        dt = 1 * asec

        pulse = ion.SincPulse(pulse_width = pw, fluence = 1 * Jcm2,
                              window = ion.SymmetricExponentialTimeWindow(window_time = 30 * pw, window_width = .2 * pw))

        sim = ion.SphericalHarmonicSpecification(
            'very_long_pulse',
            r_bound = 200 * bohr_radius,
            r_points = 1000,
            l_bound = 400,
            mask = ion.RadialCosineMask(inner_radius = 80 * bohr_radius, outer_radius = 100 * bohr_radius),
            time_initial = -32 * pw, time_final = 32 * pw, time_step = dt,
            electric_potential = pulse,
            use_numeric_eigenstates = True,
            numeric_eigenstate_max_energy = 10 * eV,
            numeric_eigenstate_max_angular_momentum = 10,
            electric_potential_dc_correction = True,
            store_data_every = 20,
        ).to_simulation()

        sim.info().log()

        si.vis.xy_plot(
            'efield_vs_time',
            sim.times,
            sim.spec.electric_potential.get_electric_field_amplitude(sim.times),
            x_label = r'$t$', x_unit = 'asec',
            y_label = rf'${ion.LATEX_EFIELD}$', y_unit = 'atomic_electric_field',
            target_dir = OUT_DIR,
        )
        si.vis.xy_plot(
            'efield_vs_time__zoom',
            sim.times,
            sim.spec.electric_potential.get_electric_field_amplitude(sim.times),
            x_label = r'$t$', x_unit = 'asec',
            y_label = rf'${ion.LATEX_EFIELD}$', y_unit = 'atomic_electric_field',
            x_lower_limit = -5 * pw,
            x_upper_limit = 5 * pw,
            target_dir = OUT_DIR,
        )

        sim.run_simulation(progress_bar = True)
        sim.plot_wavefunction_vs_time(**PLOT_KWARGS)

        sim.info().log()
