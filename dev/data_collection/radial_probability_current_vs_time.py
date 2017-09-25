import logging
import os
import time

import numpy as np
import scipy.sparse as sparse

import simulacra as si
from simulacra.units import *
import ionization as ion

import matplotlib.pyplot as plt

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

LOGMAN = si.utils.LogManager('simulacra', 'ionization', stdout_logs = True, stdout_level = logging.DEBUG)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)


if __name__ == '__main__':
    with LOGMAN as logger:
        pulse = ion.Rectangle(start_time = 50 * asec, end_time = 100 * asec, amplitude = .5 * atomic_electric_field)
        pulse += ion.Rectangle(start_time = 150 * asec, end_time = 200 * asec, amplitude = -.5 * atomic_electric_field)

        sim = ion.SphericalHarmonicSpecification(
            'radial_current_test',
            r_bound = 50 * bohr_radius, r_points = 1000, l_bound = 100, theta_points = 180,
            time_initial = 0 * asec, time_final = 200 * 60 * asec, time_step = 1 * asec,
            electric_potential = pulse,
            use_numeric_eigenstates = True,
            numeric_eigenstate_max_energy = 10 * eV,
            numeric_eigenstate_max_angular_momentum = 10,
            mask = ion.RadialCosineMask(inner_radius = 40 * bohr_radius, outer_radius = 50 * bohr_radius),
            store_radial_probability_current = True,
        ).to_simulation()

        # sim.run_simulation()
        sim.radial_probability_current_vs_time__pos_z = np.ones_like(sim.radial_probability_current_vs_time__pos_z)
        sim.radial_probability_current_vs_time__neg_z = -np.ones_like(sim.radial_probability_current_vs_time__neg_z)
        sim.electric_field_amplitude_vs_time = np.ones_like(sim.electric_field_amplitude_vs_time)

        time.sleep(.5)
        sim.info().log()

        # ### MAKE PLOTS ###
        # # z_lim = .1 * per_asec
        r_lim = 10 * bohr_radius
        # # for which in ['sum', 'pos', 'neg']:
        # for which in ['sum']:
        #     print(which)
        #     sim.plot_radial_probability_current_vs_time(
        #         which = which,
        #         r_upper_limit = r_lim,
        #         **PLOT_KWARGS
        #     )

        sim.plot_radial_probability_current_vs_time__combined(
            r_upper_limit = r_lim,
            shading = 'flat',
            **PLOT_KWARGS,
        )
