import os

import numpy as np

import simulacra as si
import simulacra.cluster as clu
from simulacra.units import *

import ionization as ion
import ionization.cluster as iclu

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)

if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization') as logger:
        sim = ion.SphericalHarmonicSpecification(
            'ev',
            r_bound = 100 * bohr_radius,
            r_points = 400,
            l_bound = 100,
            time_initial = -400 * asec,
            time_final = 400 * asec,
            use_numeric_eigenstates = True,
            numeric_eigenstate_max_energy = 10 * eV,
            numeric_eigenstate_max_angular_momentum = 5,
            electric_potential = ion.GaussianPulse(pulse_width = 100 * asec, fluence = 1 * Jcm2,
                                                   window = ion.SymmetricExponentialTimeWindow(window_time = 300 * asec, window_width = 20 * asec)),
            electric_potential_dc_correction = True,
        ).to_simulation()

        sim.run_simulation()

        print(sim.info())

        print('radial position', sim.radial_position_expectation_value_vs_time / bohr_radius)
        print('dipole moment', sim.electric_dipole_moment_expectation_value_vs_time / atomic_electric_dipole_moment)
        print('internal energy', sim.internal_energy_expectation_value_vs_time / eV)
        print('total energy', sim.total_energy_expectation_value_vs_time / eV)

        sim.plot_wavefunction_vs_time(show_vector_potential = False, **PLOT_KWARGS)
        sim.plot_radial_position_expectation_value_vs_time(**PLOT_KWARGS)
        sim.plot_dipole_moment_expectation_value_vs_time(**PLOT_KWARGS)
        sim.plot_energy_expectation_value_vs_time(**PLOT_KWARGS)



