import logging
import os

from tqdm import tqdm
from mpmath import mpf

import numpy as np
import scipy.integrate as integ

import simulacra as si
from simulacra.units import *
import ionization as ion


# import matplotlib.pyplot as plt

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)
SIM_LIB = os.path.join(OUT_DIR, 'simlib')

logman = si.utils.LogManager('simulacra', 'ionization',
                             stdout_level = logging.INFO)

PLT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 3,
)


def run(spec):
    with logman as logger:
        # sim = spec.to_simulation()
        sim = si.utils.find_or_init_sim(spec, search_dir = SIM_LIB)

        logger.info(sim.info())

        sim.run_simulation()
        sim.save(target_dir = SIM_LIB)

        logger.info(sim.info())

        sim.plot_wavefunction_vs_time(target_dir = OUT_DIR)

        for state in sim.bound_states:
            print(state, state.energy / eV)

        return sim

if __name__ == '__main__':
    with logman as logger:
        photon_energy = .5 * eV
        intensities = np.array([1e14]) * W / (cm ** 2)

        window_mult = 2.5
        bound_mult = 4

        r_bound = 200
        l_bound = 400
        dt = 4.12e-3 * fsec

        spec_kwargs = dict(
            internal_potential = ion.SoftCoulomb(softening_distance = .0265 * angstrom),
            r_bound = r_bound * bohr_radius,
            r_points = r_bound * 4,
            evolution_gauge = 'LEN', l_bound = l_bound,
            time_step = dt,
            electric_potential_dc_correction = True,
            use_numeric_eigenstates = True,
            numeric_eigenstate_max_energy = 20 * eV,
            numeric_eigenstate_max_angular_momentum = 20,
            mask = ion.RadialCosineMask(inner_radius = (r_bound - 20) * bohr_radius, outer_radius = r_bound * bohr_radius),
            store_data_every = 50,
            checkpoints = True,
            checkpoint_dir = SIM_LIB,
        )

        specs = []
        for intensity in intensities:
            efield = ion.SineWave.from_photon_energy_and_intensity(photon_energy, intensity = intensity, phase = pi / 2)
            t_window = window_mult * efield.period
            t_bound = bound_mult * efield.period
            efield.window = window = ion.SymmetricExponentialTimeWindow(window_time = t_window, window_width = .2 * efield.period)
            name = f'sine_energy={uround(photon_energy, eV)}eV_int={uround(intensity, TW / (cm ** 2))}TWcm2'

            specs.append(ion.SphericalHarmonicSpecification(
                name,
                time_initial = -t_bound, time_final = t_bound,
                electric_potential = efield,
                **spec_kwargs
            ))

        si.utils.multi_map(run, specs)
