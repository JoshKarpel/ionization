import logging
import os

import numpy as np

import simulacra as si
from simulacra.units import *

import ionization as ion
import ionization.ide as ide

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

LOGMAN = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.DEBUG)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)

if __name__ == '__main__':
    with LOGMAN as logger:
        time_bound = 3.5
        pulse = ion.GaussianPulse.from_number_of_cycles(
            pulse_width = 200 * asec,
            fluence = 5 * Jcm2,
            phase = pi / 2,
            number_of_cycles = 10,
        )

        shared_kwargs = dict(
            electric_potential = pulse,
            time_initial = -time_bound * pulse.pulse_width,
            time_final = time_bound * pulse.pulse_width,
            # electric_potential_dc_correction = True,
        )

        tdse_spec = ion.SphericalHarmonicSpecification(
            'tdse',
            r_bound = 100 * bohr_radius,
            r_points = 500,
            l_bound = 100,
            time_step = 1 * asec,
            use_numeric_eigenstates = True,
            numeric_eigenstate_max_energy = 20 * eV,
            numeric_eigenstate_max_angular_momentum = 20,
            mask = ion.RadialCosineMask(inner_radius = 75 * bohr_radius, outer_radius = 100 * bohr_radius),
            **shared_kwargs,
        )

        ide_spec = ide.IntegroDifferentialEquationSpecification(
            'ide',
            time_step = .5 * asec,
            **shared_kwargs,
        )

        sims = [spec.to_simulation() for spec in (
            # tdse_spec,
            ide_spec,
        )]
        for sim in sims:
            sim.run_simulation()
            sim.plot_wavefunction_vs_time(**PLOT_KWARGS)
