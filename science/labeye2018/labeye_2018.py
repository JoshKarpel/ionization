#!/usr/bin/env python

import logging
import os

import numpy as np

import simulacra as si
import simulacra.units as u

import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

LOGMAN = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.INFO)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)

ANIM_KWARGS = dict(
    target_dir = OUT_DIR,
    fig_dpi_scale = 2,
    length = 10,
)

if __name__ == '__main__':
    with LOGMAN as logger:
        sim = ion.mesh.LineSpecification(
            'test',
            initial_state = ion.states.OneDSoftCoulombState(),
            z_bound = 250 * u.bohr_radius,
            z_points = 10000,
            time_step = 6.3e-3 * u.atomic_time,
            time_initial = 0,
            time_final = 100 * u.atomic_time,
            internal_potential = ion.potentials.SoftCoulombPotential(
                softening_distance = 1.1545 * u.bohr_radius,
            ),
            analytic_eigenstate_type = ion.states.OneDSoftCoulombState,
            use_numeric_eigenstates = True,
            number_of_numeric_eigenstates = 1,
            animators = [
                ion.mesh.anim.RectangleAnimator(
                    axman_wavefunction = ion.mesh.anim.LineMeshAxis(),
                    **ANIM_KWARGS,
                )
            ]
        ).to_sim()

        print(sim.info())
        sim.run(progress_bar = True)

        for analytic, numeric in sim.mesh.analytic_to_numeric.items():
            print(analytic, numeric)

        sim.plot.test_states_vs_time(**PLOT_KWARGS)
