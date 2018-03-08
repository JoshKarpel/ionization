import logging
import os

import numpy as np

import simulacra as si
import simulacra.units as u

import ionization as ion

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
        qho = ion.potentials.HarmonicOscillator.from_energy_spacing_and_mass()
        sim = ion.mesh.LineSpecification(
            'test',
            z_bound = 100 * u.nm,
            z_points = 1000,
            initial_state = ion.states.QHOState.from_potential(qho, u.electron_mass),
            time_initial = 0,
            time_final = 100 * u.asec,
            time_step = 1 * u.asec
        ).to_sim()
        sim.run()

        sim.mesh.plot.g(**PLOT_KWARGS)
        sim.mesh.plot.g2(**PLOT_KWARGS)
        sim.mesh.plot.psi(**PLOT_KWARGS)
        sim.mesh.plot.psi2(**PLOT_KWARGS)

        # sim.mesh.plot.g(name_postfix = '_pc', overlay_probability_current = True, **PLOT_KWARGS)
