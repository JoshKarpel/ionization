import logging
import os
import itertools

import numpy as np
import scipy as sp
from scipy import special

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
    fig_dpi_scale = 1,
    length = 20,
    fps = 30,
)

if __name__ == '__main__':
    with LOGMAN as logger:
        spec = ion.mesh.RectangleSpecification(
            'test',
            z_bound = 10 * u.nm,
            x_bound = 10 * u.nm,
            z_points = 500,
            x_points = 500,
            initial_state = ion.states.TwoDPlaneWave(
                wavenumber_x = np.sqrt(u.twopi) / u.nm,
                wavenumber_z = np.sqrt(u.twopi) / u.nm,
            ),
            time_initial = 0,
            time_final = 10 * u.fsec,
            time_step = u.fsec / 20,
            internal_potential = ion.potentials.NoPotentialEnergy(),
            animators = [
                ion.mesh.anim.SquareAnimator(
                    postfix = '_g',
                    axman_wavefunction = ion.mesh.anim.RectangleMeshAxis(
                        which = 'g',
                        colormap = si.vis.RichardsonColormap(),
                        norm = si.vis.RichardsonNormalization(),
                        distance_unit = 'nm',
                    ),
                    **ANIM_KWARGS,
                ),
                ion.mesh.anim.SquareAnimator(
                    postfix = '_g2',
                    axman_wavefunction = ion.mesh.anim.RectangleMeshAxis(
                        which = 'g2',
                        distance_unit = 'nm',
                    ),
                    **ANIM_KWARGS,
                ),
            ],
        )
        # print(spec.info())

        # print('\n' + '-' * 80 + '\n')

        sim = spec.to_sim()
        print(sim.info())

        sigma = 1 * u.nm
        center_x = 0
        center_z = 0

        x = sim.mesh.x_mesh - center_x
        z = sim.mesh.z_mesh - center_z
        gaussian = np.exp(-.25 * (((x ** 2) + (z ** 2)) / sigma ** 2))
        norm = 1 / (np.sqrt(u.twopi) * sigma)

        sim.mesh.g = norm * gaussian
        print('norm', sim.mesh.norm())

        sim.mesh.plot.g(**PLOT_KWARGS)
        sim.mesh.plot.g2(**PLOT_KWARGS)

        sim.run(progress_bar = True)

        # sim.mesh.plot.g(**PLOT_KWARGS)
        # sim.mesh.plot.g2(**PLOT_KWARGS)

        print(sim.data.norm)
