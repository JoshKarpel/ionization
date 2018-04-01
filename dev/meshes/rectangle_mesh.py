import logging
import os
import itertools

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
    fig_dpi_scale = 1,
)

if __name__ == '__main__':
    with LOGMAN as logger:
        spec = ion.mesh.RectangleSpecification(
            'test',
            z_bound = 10 * u.nm,
            x_bound = 10 * u.nm,
            initial_state = ion.states.TwoDPlaneWave(
                wavenumber_x = u.twopi / u.nm,
                wavenumber_z = 0,
            ),
            time_initial = 0 * u.asec,
            time_final = 200 * u.asec,
            animators = [
                ion.mesh.anim.RectangleAnimator(
                    postfix = '_g',
                    axman_wavefunction = ion.mesh.anim.RectangleMeshAxis(
                        which = 'g',
                        colormap = si.vis.RichardsonColormap(),
                    ),
                    length = 10,
                    **ANIM_KWARGS,
                ),
                ion.mesh.anim.RectangleAnimator(
                    postfix = '_g2',
                    axman_wavefunction = ion.mesh.anim.RectangleMeshAxis(
                        which = 'g2',
                    ),
                    length = 10,
                    **ANIM_KWARGS,
                ),
            ],
        )
        print(spec.info())

        print('\n' + '-' * 80 + '\n')

        sim = spec.to_sim()
        print(sim.info())

        sim.run(progress_bar = True)

        sim.mesh.plot.g(**PLOT_KWARGS)
        sim.mesh.plot.g2(**PLOT_KWARGS)
