import logging
import os

import numpy as np

import simulacra as si
import simulacra.units as u

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

LOGMAN = si.utils.LogManager("simulacra", "ionization", stdout_level=logging.INFO)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)

ANIM_KWARGS = dict(target_dir=OUT_DIR, fig_dpi_scale=1, length=20, fps=30)

if __name__ == "__main__":
    with LOGMAN as logger:
        scatterer = potentials.GaussianScatterer(
            x_center=-5 * u.nm, x_width=0.2 * u.nm, z_width=0.2 * u.nm
        )

        spec = mesh.RectangleSpecification(
            "test",
            z_bound=20 * u.nm,
            x_bound=20 * u.nm,
            z_points=1000,
            x_points=1000,
            initial_state=states.TwoDPlaneWave(
                wavenumber_x=np.sqrt(u.twopi) / u.nm,
                wavenumber_z=np.sqrt(u.twopi) / u.nm,
            ),
            time_initial=0,
            time_final=30 * u.fsec,
            time_step=u.fsec / 10,
            internal_potential=scatterer,
            animators=[
                mesh.anim.SquareAnimator(
                    postfix="_g",
                    axman_wavefunction=mesh.anim.RectangleMeshAxis(
                        which="g",
                        colormap=si.vis.RichardsonColormap(),
                        norm=si.vis.RichardsonNormalization(),
                        distance_unit="nm",
                    ),
                    **ANIM_KWARGS,
                ),
                mesh.anim.SquareAnimator(
                    postfix="_g2",
                    axman_wavefunction=mesh.anim.RectangleMeshAxis(
                        which="g2", distance_unit="nm"
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
        center_x = -15 * u.nm
        center_z = 0
        k = u.twopi / u.nm

        x = sim.mesh.x_mesh - center_x
        z = sim.mesh.z_mesh - center_z
        gaussian = np.exp(-0.25 * (((x ** 2) + (z ** 2)) / sigma ** 2))
        norm = 1 / (np.sqrt(u.twopi) * sigma)
        motion = np.exp(1j * k * x)

        sim.mesh.g = norm * gaussian * motion
        print("norm", sim.mesh.norm())

        sim.mesh.plot.g(**PLOT_KWARGS)
        sim.mesh.plot.g2(**PLOT_KWARGS)

        sim.mesh.plot.plot_mesh(
            scatterer(z=sim.mesh.z_mesh, x=sim.mesh.x_mesh),
            name="scatterer",
            **PLOT_KWARGS,
        )

        sim.run(progress_bar=True)

        # sim.mesh.plot.g(**PLOT_KWARGS)
        # sim.mesh.plot.g2(**PLOT_KWARGS)

        print(sim.data.norm)
