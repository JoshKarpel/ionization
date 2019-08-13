import logging
import os

import simulacra as si
import simulacra.units as u

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

LOGMAN = si.utils.LogManager("simulacra", "ionization", stdout_level=logging.INFO)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)

ANIM_KWARGS = dict(target_dir=OUT_DIR, fig_dpi_scale=1, length=30, fps=30)

if __name__ == "__main__":
    with LOGMAN as logger:
        well = potentials.InfiniteEllipticalWell(z_axis=5 * u.nm, x_axis=10 * u.nm)

        spec = mesh.RectangleSpecification(
            "test",
            z_bound=10.1 * u.nm,
            x_bound=10.1 * u.nm,
            z_points=1000,
            x_points=1000,
            initial_state=states.TwoDGaussianWavepacket(
                width_x=1 * u.nm,
                width_z=1 * u.nm,
                k_x=1 * u.twopi / u.nm,
                k_z=1 * u.twopi / u.nm,
            ),
            time_initial=0,
            time_final=100 * u.fsec,
            time_step=u.fsec / 20,
            internal_potential=well,
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
        sim.run(progress_bar=True)

        # sim.mesh.plot.g(**PLOT_KWARGS)
        # sim.mesh.plot.g2(**PLOT_KWARGS)

        print(sim.data.norm)
