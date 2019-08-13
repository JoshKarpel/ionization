import matplotlib

matplotlib.use("Agg")

import logging
import os

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
            x_center=-5 * u.nm,
            z_center=0 * u.nm,
            x_width=0.5 * u.nm,
            z_width=0.1 * u.nm,
        )
        scatterer += potentials.GaussianScatterer(
            x_center=0 * u.nm, z_center=5 * u.nm, x_width=0.1 * u.nm, z_width=0.1 * u.nm
        )
        scatterer += potentials.GaussianScatterer(
            x_center=0 * u.nm,
            z_center=-5 * u.nm,
            x_width=0.1 * u.nm,
            z_width=0.1 * u.nm,
        )
        scatterer += potentials.GaussianScatterer(
            x_center=2.5 * u.nm,
            z_center=0 * u.nm,
            x_width=0.1 * u.nm,
            z_width=0.1 * u.nm,
        )

        sim = mesh.RectangleSpecification(
            "test",
            z_bound=2 * 9 * u.nm,
            x_bound=2 * 16 * u.nm,
            z_points=9 * 2 * 50,
            x_points=16 * 2 * 50,
            initial_state=states.TwoDGaussianWavepacket(
                width_x=2 * u.nm,
                width_z=3 * u.nm,
                center_x=-15 * u.nm,
                k_x=1.5 * u.twopi / u.nm,
            ),
            time_initial=0,
            time_final=50 * u.fsec,
            time_step=u.fsec / (1200 / 50),
            internal_potential=scatterer,
            animators=[
                mesh.anim.SquareAnimator(
                    postfix="_g_flat",
                    axman_wavefunction=mesh.anim.RectangleMeshAxis(
                        which="g",
                        colormap=si.vis.RichardsonColormap(),
                        norm=si.vis.RichardsonNormalization(),
                        distance_unit="nm",
                        shading=si.vis.ColormapShader.FLAT,
                        show_grid=False,
                        axis_off=True,
                    ),
                    fig_width=16,
                    fig_height=9,
                    fullscreen=True,
                    **ANIM_KWARGS,
                ),
                mesh.anim.SquareAnimator(
                    postfix="_g_gouraud",
                    axman_wavefunction=mesh.anim.RectangleMeshAxis(
                        which="g",
                        colormap=si.vis.RichardsonColormap(),
                        norm=si.vis.RichardsonNormalization(),
                        distance_unit="nm",
                        shading=si.vis.ColormapShader.GOURAUD,
                        show_grid=False,
                        axis_off=True,
                    ),
                    fig_width=16,
                    fig_height=9,
                    fullscreen=True,
                    **ANIM_KWARGS,
                ),
            ],
        ).to_sim()

        sim.run(progress_bar=True)
