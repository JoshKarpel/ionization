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
        z_width = 0.2 * u.nm
        x_width = 0.2 * u.nm

        z_spacing = 3 * u.nm
        x_spacing = 3 * u.nm
        z_offset = 1.5 * u.nm
        scatterers = []
        x = 0 * u.nm
        offset = False
        while x < 20 * u.nm:
            z = -25 * u.nm
            if offset:
                z += z_offset
            offset = not offset
            while z < 25 * u.nm:
                s = potentials.GaussianScatterer(
                    z_width=z_width, x_width=x_width, z_center=z, x_center=x
                )
                print(s)
                scatterers.append(
                    potentials.GaussianScatterer(
                        z_width=z_width, x_width=x_width, z_center=z, x_center=x
                    )
                )
                z += z_spacing

            x += x_spacing

        scatterer = potentials.PotentialEnergySum(*scatterers)
        print(scatterer)

        # scatterer = ion.potentials.GaussianScatterer(
        #     x_center = -5 * u.nm,
        #     x_width = .2 * u.nm,
        #     z_width = .2 * u.nm,
        # )
        # scatterer += ion.potentials.GaussianScatterer(
        #     x_center = -3 * u.nm,
        #     z_center = 3 * u.nm,
        #     x_width = .2 * u.nm,
        #     z_width = .2 * u.nm,
        # )

        spec = mesh.RectangleSpecification(
            "pinball",
            z_bound=20 * u.nm,
            x_bound=20 * u.nm,
            z_points=1000,
            x_points=1000,
            initial_state=states.TwoDGaussianWavepacket(
                center_x=-5 * u.nm,
                center_z=-5 * u.nm,
                k_x=2 * u.twopi / u.nm,
                k_z=2 * u.twopi / u.nm,
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
