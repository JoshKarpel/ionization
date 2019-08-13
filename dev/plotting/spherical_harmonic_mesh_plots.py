import logging
import os

import simulacra as si
import simulacra.units as u

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

LOGMAN = si.utils.LogManager("simulacra", "ionization", stdout_level=logging.DEBUG)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)

if __name__ == "__main__":
    with LOGMAN as logger:
        sim = mesh.SphericalHarmonicSpecification(
            "test",
            r_bound=20 * u.bohr_radius,
            r_points=500,
            l_bound=50,
            initial_state=states.HydrogenBoundState(2, 1),
            numeric_eigenstate_max_angular_momentum=3,
            time_initial=0,
            time_final=50 * u.asec,
            time_step=1 * u.asec,
        ).to_sim()
        sim.run()

        sim.mesh.plot.g(**PLOT_KWARGS)
        sim.mesh.plot.g2(**PLOT_KWARGS)
        sim.mesh.plot.psi(**PLOT_KWARGS)
        sim.mesh.plot.psi2(**PLOT_KWARGS)

        # sim.mesh.plot.g(name_postfix = '_pc', overlay_probability_current = True, **PLOT_KWARGS)
