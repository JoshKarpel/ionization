import logging
import os

import simulacra as si
import simulacra.units as u

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

LOGMAN = si.utils.LogManager("simulacra", "ionization", stdout_level=logging.INFO)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)

if __name__ == "__main__":
    with si.utils.LogManager("simulacra", "ionization") as logger:
        sim = mesh.CylindricalSliceSpecification(
            "cylindrical",
            initial_state=states.ThreeDPlaneWave(0, 0, u.twopi / u.bohr_radius),
        ).to_sim()
        sim.mesh.plot.psi(**PLOT_KWARGS)

        sim = mesh.SphericalSliceSpecification(
            "spherical",
            initial_state=states.ThreeDPlaneWave(
                u.twopi / u.bohr_radius, 0, u.twopi / u.bohr_radius
            ),
        ).to_sim()
        sim.mesh.plot.psi(**PLOT_KWARGS)
