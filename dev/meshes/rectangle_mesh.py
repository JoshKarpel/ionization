import logging
import os

import numpy as np
import scipy as sp
from scipy import special

import simulacra as si
import simulacra.units as u

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

LOGMAN = si.utils.LogManager("simulacra", "ionization", stdout_level=logging.INFO)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)

ANIM_KWARGS = dict(target_dir=OUT_DIR, fig_dpi_scale=1, length=20, fps=30)

if __name__ == "__main__":
    with LOGMAN as logger:
        qho = potentials.HarmonicOscillator.from_energy_spacing_and_mass(1 * u.eV)
        qho_period = u.twopi / qho.omega(u.electron_mass)

        sine = potentials.SineWave.from_photon_energy(
            1 * u.eV, amplitude=0.001 * u.atomic_electric_field
        )

        spec = mesh.RectangleSpecification(
            "test",
            z_bound=5 * u.nm,
            x_bound=5 * u.nm,
            z_points=500,
            x_points=500,
            initial_state=states.TwoDPlaneWave(
                wavenumber_x=np.sqrt(u.twopi) / u.nm,
                wavenumber_z=np.sqrt(u.twopi) / u.nm,
            ),
            time_initial=0,
            time_final=5 * sine.period,
            time_step=qho_period / 200,
            internal_potential=qho,
            electric_potential=sine,
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

        n = 3
        m = 3

        energy = (m + n + 1) * u.hbar * qho.omega(u.electron_mass)

        ksi = np.sqrt(u.electron_mass * qho.omega(u.electron_mass) / u.hbar)

        norm = ksi / np.sqrt(
            (2 ** (n + m)) * sp.misc.factorial(n) * sp.misc.factorial(m) * u.pi
        )
        gaussian = np.exp(
            -(((ksi * sim.mesh.x_mesh) ** 2) + ((ksi * sim.mesh.z_mesh) ** 2)) / 2
        )
        hermite = special.hermite(n)(ksi * sim.mesh.x_mesh) * special.hermite(m)(
            ksi * sim.mesh.z_mesh
        )

        sim.mesh.g = norm * gaussian * hermite
        print("norm", sim.mesh.norm())
        print("energy", energy / u.eV)

        sim.mesh.plot.g(**PLOT_KWARGS)
        sim.mesh.plot.g2(**PLOT_KWARGS)

        sim.run(progress_bar=True)

        # sim.mesh.plot.g(**PLOT_KWARGS)
        # sim.mesh.plot.g2(**PLOT_KWARGS)

        print(sim.data.norm)
