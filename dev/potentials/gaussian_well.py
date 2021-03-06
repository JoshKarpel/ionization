import os

import numpy as np
import simulacra as si
from simulacra.units import *

import ionization as ion


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=5)

if __name__ == "__main__":
    with si.utils.LogManager("simulacra", "ionization") as logger:
        x = np.linspace(-25 * bohr_radius, 25 * bohr_radius, 1000)

        gaussian_well = ion.GaussianPotential(
            potential_extrema=-10 * eV, width=5 * bohr_radius
        )

        variational_ground_state = ion.GaussianWellState.from_potential(
            gaussian_well, electron_mass
        )
        print("width", variational_ground_state.width / bohr_radius)
        print("energy", variational_ground_state.energy / eV)

        si.vis.xy_plot(
            "potential",
            x,
            gaussian_well(distance=x),
            x_label=r"$x$",
            x_unit="bohr_radius",
            y_label=r"$V(x)$",
            y_unit="eV",
            **PLOT_KWARGS,
        )

        sim = ion.LineSpecification(
            "gaussian_well",
            x_bound=100 * bohr_radius,
            x_points=2 ** 10,
            test_mass=electron_mass,
            internal_potential=gaussian_well,
            initial_state=variational_ground_state,
            use_numeric_eigenstates=True,
            analytic_eigenstate_type=ion.GaussianWellState,
            time_initial=0,
            time_final=1 * fsec,
            animators=[
                animation.animators.RectangleAnimator(
                    axman_wavefunction=animation.animators.LineMeshAxis(),
                    target_dir=OUT_DIR,
                )
            ],
        ).to_sim()

        sim.info().log()

        numeric_ground_state = sim.spec.test_states[0]

        si.vis.xy_plot(
            "ground_state_comparison",
            sim.mesh.x_mesh,
            variational_ground_state(sim.mesh.x_mesh),
            np.abs(numeric_ground_state(sim.mesh.x_mesh)),
            line_labels=["Variational", "Numeric"],
            x_label=r"$x$",
            x_unit="bohr_radius",
            y_label=r"$\psi_0(x)$",
            x_lower_limit=-25 * bohr_radius,
            x_upper_limit=25 * bohr_radius,
            **PLOT_KWARGS,
        )

        for state in [variational_ground_state, numeric_ground_state]:
            print(f"{state}: E = {state.energy / eV:.3f} eV")

        print(
            f"Variational ground state width = {variational_ground_state.width / bohr_radius:.3f} a_0"
        )

        sim.run()

        sim.info().log()
