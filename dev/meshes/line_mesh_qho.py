import logging
import os

import simulacra as si
import simulacra.units as u

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

LOGMAN = si.utils.LogManager("simulacra", "ionization", stdout_level=logging.INFO)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)

if __name__ == "__main__":
    with LOGMAN as logger:
        energy_spacing = 0.1 * u.eV
        test_mass = u.electron_mass

        qho = potentials.HarmonicOscillator.from_energy_spacing_and_mass(
            energy_spacing=energy_spacing, mass=test_mass
        )
        states = [
            states.QHOState.from_potential(qho, n=n, mass=test_mass) for n in range(30)
        ]

        efield = potentials.SineWave.from_photon_energy(
            photon_energy=energy_spacing, amplitude=0.0001 * u.atomic_electric_field
        )

        sim = mesh.LineSpecification(
            "qho",
            internal_potential=qho,
            electric_potential=efield,
            time_initial=0,
            time_final=5 * efield.period,
            time_step=0.001 * efield.period,
            initial_state=states[0],
            test_states=states,
            z_bound=100 * u.nm,
            z_points=2 ** 12,
            evolution_method=mesh.AlternatingDirectionImplicit(),
            # evolution_method = ion.mesh.SplitInteractionOperator(),
            animators=[
                mesh.anim.RectangleSplitLowerAnimator(
                    axman_wavefunction=mesh.anim.LineMeshAxis(),
                    fig_dpi_scale=2,
                    length=10,
                    target_dir=OUT_DIR,
                )
            ],
        ).to_sim()

        sim.run(progress_bar=True)
        print(sim.info())

        sim.plot_state_overlaps_vs_time(**PLOT_KWARGS)
