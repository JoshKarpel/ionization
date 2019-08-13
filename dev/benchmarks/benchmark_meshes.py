#!/usr/bin/env python

import logging
import os

import simulacra as si
import simulacra.units as u

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

LOGMAN = si.utils.LogManager("simulacra", "ionization", stdout_level=logging.INFO)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)


def run_with_timing(spec):
    with si.utils.BlockTimer() as timer:
        sim = spec.to_sim()
        sim.run(progress_bar=True)

    return sim, timer


def spacetime_points(sim):
    return sim.time_steps * sim.mesh.mesh_points


def report(results):
    headers = ("Mesh", "Operators", "Evolution Method", "Spacetime pts/s")

    meshes = [sim.mesh.__class__.__name__.replace("Mesh", "") for sim in results]
    operators = [
        sim.mesh.operators.__class__.__name__.replace("Operators", "")
        for sim in results
    ]
    methods = [sim.spec.evolution_method.__class__.__name__ for sim in results]
    pts_per_sec = [
        str(round(spacetime_points(sim) / timer.proc_time_elapsed))
        for sim, timer in results.items()
    ]
    rows = zip(meshes, operators, methods, pts_per_sec)

    return si.utils.table(headers, rows)


if __name__ == "__main__":
    with LOGMAN as logger:
        energy_spacing = 0.1 * u.eV
        test_mass = u.electron_mass
        line_qho = potentials.HarmonicOscillator.from_energy_spacing_and_mass(
            energy_spacing=energy_spacing, mass=test_mass
        )
        line_states = [
            states.QHOState.from_potential(line_qho, n=n, mass=test_mass)
            for n in range(5)
        ]
        line_sine_wave = potentials.SineWave.from_photon_energy(
            photon_energy=energy_spacing, amplitude=0.0001 * u.atomic_electric_field
        )
        line_kwargs = dict(
            time_initial=0,
            time_final=1 * line_sine_wave.period,
            time_step=0.001 * line_sine_wave.period,
            internal_potential=line_qho,
            initial_state=line_states[0],
            z_bound=100 * u.nm,
            z_points=2 ** 12,
            store_data_every=-1,
        )

        three_d_spec_kwargs = dict(
            time_initial=0,
            time_final=1 * u.fsec,
            time_step=1 * u.asec,
            r_bound=50 * u.bohr_radius,
            r_points=250,
            l_bound=100,
            theta_points=180,
            rho_bound=50 * u.bohr_radius,
            rho_points=100,
            z_bound=50 * u.bohr_radius,
            z_points=250,
            store_data_every=-1,
        )

        spherical_harmonic_numeric_eigenstate_kwargs = dict(
            use_numeric_eigenstates=False,
            numeric_eigenstate_max_energy=20 * u.eV,
            numeric_eigenstate_max_angular_momentum=3,
        )

        specs = []

        specs.append(
            mesh.LineSpecification(
                f"Line_CN_LEN",
                operators=mesh.LineLengthGaugeOperators(),
                evolution_method=mesh.AlternatingDirectionImplicit(),
                **line_kwargs,
            )
        )

        specs.append(
            mesh.LineSpecification(
                f"Line_SO_LEN",
                operators=mesh.LineLengthGaugeOperators(),
                evolution_method=mesh.SplitInteractionOperator(),
                **line_kwargs,
            )
        )

        specs.append(
            mesh.LineSpecification(
                f"Line_SO_VEL",
                operators=mesh.LineVelocityGaugeOperators(),
                evolution_method=mesh.SplitInteractionOperator(),
                **line_kwargs,
            )
        )

        specs.append(
            mesh.CylindricalSliceSpecification(
                f"CylindricalSlice_CN_LEN",
                evolution_equations="HAM",
                operators=mesh.CylindricalSliceLengthGaugeOperators(),
                evolution_method=mesh.AlternatingDirectionImplicit(),
                **three_d_spec_kwargs,
            )
        )

        specs.append(
            mesh.SphericalSliceSpecification(
                f"SphericalSlice_CN_LEN",
                operators=mesh.SphericalSliceLengthGaugeOperators(),
                evolution_method=mesh.AlternatingDirectionImplicit(),
                **three_d_spec_kwargs,
            )
        )

        specs.append(
            mesh.SphericalHarmonicSpecification(
                f"SphericalHarmonic_LAG_CN_LEN",
                operators=mesh.SphericalHarmonicLengthGaugeOperators(),
                evolution_method=mesh.AlternatingDirectionImplicit(),
                **spherical_harmonic_numeric_eigenstate_kwargs,
                **three_d_spec_kwargs,
            )
        )

        specs.append(
            mesh.SphericalHarmonicSpecification(
                f"SphericalHarmonic_LAG_SO_LEN",
                operators=mesh.SphericalHarmonicLengthGaugeOperators(),
                evolution_method=mesh.SplitInteractionOperator(),
                **spherical_harmonic_numeric_eigenstate_kwargs,
                **three_d_spec_kwargs,
            )
        )

        specs.append(
            mesh.SphericalHarmonicSpecification(
                f"SphericalHarmonic_LAG_SO_VEL",
                operators=mesh.SphericalHarmonicVelocityGaugeOperators(),
                evolution_method=mesh.SplitInteractionOperator(),
                **spherical_harmonic_numeric_eigenstate_kwargs,
                **three_d_spec_kwargs,
            )
        )

        results = dict(run_with_timing(spec) for spec in specs)

        rep = report(results)
        print("\n" + rep + "\n")
