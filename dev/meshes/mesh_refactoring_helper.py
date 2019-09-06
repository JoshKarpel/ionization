#!/usr/bin/env python

import logging
import os

import numpy as np

import simulacra as si
import simulacra.units as u

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

LOGMAN = si.utils.LogManager("simulacra", "ionization", stdout_level=logging.INFO)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)


def run(spec):
    with LOGMAN as logger:
        sim = spec.to_sim()

        sim.run()

        return sim


if __name__ == "__main__":
    with LOGMAN as logger:
        dt = 1 * u.asec

        pw = 100 * u.asec
        flu = 1 * u.Jcm2
        cep = 0
        tb = 4
        three_d_gaussian_pulse = ion.potentials.GaussianPulse.from_number_of_cycles(
            pulse_width=pw, fluence=flu, phase=cep, number_of_cycles=3
        )

        energy_spacing = 0.1 * u.eV
        test_mass = u.electron_mass
        line_qho = ion.potentials.HarmonicOscillator.from_energy_spacing_and_mass(
            energy_spacing=energy_spacing, mass=test_mass
        )
        line_states = [
            states.QHOState.from_potential(line_qho, n=n, mass=test_mass)
            for n in range(5)
        ]
        sine_wave = ion.potentials.SineWave.from_photon_energy(
            photon_energy=energy_spacing, amplitude=0.0001 * u.atomic_electric_field
        )
        line_kwargs = dict(
            time_initial=0,
            time_final=1 * sine_wave.period,
            time_step=0.001 * sine_wave.period,
            internal_potential=line_qho,
            electric_potential=sine_wave,
            initial_state=line_states[0],
            test_states=line_states,
            z_bound=100 * u.nm,
            z_points=2 ** 12,
            store_data_every=-1,
        )

        three_d_spec_kwargs = dict(
            time_initial=-tb * three_d_gaussian_pulse.pulse_width,
            time_final=tb * three_d_gaussian_pulse.pulse_width,
            time_step=dt,
            electric_potential=three_d_gaussian_pulse,
            dc_correct_electric_potential=True,
            r_bound=50 * u.bohr_radius,
            r_points=500,
            l_bound=200,
            theta_points=360,
            rho_bound=50 * u.bohr_radius,
            rho_points=250,
            z_bound=50 * u.bohr_radius,
            z_points=500,
            store_data_every=-1,
        )

        spherical_harmonic_numeric_eigenstate_kwargs = dict(
            use_numeric_eigenstates=True,
            numeric_eigenstate_max_energy=20 * u.eV,
            numeric_eigenstate_max_angular_momentum=3,
        )

        specs = []

        qho = ion.potentials.HarmonicOscillator.from_energy_spacing_and_mass(
            energy_spacing, mass=test_mass
        )
        specs.append(
            mesh.RectangleSpecification(
                f"Rectangle_CN_LEN",
                operators=mesh.RectangleLengthGaugeOperators(),
                evolution_method=mesh.AlternatingDirectionImplicit(),
                z_bound=5 * u.nm,
                x_bound=5 * u.nm,
                z_points=500,
                x_points=500,
                initial_state=states.TwoDQuantumHarmonicOscillator.from_potential(
                    potential=qho, n_z=2, n_x=3, mass=u.electron_mass
                ),
                time_initial=0,
                time_final=2 * sine_wave.period,
                time_step=sine_wave.period / 200,
                internal_potential=qho,
                electric_potential=sine_wave,
            )
        )

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

        results = si.utils.multi_map(run, specs, processes=4)

        identifier_to_final_initial_overlap = {
            (
                r.mesh.__class__,
                r.spec.operators.__class__,
                r.spec.evolution_method.__class__,
            ): r.data.initial_state_overlap[-1]
            for r in results
        }

        ### look at results before comparison
        # for wavenumber, v in identifier_to_final_initial_overlap.items():
        #     print(wavenumber, v)

        expected = {
            (
                mesh.RectangleMesh,
                mesh.RectangleLengthGaugeOperators,
                mesh.AlternatingDirectionImplicit,
            ): 0.016739782924469107,
            (
                mesh.LineMesh,
                mesh.LineLengthGaugeOperators,
                mesh.AlternatingDirectionImplicit,
            ): 0.370010185740,
            (
                mesh.LineMesh,
                mesh.LineLengthGaugeOperators,
                mesh.SplitInteractionOperator,
            ): 0.370008474418,
            (
                mesh.LineMesh,
                mesh.LineVelocityGaugeOperators,
                mesh.SplitInteractionOperator,
            ): 0.370924310122,
            # (ion.mesh.LineMesh, 'HAM', ion.mesh.LineSpectral, ion.Gauge.LENGTH): 0.000568901854635,  # why is this not the same as the other line mesh methods?
            (
                mesh.CylindricalSliceMesh,
                mesh.CylindricalSliceLengthGaugeOperators,
                mesh.AlternatingDirectionImplicit,
            ): 0.293741923689,
            (
                mesh.SphericalSliceMesh,
                mesh.SphericalSliceLengthGaugeOperators,
                mesh.AlternatingDirectionImplicit,
            ): 0.178275457029,
            (
                mesh.SphericalHarmonicMesh,
                mesh.SphericalHarmonicLengthGaugeOperators,
                mesh.AlternatingDirectionImplicit,
            ): 0.312910470190,
            (
                mesh.SphericalHarmonicMesh,
                mesh.SphericalHarmonicLengthGaugeOperators,
                mesh.SplitInteractionOperator,
            ): 0.312928752359,
            (
                mesh.SphericalHarmonicMesh,
                mesh.SphericalHarmonicVelocityGaugeOperators,
                mesh.SplitInteractionOperator,
            ): 0.319513371899,
        }

        print()

        headers = ("Mesh", "Operators", "Evolution Method", "Expected", "Actual")
        rows = [
            (
                *(k.__name__ for k in key),
                f"{res:.6f}",
                f"{identifier_to_final_initial_overlap[key]:.6f}",
            )
            for key, res in expected.items()
        ]

        print(si.utils.table(headers, rows))

        for key, val in identifier_to_final_initial_overlap.items():
            assert np.allclose(val, expected[key])

        print("\nAll good!")
