#!/usr/bin/env python

import logging
import os

import numpy as np

import simulacra as si
import simulacra.units as u

import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

LOGMAN = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.INFO)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)


def run(spec):
    with LOGMAN as logger:
        sim = spec.to_sim()

        sim.run()

        return sim


if __name__ == '__main__':
    with LOGMAN as logger:
        dt = 1 * u.asec

        pw = 100 * u.asec
        flu = 1 * u.Jcm2
        cep = 0
        tb = 4
        three_d_gaussian_pulse = ion.potentials.GaussianPulse.from_number_of_cycles(
            pulse_width = pw,
            fluence = flu,
            phase = cep,
            number_of_cycles = 3,
        )

        energy_spacing = .1 * u.eV
        test_mass = u.electron_mass
        line_qho = ion.potentials.HarmonicOscillator.from_energy_spacing_and_mass(
            energy_spacing = energy_spacing,
            mass = test_mass,
        )
        line_states = [ion.states.QHOState.from_potential(line_qho, n = n, mass = test_mass) for n in range(5)]
        line_sine_wave = ion.potentials.SineWave.from_photon_energy(
            photon_energy = energy_spacing,
            amplitude = .0001 * u.atomic_electric_field,
        )
        line_kwargs = dict(
            time_initial = 0,
            time_final = 1 * line_sine_wave.period,
            time_step = .001 * line_sine_wave.period,
            internal_potential = line_qho,
            electric_potential = line_sine_wave,
            initial_state = line_states[0],
            test_states = line_states,
            z_bound = 100 * u.nm,
            z_points = 2 ** 12,
            store_data_every = -1,
        )

        three_d_spec_kwargs = dict(
            time_initial = -tb * three_d_gaussian_pulse.pulse_width,
            time_final = tb * three_d_gaussian_pulse.pulse_width,
            time_step = dt,
            electric_potential = three_d_gaussian_pulse,
            dc_correct_electric_potential = True,
            r_bound = 50 * u.bohr_radius,
            r_points = 500,
            l_bound = 200,
            theta_points = 360,
            rho_bound = 50 * u.bohr_radius,
            rho_points = 250,
            z_bound = 50 * u.bohr_radius,
            z_points = 500,
            store_data_every = -1,
        )

        spherical_harmonic_numeric_eigenstate_kwargs = dict(
            use_numeric_eigenstates = True,
            numeric_eigenstate_max_energy = 20 * u.eV,
            numeric_eigenstate_max_angular_momentum = 3,
        )

        specs = []

        specs.append(
            ion.mesh.LineSpecification(
                f'Line_CN_LEN',
                operators = ion.mesh.LineLengthGaugeOperators(),
                evolution_method = ion.mesh.AlternatingDirectionImplicit(),
                **line_kwargs
            )
        )

        specs.append(
            ion.mesh.LineSpecification(
                f'Line_SO_LEN',
                operators = ion.mesh.LineLengthGaugeOperators(),
                evolution_method = ion.mesh.SplitInteractionOperator(),
                **line_kwargs
            )
        )

        specs.append(
            ion.mesh.LineSpecification(
                f'Line_SO_VEL',
                operators = ion.mesh.LineVelocityGaugeOperators(),
                evolution_method = ion.mesh.SplitInteractionOperator(),
                **line_kwargs
            )
        )

        specs.append(
            ion.mesh.CylindricalSliceSpecification(
                f'CylindricalSlice_CN_LEN',
                evolution_equations = 'HAM',
                operators = ion.mesh.CylindricalSliceLengthGaugeOperators(),
                evolution_method = ion.mesh.AlternatingDirectionImplicit(),
                **three_d_spec_kwargs
            )
        )

        specs.append(
            ion.mesh.SphericalSliceSpecification(
                f'SphericalSlice_CN_LEN',
                operators = ion.mesh.SphericalSliceLengthGaugeOperators(),
                evolution_method = ion.mesh.AlternatingDirectionImplicit(),
                **three_d_spec_kwargs
            )
        )

        specs.append(
            ion.mesh.SphericalHarmonicSpecification(
                f'SphericalHarmonic_LAG_CN_LEN',
                operators = ion.mesh.SphericalHarmonicLengthGaugeOperators(),
                evolution_method = ion.mesh.AlternatingDirectionImplicit(),
                **spherical_harmonic_numeric_eigenstate_kwargs,
                **three_d_spec_kwargs
            )
        )

        specs.append(
            ion.mesh.SphericalHarmonicSpecification(
                f'SphericalHarmonic_LAG_SO_LEN',
                operators = ion.mesh.SphericalHarmonicLengthGaugeOperators(),
                evolution_method = ion.mesh.SplitInteractionOperator(),
                **spherical_harmonic_numeric_eigenstate_kwargs,
                **three_d_spec_kwargs
            )
        )

        specs.append(
            ion.mesh.SphericalHarmonicSpecification(
                f'SphericalHarmonic_LAG_SO_VEL',
                operators = ion.mesh.SphericalHarmonicVelocityGaugeOperators(),
                evolution_method = ion.mesh.SplitInteractionOperator(),
                **spherical_harmonic_numeric_eigenstate_kwargs,
                **three_d_spec_kwargs
            )
        )

        results = si.utils.multi_map(run, specs, processes = 4)

        identifier_to_final_initial_overlap = {(r.mesh.__class__, r.spec.operators.__class__, r.spec.evolution_method.__class__): r.data.initial_state_overlap[-1] for r in results}

        ### look at results before comparison
        # for k, v in identifier_to_final_initial_overlap.items():
        #     print(k, v)

        expected_results = {
            (ion.mesh.LineMesh, ion.mesh.LineLengthGaugeOperators, ion.mesh.AlternatingDirectionImplicit): 0.370010185740,
            (ion.mesh.LineMesh, ion.mesh.LineLengthGaugeOperators, ion.mesh.SplitInteractionOperator): 0.370008474418,
            (ion.mesh.LineMesh, ion.mesh.LineVelocityGaugeOperators, ion.mesh.SplitInteractionOperator): 0.370924310122,
            # (ion.mesh.LineMesh, 'HAM', ion.mesh.LineSpectral, ion.Gauge.LENGTH): 0.000568901854635,  # why is this not the same as the other line mesh methods?
            (ion.mesh.CylindricalSliceMesh, ion.mesh.CylindricalSliceLengthGaugeOperators, ion.mesh.AlternatingDirectionImplicit): 0.293741923689,
            (ion.mesh.SphericalSliceMesh, ion.mesh.SphericalSliceLengthGaugeOperators, ion.mesh.AlternatingDirectionImplicit): 0.178275457029,
            (ion.mesh.SphericalHarmonicMesh, ion.mesh.SphericalHarmonicLengthGaugeOperators, ion.mesh.AlternatingDirectionImplicit): 0.312910470190,
            (ion.mesh.SphericalHarmonicMesh, ion.mesh.SphericalHarmonicLengthGaugeOperators, ion.mesh.SplitInteractionOperator): 0.312928752359,
            (ion.mesh.SphericalHarmonicMesh, ion.mesh.SphericalHarmonicVelocityGaugeOperators, ion.mesh.SplitInteractionOperator): 0.319513371899,
        }

        print()

        headers = ('Mesh', 'Operators', 'Evolution Method', 'Expected', 'Actual')
        rows = [(*(k.__name__ for k in key), f'{res:.6f}', f'{identifier_to_final_initial_overlap[key]:.6f}') for key, res in expected_results.items()]

        print(si.utils.table(headers, rows))

        for key, val in identifier_to_final_initial_overlap.items():
            np.testing.assert_allclose(val, expected_results[key])

        print('\nAll good!')
