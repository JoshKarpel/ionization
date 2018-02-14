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
        pulse = ion.potentials.GaussianPulse.from_number_of_cycles(
            pulse_width = pw,
            fluence = flu,
            phase = cep,
            number_of_cycles = 3,
        )

        shared_spec_kwargs = dict(
            time_initial = -tb * pulse.pulse_width,
            time_final = tb * pulse.pulse_width,
            time_step = dt,
            electric_potential = pulse,
            dc_correct_electric_potential = True,
            r_bound = 50 * u.bohr_radius,
            r_points = 500,
            l_bound = 200,
            theta_points = 360,
            rho_bound = 50 * u.bohr_radius,
            rho_points = 250,
            z_bound = 50 * u.bohr_radius,
            z_points = 500,
            x_bound = 100 * u.bohr_radius,
            x_points = 10000,
            store_data_every = -1,
        )

        spherical_harmonic_numeric_eigenstate_kwargs = dict(
            use_numeric_eigenstates = True,
            numeric_eigenstate_max_energy = 20 * u.eV,
            numeric_eigenstate_max_angular_momentum = 3,
        )

        specs = []
        for evolution_method in (
                ion.mesh.LineCrankNicolson(),
                ion.mesh.LineSplitOperator(),
                # ion.mesh.LineSpectral(),  # something is wrong with spectral
        ):
            specs.append(
                ion.mesh.LineSpecification(
                    f'Line_HAM_{evolution_method}_LEN',
                    evolution_equations = 'HAM',
                    evolution_method = evolution_method,
                    evolution_gauge = 'LEN',
                    **shared_spec_kwargs
                )
            )
        specs.append(
            ion.mesh.CylindricalSliceSpecification(
                f'CylindricalSlice_HAM_CN_LEN',
                evolution_equations = 'HAM',
                evolution_method = ion.mesh.CylindricalSliceCrankNicolson(),
                evolution_gauge = 'LEN',
                **shared_spec_kwargs
            )
        )
        specs.append(
            ion.mesh.SphericalSliceSpecification(
                f'SphericalSlice_HAM_CN_LEN',
                evolution_equations = 'HAM',
                evolution_method = ion.mesh.SphericalSliceCrankNicolson(),
                evolution_gauge = 'LEN',
                **shared_spec_kwargs
            )
        )
        specs.append(
            ion.mesh.SphericalHarmonicSpecification(
                f'SphericalHarmonic_LAG_CN_LEN',
                evolution_equations = 'LAG',
                evolution_method = ion.mesh.SphericalHarmonicCrankNicolson(),
                evolution_gauge = 'LEN',
                **spherical_harmonic_numeric_eigenstate_kwargs,
                **shared_spec_kwargs
            )
        )
        for evolution_gauge in ('LEN', 'VEL'):
            specs.append(
                ion.mesh.SphericalHarmonicSpecification(
                    f'SphericalHarmonic_LAG_SO_{evolution_gauge}',
                    evolution_equations = 'LAG',
                    evolution_method = ion.mesh.SphericalHarmonicSplitOperator(),
                    evolution_gauge = evolution_gauge,
                    **spherical_harmonic_numeric_eigenstate_kwargs,
                    **shared_spec_kwargs
                )
            )

        results = si.utils.multi_map(run, specs, processes = 2)

        identifier_to_final_initial_overlap = {(r.mesh.__class__, r.spec.evolution_equations, r.spec.evolution_method.__class__, r.spec.evolution_gauge): r.data.initial_state_overlap[-1] for r in results}

        ### look at results before comparison
        # for k, v in identifier_to_final_initial_overlap.items():
        #     print(k, v)

        expected_results = {
            (ion.mesh.LineMesh, 'HAM', ion.mesh.LineCrankNicolson, 'LEN'): 0.0143651217635,
            (ion.mesh.LineMesh, 'HAM', ion.mesh.LineSplitOperator, 'LEN'): 0.0143755731217,
            # (ion.mesh.LineMesh, 'HAM', ion.mesh.LineSpectral, 'LEN'): 0.000568901854635,  # why is this not the same as the other line mesh methods?
            (ion.mesh.CylindricalSliceMesh, 'HAM', ion.mesh.CylindricalSliceCrankNicolson, 'LEN'): 0.293741923689,
            (ion.mesh.SphericalSliceMesh, 'HAM', ion.mesh.SphericalSliceCrankNicolson, 'LEN'): 0.178275457029,
            (ion.mesh.SphericalHarmonicMesh, 'LAG', ion.mesh.SphericalHarmonicCrankNicolson, 'LEN'): 0.312970628484,
            (ion.mesh.SphericalHarmonicMesh, 'LAG', ion.mesh.SphericalHarmonicSplitOperator, 'LEN'): 0.312928752359,
            (ion.mesh.SphericalHarmonicMesh, 'LAG', ion.mesh.SphericalHarmonicSplitOperator, 'VEL'): 0.319513371899,
        }

        summary = 'Results:\n'
        lines = []
        for identifier, latest_result in identifier_to_final_initial_overlap.items():
            s = ", ".join((identifier[0].__name__, identifier[1], identifier[2].__name__, identifier[3]))
            s += f': {latest_result:.6f} | {expected_results[identifier]:.6f}'

            lines.append(s)
        summary += '\n'.join(lines)
        print(summary)

        for key, val in identifier_to_final_initial_overlap.items():
            np.testing.assert_allclose(val, expected_results[key])
