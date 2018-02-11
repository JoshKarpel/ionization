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
        sim = spec.to_simulation()

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
        )

        specs = []
        for evolution_method in ('CN', 'SO', 'S'):
            specs.append(
                ion.LineSpecification(
                    f'Line_HAM_{evolution_method}_LEN',
                    evolution_equations = 'HAM',
                    evolution_method = evolution_method,
                    evolution_gauge = 'LEN',
                    **shared_spec_kwargs
                )
            )
        specs.append(
            ion.CylindricalSliceSpecification(
                f'CylindricalSlice_HAM_CN_LEN',
                evolution_equations = 'HAM',
                evolution_method = 'CN',
                evolution_gauge = 'LEN',
                **shared_spec_kwargs
            )
        )
        specs.append(
            ion.SphericalSliceSpecification(
                f'SphericalSlice_HAM_CN_LEN',
                evolution_equations = 'HAM',
                evolution_method = 'CN',
                evolution_gauge = 'LEN',
                **shared_spec_kwargs
            )
        )
        specs.append(
            ion.SphericalHarmonicSpecification(
                f'SphericalHarmonic_LAG_CN_LEN',
                evolution_equations = 'LAG',
                evolution_method = 'CN',
                evolution_gauge = 'LEN',
                **shared_spec_kwargs
            )
        )
        for evolution_gauge in ('LEN', 'VEL'):
            specs.append(
                ion.SphericalHarmonicSpecification(
                    f'SphericalHarmonic_LAG_SO_{evolution_gauge}',
                    evolution_equations = 'LAG',
                    evolution_method = 'SO',
                    evolution_gauge = evolution_gauge,
                    **shared_spec_kwargs
                )
            )

        results = si.utils.multi_map(run, specs, processes = 3)
        for r in results:
            print(r.info())
            print()

        si.vis.xxyy_plot(
            'method_comparison',
            [
                *[r.data_times for r in results]
            ],
            [
                *[r.state_overlaps_vs_time[r.spec.initial_state] for r in results]
            ],
            line_labels = [', '.join((r.__class__.__name__, r.spec.evolution_method, r.spec.evolution_gauge)) for r in results],
            x_unit = 'asec',
            x_label = r'$t$',
            **PLOT_KWARGS,
        )

        identifier_to_final_initial_overlap = {(r.mesh.__class__, r.spec.evolution_equations, r.spec.evolution_method, r.spec.evolution_gauge): r.state_overlaps_vs_time[r.spec.initial_state][-1] for r in results}
        for k, v in identifier_to_final_initial_overlap.items():
            print(k, v)
        expected_results = {
            (ion.LineMesh, 'HAM', 'CN', 'LEN'): 0.0143651217635,
            (ion.LineMesh, 'HAM', 'SO', 'LEN'): 0.0143755731217,
            (ion.LineMesh, 'HAM', 'S', 'LEN'): 0.000568901854635,  # why is this not the same as the other line mesh methods?
            (ion.CylindricalSliceMesh, 'HAM', 'CN', 'LEN'): 0.293741923689,
            (ion.SphericalSliceMesh, 'HAM', 'CN', 'LEN'): 0.178275457029,
            (ion.SphericalHarmonicMesh, 'LAG', 'CN', 'LEN'): 0.31291499645,
            (ion.SphericalHarmonicMesh, 'LAG', 'SO', 'LEN'): 0.312873123597,
            (ion.SphericalHarmonicMesh, 'LAG', 'SO', 'VEL'): 0.319447857317,
        }

        summary = 'Results:\n'
        summary += '\n'.join(f'{", ".join((key[0].__name__, key[1], key[2]))}: {identifier_to_final_initial_overlap[key]:.6f} | {val:.6f}' for key, val in expected_results.items())
        print(summary)

        for key, val in expected_results.items():
            np.testing.assert_allclose(identifier_to_final_initial_overlap[key], val)
