"""
This script tests all of the evolution methods on each mesh.
"""

import itertools as it
import logging
import os

import simulacra as si
from simulacra.units import *

import ionization as ion


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

logman = si.utils.LogManager('simulacra', 'ionization',
                             stdout_level = logging.DEBUG,
                             file_logs = True, file_level = logging.WARN, file_dir = OUT_DIR, file_name = FILE_NAME)


def run_spec(spec):
    with logman as logger:
        try:
            sim = spec.to_sim()

            # sim.info().log()
            sim.run()
            sim.info().log()

            sim.plot_state_overlaps_vs_time(target_dir = OUT_DIR,
                                            img_format = 'png',
                                            fig_dpi_scale = 3, )

            return sim
        except NotImplementedError as e:
            logger.warn(f'{sim.mesh.__class__.__name__} does not support {sim.spec.evolution_method}-{sim.spec.evolution_equations}-{sim.spec.evolution_gauge}')
        except Exception as e:
            logger.exception(f'Simulation {sim.name} failed with exception {e}')


if __name__ == '__main__':
    try:
        os.remove(os.path.join(OUT_DIR, 'evolution_algorithms.log'))
    except FileNotFoundError:
        pass

    with logman as logger:

        bound = 25 * bohr_radius

        hyd_efield = ion.SineWave.from_frequency(1 / (100 * asec), amplitude = .2 * atomic_electric_field)
        hyd_efield.window = ion.SymmetricExponentialTimeWindow(window_time = hyd_efield.period_carrier * 3)
        hyd_spec_base = dict(
            r_bound = bound, rho_bound = bound, z_bound = bound,
            l_bound = 50,
            r_points = 400, theta_points = 100, rho_points = int(bound / bohr_radius) * 10, z_points = int(bound / bohr_radius) * 20,
            initial_state = ion.HydrogenBoundState(2, 0),
            test_states = tuple(ion.HydrogenBoundState(n, l) for n in range(5) for l in range(n)),
            electric_potential = hyd_efield,
            time_initial = -5 * hyd_efield.period_carrier, time_final = 5 * hyd_efield.period_carrier, time_step = 1 * asec,
            electric_potential_dc_correction = True,
        )

        line_potential = ion.HarmonicOscillator.from_energy_spacing_and_mass(1 * eV, electron_mass)
        line_efield = ion.SineWave.from_photon_energy(1 * eV, amplitude = .2 * atomic_electric_field)
        line_efield.window = ion.SymmetricExponentialTimeWindow(window_time = hyd_efield.period_carrier * 3)
        line_spec_base = hyd_spec_base.copy()
        line_spec_base.update(dict(
            x_bound = bound, x_points = 2 ** 10,
            internal_potential = line_potential,
            electric_potential = line_efield,
            time_initial = -5 * hyd_efield.period_carrier, time_final = 5 * hyd_efield.period_carrier,
            initial_state = ion.QHOState.from_potential(line_potential, electron_mass),
            test_states = tuple(ion.QHOState.from_potential(line_potential, electron_mass, n) for n in range(20)),
        ))

        specs = []

        for method, equations, gauge in it.product(
                ('CN', 'SO'),
                ('HAM', 'LAG'),
                ('LEN', 'VEL')):
            for spec_type in (ion.CylindricalSliceSpecification, ion.SphericalSliceSpecification, ion.SphericalHarmonicSpecification):
                specs.append(
                    spec_type(f'{spec_type.__name__[:-13]}__gauge={gauge}_method={method}_equations={equations}',
                              **hyd_spec_base,
                              evolution_method = method, evolution_equations = equations, evolution_gauge = gauge,
                              )
                )

            specs.append(
                ion.LineSpecification(f'Line__gauge={gauge}_method={method}_equations={equations}',
                                      **line_spec_base,
                                      evolution_method = method, evolution_equations = equations, evolution_gauge = gauge,
                                      )
            )

        results = si.utils.multi_map(run_spec, specs)
