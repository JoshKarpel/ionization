import logging
import os

import simulacra as si
from simulacra.units import *

import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

# def make_movie(spec):
#     with si.utils.LogManager('simulacra', 'ionization', stdout_logs = True, stdout_level = logging.INFO,
#                              file_logs = True, file_name = spec.name, file_dir = OUT_DIR, file_mode = 'w', file_level = logging.DEBUG) as logger:
#         sim = spec.to_simulation()
#
#         sim.info().log()
#         sim.run_simulation()
#         sim.info().log()


if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization', stdout_logs = True, stdout_level = logging.DEBUG) as logger:
        anim_kwargs = dict(
                length = 10,
                target_dir = OUT_DIR,
        )

        animators = [
            animation.animators.RectangleAnimator(
                    postfix = 'g2',
                    axman_wavefunction = animation.animators.CylindricalSliceMeshAxis(shading = 'flat'),
                    axman_lower = animation.animators.ElectricPotentialPlotAxis(),
                    **anim_kwargs,
            ),
            animation.animators.RectangleAnimator(
                    postfix = 'g',
                    axman_wavefunction = animation.animators.CylindricalSliceMeshAxis(
                            which = 'g',
                            colormap = si.vis.RichardsonColormap(),
                            norm = si.vis.RichardsonNormalization(),
                            shading = 'flat'),
                    axman_lower = animation.animators.ElectricPotentialPlotAxis(),
                    **anim_kwargs,
            )
        ]

        sim = ion.CylindricalSliceSpecification('cyl_slice',
                                                time_initial = 0 * asec, time_final = 300 * asec,
                                                z_bound = 20 * bohr_radius, rho_bound = 20 * bohr_radius,
                                                z_points = 200, rho_points = 100,
                                                electric_potential = ion.Rectangle(start_time = 100 * asec, end_time = 150 * asec, amplitude = 1 * atomic_electric_field),
                                                animators = animators).to_simulation()

        sim.info().log()
        sim.run()
        sim.info().log()

        # window = ion.LinearRampTimeWindow(ramp_on_time = t_init * asec, ramp_time = 200 * asec)
        # e_field = ion.SineWave.from_frequency(1 / (50 * asec), amplitude = 1 * atomic_electric_field, window = window)
        # mask = ion.RadialCosineMask(inner_radius = (bound - 25) * bohr_radius, outer_radius = bound * bohr_radius)
        #
        # animators = [
        #     src.ionization.animators.CylindricalSliceAnimator(postfix = '_nm', target_dir = OUT_DIR, distance_unit = 'nm'),
        #     src.ionization.animators.CylindricalSliceAnimator(postfix = '_br', target_dir = OUT_DIR, distance_unit = 'bohr_radius'),
        # ]
        # specs.append(ion.CylindricalSliceSpecification('cyl_slice',
        #                                                time_initial = t_init * asec, time_final = t_final * asec, time_step = dt * asec,
        #                                                z_bound = 20 * bohr_radius, z_points = 300,
        #                                                rho_bound = 20 * bohr_radius, rho_points = 150,
        #                                                initial_state = ion.HydrogenBoundState(1, 0, 0),
        #                                                electric_potential = e_field,
        #                                                mask = ion.RadialCosineMask(inner_radius = 15 * bohr_radius, outer_radius = 20 * bohr_radius),
        #                                                animators = animators
        #                                                ))
        #
        # animators = [
        #     src.ionization.animators.SphericalSliceAnimator(postfix = '_nm', target_dir = OUT_DIR, distance_unit = 'nm'),
        #     src.ionization.animators.SphericalSliceAnimator(postfix = '_br', target_dir = OUT_DIR, distance_unit = 'bohr_radius'),
        # ]
        # specs.append(ion.SphericalSliceSpecification('sph_slice',
        #                                              time_initial = t_init * asec, time_final = t_final * asec, time_step = dt * asec,
        #                                              r_bound = bound * bohr_radius, r_points = radial_points,
        #                                              theta_points = angular_points,
        #                                              initial_state = initial_state,
        #                                              electric_potential = e_field,
        #                                              mask = mask,
        #                                              animators = animators
        #                                              ))
        #
        # animators = [
        #     src.ionization.animators.SphericalHarmonicAnimator(postfix = '_nm', target_dir = OUT_DIR, distance_unit = 'nm'),
        #     src.ionization.animators.SphericalHarmonicAnimator(postfix = '_br', target_dir = OUT_DIR, distance_unit = 'bohr_radius'),
        # ]
        # specs.append(ion.SphericalHarmonicSpecification('sph_harm', time_initial = t_init * asec, time_final = t_final * asec, time_step = dt * asec,
        #                                                 r_bound = bound * bohr_radius, r_points = radial_points,
        #                                                 l_bound = angular_points,
        #                                                 initial_state = initial_state,
        #                                                 electric_potential = e_field,
        #                                                 mask = mask,
        #                                                 animators = animators,
        #                                                 ))
        #
        # #######
        #
        # mass = electron_mass
        # pot = ion.HarmonicOscillator.from_energy_spacing_and_mass(energy_spacing = 1 * eV, mass = mass)
        #
        # init = ion.QHOState.from_potential(pot, mass, n = 1) + ion.QHOState.from_potential(pot, mass, n = 2)
        #
        # animators = [
        #     src.ionization.animators.LineAnimator(postfix = '_nm', target_dir = OUT_DIR, distance_unit = 'nm'),
        #     src.ionization.animators.LineAnimator(postfix = '_br', target_dir = OUT_DIR, distance_unit = 'bohr_radius'),
        # ]
        # specs.append(ion.LineSpecification('line',
        #                                    x_bound = 50 * nm, x_points = 2 ** 14,
        #                                    internal_potential = pot,
        #                                    electric_potential = ion.SineWave.from_photon_energy(1 * eV, amplitude = .05 * atomic_electric_field),
        #                                    test_states = (ion.QHOState.from_potential(pot, mass, n = n) for n in range(20)),
        #                                    initial_state = init,
        #                                    time_initial = t_init * asec, time_final = t_final * 10 * asec, time_step = dt * asec,
        #                                    mask = ion.RadialCosineMask(inner_radius = 40 * nm, outer_radius = 50 * nm),
        #                                    animators = animators
        #                                    ))
        #
        # si.utils.multi_map(make_movie, specs, processes = 4)
