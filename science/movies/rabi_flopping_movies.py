import logging
import os

import numpy as np
import simulacra as si
from simulacra.units import *

import ionization as ion

import matplotlib.pyplot as plt


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

logman = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.DEBUG, )


def run(spec):
    with logman as logger:
        sim = spec.to_simulation()

        logger.info(sim.info())
        sim.run_simulation()
        logger.info(sim.info())

        return sim


if __name__ == '__main__':
    with logman as logger:
        state_a = ion.HydrogenBoundState(1, 0)
        state_b = ion.HydrogenBoundState(2, 1)

        amplitudes = [.001, .005, .01, .1]
        cycles = [1]
        dt = 1

        specs = []

        for amplitude in amplitudes:
            for cycle in cycles:
                animator_kwargs = dict(
                        target_dir = OUT_DIR,
                        length = 30,
                )

                animators = [
                    ion.animators.SphericalHarmonicAnimator(
                            postfix = 'g2_full',
                            axman_wavefunction = ion.animators.SphericalHarmonicPhiSliceMeshAxis(
                                    which = 'g2',
                            ),
                            **animator_kwargs,
                    ),
                    ion.animators.SphericalHarmonicAnimator(
                            postfix = 'g2_zoom',
                            axman_wavefunction = ion.animators.SphericalHarmonicPhiSliceMeshAxis(
                                    which = 'g2',
                                    plot_limit = 20 * bohr_radius,
                            ),
                            **animator_kwargs,
                    ),
                    ion.animators.SphericalHarmonicAnimator(
                            postfix = 'g_full',
                            axman_wavefunction = ion.animators.SphericalHarmonicPhiSliceMeshAxis(
                                    which = 'g',
                                    colormap = plt.get_cmap('richardson'),
                                    norm = si.plots.RichardsonNormalization(),
                            ),
                            **animator_kwargs,
                    ),
                    ion.animators.SphericalHarmonicAnimator(
                            postfix = 'g_zoom',
                            axman_wavefunction = ion.animators.SphericalHarmonicPhiSliceMeshAxis(
                                    which = 'g',
                                    colormap = plt.get_cmap('richardson'),
                                    norm = si.plots.RichardsonNormalization(),
                                    plot_limit = 20 * bohr_radius,
                            ),
                            **animator_kwargs,
                    ),
                ]

                ###

                electric_field = ion.SineWave.from_photon_energy(np.abs(state_a.energy - state_b.energy), amplitude = amplitude * atomic_electric_field)

                bound = 50
                ppbr = 8

                spec_kwargs = dict(
                        r_bound = bound * bohr_radius,
                        r_points = bound * ppbr,
                        l_bound = max(state_a.l, state_b.l) + 10,
                        initial_state = state_a,
                        test_states = (state_a, state_b),
                        time_initial = 0 * asec,
                        time_step = dt * asec,
                        mask = ion.RadialCosineMask(inner_radius = .8 * bound * bohr_radius, outer_radius = bound * bohr_radius),
                        animators = animators,
                        out_dir = OUT_DIR
                )

                dummy = ion.SphericalHarmonicSpecification('dummy', **spec_kwargs).to_simulation()
                matrix_element = np.abs(dummy.mesh.dipole_moment_expectation_value(mesh_a = dummy.mesh.get_g_for_state(state_b)))
                rabi_frequency = amplitude * atomic_electric_field * matrix_element / hbar / twopi
                rabi_time = 1 / rabi_frequency

                specs.append(ion.SphericalHarmonicSpecification(
                        f'rabi_{state_a.n}_{state_a.l}_to_{state_b.n}_{state_b.l}__amp={amplitude}aef__{cycle}cycles__bound={bound}br__ppbr={ppbr}__dt={dt}as',
                        time_final = cycle * rabi_time,
                        electric_potential = electric_field,
                        rabi_frequency = rabi_frequency,
                        rabi_time = rabi_time,
                        **spec_kwargs
                ))

        si.utils.multi_map(run, specs)
