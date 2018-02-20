import logging
import os
import itertools

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

if __name__ == '__main__':
    with LOGMAN as logger:
        energy_spacing = .1 * u.eV
        test_mass = u.electron_mass

        qho = ion.potentials.HarmonicOscillator.from_energy_spacing_and_mass(
            energy_spacing = energy_spacing,
            mass = test_mass,
        )
        states = [ion.states.QHOState.from_potential(qho, n = n, mass = test_mass) for n in range(30)]

        efield = ion.potentials.SineWave.from_photon_energy(
            photon_energy = energy_spacing,
            amplitude = .0001 * u.atomic_electric_field,
        )

        sim = ion.mesh.LineSpecification(
            'qho',
            internal_potential = qho,
            electric_potential = efield,
            time_initial = 0,
            time_final = 5 * efield.period,
            time_step = .001 * efield.period,
            initial_state = states[0],
            test_states = states,
            z_bound = 100 * u.nm,
            z_points = 2 ** 12,
            evolution_method = ion.mesh.AlternatingDirectionImplicitCrankNicolson(),
            # evolution_method = ion.mesh.LineSplitOperator(),
            animators = [
                ion.mesh.anim.RectangleSplitLowerAnimator(
                    axman_wavefunction = ion.mesh.anim.LineMeshAxis(),
                    fig_dpi_scale = 2,
                    length = 10,
                    target_dir = OUT_DIR,
                )
            ],
        ).to_sim()

        sim.run(progress_bar = True)
        sim.info().log()

        sim.plot_state_overlaps_vs_time(**PLOT_KWARGS)
