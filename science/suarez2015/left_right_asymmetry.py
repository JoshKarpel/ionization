import logging
import os
import datetime

import numpy as np

import simulacra as si
from simulacra.units import *

import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

SIM_LIB = os.path.join(OUT_DIR, 'SIMLIB')

logman = si.utils.LogManager(
    'simulacra', 'ionization',
    stdout_level = logging.DEBUG,
)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)


def make_numeric_eigenstate_plots(sim):
    normalize = np.max(np.abs(list(sim.bound_states)[0].g) ** 2)

    for state in sim.bound_states:
        si.vis.xy_plot(
            f'state_{state.n}',
            sim.mesh.x_mesh,
            np.abs(state.g) ** 2 / normalize,
            x_lower_limit = -50 * bohr_radius, x_upper_limit = 50 * bohr_radius,
            x_unit = 'bohr_radius',
            x_label = r'$ z $',
            y_label = r'$ \left| \psi(z) \right|^2 $',
            y_lower_limit = 0, y_upper_limit = 1, y_pad = 0,
            **PLOT_KWARGS,
        )


if __name__ == '__main__':
    with logman as logger:
        num_states = 10

        peak_intensity = 1e14 * Wcm2
        wavelength = 800 * nm
        number_of_cycles = 4

        peak_amplitude = np.sqrt(peak_intensity / (epsilon_0 * c))

        pulse = ion.CosSquaredPulse(
            amplitude = peak_amplitude,
            wavelength = wavelength,
            number_of_cycles = number_of_cycles,
        )

        time_final = pulse.number_of_cycles * pulse.period
        pulse.pulse_center = time_final / 2

        spec = ion.LineSpecification(
            f'LR_{num_states}',
            internal_potential = ion.SoftCoulomb(),
            electric_potential = pulse,
            initial_state = ion.OneDSoftCoulombState(),
            x_bound = 3400 * bohr_radius, x_points = 17000,
            time_initial = 0, time_final = time_final, time_step = 0.02 * atomic_time,
            use_numeric_eigenstates = True,
            number_of_numeric_eigenstates = num_states,
            analytic_eigenstate_type = ion.OneDSoftCoulombState,
            # checkpoints = True,
            # checkpoint_every = datetime.timedelta(minutes = 1),
            # checkpoint_dir = SIM_LIB,
            animators = [
                ion.animators.RectangleAnimator(
                    axman_wavefunction = ion.animators.LineMeshAxis(
                        log = True,
                        plot_limit = 1000 * bohr_radius,
                    ),
                    fig_dpi_scale = 2,
                    length = 30,
                    target_dir = OUT_DIR,
                ),
            ]
        )

        sim = si.utils.find_or_init_sim(spec, search_dir = SIM_LIB)

        sim.info().log()

        # make_numeric_eigenstate_plots(sim)

        if sim.status != si.Status.FINISHED:
            sim.run_simulation()
            sim.animators = None
            sim.save(save_mesh = True, target_dir = SIM_LIB)

        sim.info().log()
