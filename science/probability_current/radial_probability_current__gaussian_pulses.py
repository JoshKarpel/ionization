import logging
import os
import itertools
import datetime

import numpy as np
import simulacra as si
from simulacra.units import *

import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)
SIM_LIB = os.path.join(OUT_DIR, 'SIMLIB')

LOGMAN = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.INFO)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)


def run_spec(spec):
    with LOGMAN as logger:
        sim = si.utils.find_or_init_sim(spec, search_dir = SIM_LIB)

        sim.info().log()
        if sim.status != si.Status.FINISHED:
            sim.run_simulation()
            sim.save(save_mesh = True, target_dir = SIM_LIB)
        sim.info().log()

        sim.plot_wavefunction_vs_time(
            show_vector_potential = False,
            **PLOT_KWARGS,
        )

        sim.plot_radial_probability_current_vs_time__combined(
            r_limit = 15 * bohr_radius,
            time_lower_limit = -3 * sim.spec.electric_potential[0].pulse_width,
            **PLOT_KWARGS
        )

    return sim


if __name__ == '__main__':
    with LOGMAN as logger:
        pulse_type = ion.GaussianPulse

        pulse_widths = np.array([50, 100, 200, 400, 800]) * asec
        fluences = np.array([.1, 1, 10, 20]) * Jcm2
        phases = [0, pi / 4, pi / 2]
        number_of_cycles = [2, ]

        pulse_time_bound = 5
        sim_time_bound = 6
        extra_end_time = 2

        dt = 1 * asec
        r_bound = 100 * bohr_radius
        r_points_per_br = 10
        l_bound = 500

        specs = []
        for pw, flu, cep, n_cycles in itertools.product(pulse_widths, fluences, phases, number_of_cycles):
            pulse = pulse_type.from_number_of_cycles(
                pulse_width = pw, fluence = flu, phase = cep,
                number_of_cycles = n_cycles,
                number_of_pulse_widths = 3,
                window = ion.SymmetricExponentialTimeWindow(
                    window_time = pulse_time_bound * pw,
                    window_width = .2 * pw,
                )
            )

            specs.append(
                ion.SphericalHarmonicSpecification(
                    f'{pulse_type.__name__}__Nc={n_cycles}_pw={uround(pw, asec)}as_flu={uround(flu, Jcm2)}jcm2_cep={uround(cep, pi)}pi__R={uround(r_bound, bohr_radius)}br_ppbr={r_points_per_br}_L={l_bound}_dt={uround(dt, asec)}as',
                    r_bound = r_bound,
                    r_points = r_points_per_br * r_bound / bohr_radius,
                    l_bound = l_bound,
                    time_step = dt,
                    time_initial = -sim_time_bound * pw, time_final = (sim_time_bound + extra_end_time) * pw,
                    mask = ion.RadialCosineMask(.8 * r_bound, r_bound),
                    electric_potential = pulse,
                    electric_potential_dc_correction = True,
                    use_numeric_eigenstates = True,
                    numeric_eigenstate_max_energy = 20 * eV,
                    numeric_eigenstate_max_angular_momentum = 20,
                    checkpoints = True,
                    checkpoint_dir = SIM_LIB,
                    checkpoint_every = datetime.timedelta(minutes = 1),
                    store_radial_probability_current = True,
                )
            )

        si.utils.multi_map(run_spec, specs, processes = 4)
