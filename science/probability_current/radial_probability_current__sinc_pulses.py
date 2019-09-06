import logging
import os
import itertools
import datetime

import numpy as np
import simulacra as si
from simulacra.units import *

import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)
SIM_LIB = os.path.join(OUT_DIR, "SIMLIB")

LOGMAN = si.utils.LogManager("simulacra", "ionization", stdout_level=logging.INFO)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)


def run_spec(spec):
    with LOGMAN as logger:
        sim = si.utils.find_or_init_sim(spec, search_dir=SIM_LIB)

        sim.info().log()
        if sim.status != si.Status.FINISHED:
            sim.run_simulation()
            sim.save(save_mesh=True, target_dir=SIM_LIB)
        sim.info().log()

        sim.plot_wavefunction_vs_time(show_vector_potential=False, **PLOT_KWARGS)

        sim.plot_radial_probability_current_vs_time__combined(
            r_upper_limit=15 * bohr_radius,
            t_lower_limit=-5 * sim.spec.electric_potential[0].pulse_width,
            t_upper_limit=10
            * sim.spec.electric_potential[0].pulse_width ** PLOT_KWARGS,
        )

    return sim


if __name__ == "__main__":
    with LOGMAN as logger:
        pulse_type = ion.potentials.SincPulse

        pulse_widths = np.array([50, 100, 200, 400, 800]) * asec
        fluences = np.array([0.1, 1, 10, 20]) * Jcm2
        phases = [0, pi / 4, pi / 2]

        pulse_time_bound = 20
        sim_time_bound = 23

        dt = 1 * asec
        r_bound = 100 * bohr_radius
        r_points_per_br = 10
        l_bound = 500

        specs = []
        for pw, flu, cep in itertools.product(pulse_widths, fluences, phases):
            pulse = pulse_type(
                pulse_width=pw,
                fluence=flu,
                phase=cep,
                window=ion.potentials.LogisticWindow(
                    window_time=pulse_time_bound * pw, window_width=0.2 * pw
                ),
            )

            specs.append(
                ion.SphericalHarmonicSpecification(
                    f"{pulse_type.__name__}__pw={pw / asec:3f}as_flu={flu / Jcm2:3f}jcm2_cep={cep / pi:3f}pi__R={r_bound / bohr_radius:3f}br_ppbr={r_points_per_br}_L={l_bound}_dt={dt / asec:3f}as",
                    r_bound=r_bound,
                    r_points=r_points_per_br * r_bound / bohr_radius,
                    l_bound=l_bound,
                    time_step=dt,
                    time_initial=-sim_time_bound * pw,
                    time_final=sim_time_bound * pw,
                    mask=ion.RadialCosineMask(0.8 * r_bound, r_bound),
                    electric_potential=pulse,
                    electric_potential_dc_correction=True,
                    use_numeric_eigenstates=True,
                    numeric_eigenstate_max_energy=10 * eV,
                    numeric_eigenstate_max_angular_momentum=20,
                    checkpoints=True,
                    checkpoint_dir=SIM_LIB,
                    checkpoint_every=datetime.timedelta(minutes=1),
                    store_radial_probability_current=True,
                    store_data_every=4,
                )
            )

        si.utils.multi_map(run_spec, specs, processes=5)
