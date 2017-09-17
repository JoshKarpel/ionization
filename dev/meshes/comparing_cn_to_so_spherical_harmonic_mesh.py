import logging
import os

import simulacra as si
from simulacra.units import *

import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

LOGMAN = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.DEBUG)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)


def run_sim(spec):
    with LOGMAN as logger:
        sim = spec.to_simulation()

        sim.info().log()
        sim.run_simulation()
        sim.info().log()

        sim.plot_wavefunction_vs_time(**PLOT_KWARGS)
        sim.plot_radial_probability_current_vs_time(**PLOT_KWARGS)


if __name__ == '__main__':
    with LOGMAN as logger:
        pulse = ion.Rectangle(start_time = 50 * asec, end_time = 150 * asec, amplitude = .5 * atomic_electric_field)

        shared_kwargs = dict(
            electric_potential = pulse,
            r_bound = 50 * bohr_radius,
            r_points = 200,
            l_bound = 100,
            time_initial = 0, time_final = 300 * asec, time_step = 1 * asec,
            use_numeric_eigenstates = True,
            numeric_eigenstate_max_energy = 10 * eV,
            numeric_eigenstate_max_angular_momentum = 5,
            store_radial_probability_current = True,
        )

        specs = [
            ion.SphericalHarmonicSpecification(
                'CN',
                evolution_method = 'CN',
                **shared_kwargs,
            ),
            ion.SphericalHarmonicSpecification(
                'SO',
                evolution_method = 'SO',
                **shared_kwargs,
            )
        ]

        si.utils.multi_map(run_sim, specs, processes = 2)
