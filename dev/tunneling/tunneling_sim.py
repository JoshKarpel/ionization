import logging
import os

import numpy as np

import simulacra as si
import simulacra.units as u

import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

LOGMAN = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.DEBUG)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)

if __name__ == '__main__':
    with LOGMAN as logger:
        sim = ion.tunneling.TunnelingSpecification(
            'test',
            model = ion.tunneling.models.LandauRate(),
            time_final = 10 * u.fsec,
            electric_potential = ion.potentials.Rectangle(
                start_time = 1 * u.fsec,
                end_time = 9 * u.fsec,
                amplitude = .05 * u.atomic_electric_field,
            ),
        ).to_simulation()

        print(sim.info())

        sim.run()

        print(sim.b2)

        si.vis.xy_plot(
            'b_vs_t',
            sim.times,
            sim.b2,
            x_unit = 'asec',
            **PLOT_KWARGS
        )
