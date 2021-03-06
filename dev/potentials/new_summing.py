import logging
import os

import numpy as np
import simulacra as si
from simulacra.units import *

import ionization as ion


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

if __name__ == "__main__":
    with si.utils.LogManager(
        "simulacra", "ionization", stdout_level=logging.DEBUG
    ) as logger:
        times = np.linspace(0, 20, 1000) * s

        field_1 = ion.SineWave(1 * Hz, amplitude=1 * V / m)
        field_2 = ion.potentials.Rectangle(
            start_time=5 * s, end_time=10 * s, amplitude=1 * V / m
        )

        field_sum = field_1 + field_2

        si.vis.xy_plot(
            "sum_plot",
            times,
            field_1.get_electric_field_amplitude(times),
            field_2.get_electric_field_amplitude(times),
            field_sum.get_electric_field_amplitude(times),
            line_labels=("field 1", "field 2", "field sum"),
            target_dir=OUT_DIR,
        )
