import logging
import os

import numpy as np

import simulacra as si
from simulacra.units import *

import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

logman = si.utils.LogManager("simulacra", "ionization", stdout_level=logging.DEBUG)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=5)


# class VolkovaWindow(ion.potentials.TimeWindow):
#     def __init__(self, *, t_front, t_platueau):
#         super().__init__()
#
#         self.t_front = t_front
#         self.t_plateau = t_platueau
#
#     def __call__(self, t):
#         cond_before = np.less(t, self.t_front)
#         cond_middle = np.less_equal(self.t_front, t) * np.less_equal(t, self.t_front + self.t_plateau)
#         cond_after = np.less(self.t_front + self.t_plateau, t) * np.less_equal(t, (2 * self.t_front) + self.t_plateau)
#
#         out = np.where(cond_before, np.sin(pi * t / (2 * self.t_front)) ** 2, 0)
#         out += np.where(cond_middle, 1, 0)
#         out += np.where(cond_after, np.cos(pi * (t - (self.t_front + self.t_plateau)) / (2 * self.t_front)) ** 2, 0)
#
#         return out


if __name__ == "__main__":
    with logman as logger:
        dummy = ion.SineWave.from_photon_energy_and_intensity(
            0.5 * eV, intensity=100 * TWcm2, phase=pi / 2
        )
        efield = ion.SineWave.from_photon_energy_and_intensity(
            2 * eV, intensity=100 * TWcm2, phase=pi / 2
        )
        efield.window = ion.potentials.SmoothedTrapezoidalWindow(
            time_front=1 * dummy.period_carrier, time_plateau=5 * dummy.period_carrier
        )

        times = np.linspace(0, 8 * dummy.period_carrier, 1e5)

        si.vis.xy_plot(
            "window",
            times,
            efield.window(times),
            x_label=r"$ t $",
            x_unit="fsec",
            y_label="Window Function",
            **PLOT_KWARGS,
        )

        si.vis.xy_plot(
            "field",
            times,
            efield.get_electric_field_amplitude(times),
            # np.abs(efield.get_electric_field_amplitude(times)),
            # line_labels = (fr'$ {ion.vis.LATEX_EFIELD}(t) $', fr'$ \left| {ion.vis.LATEX_EFIELD}(t) \right| $'),
            # line_kwargs = (None, {'linestyle': '--'}),
            x_label=r"$ t $",
            x_unit="fsec",
            y_label=fr"$ {ion.vis.LATEX_EFIELD}(t) $",
            y_unit="atomic_electric_field",
            **PLOT_KWARGS,
        )

        si.vis.xy_plot(
            "field_and_vector",
            times,
            efield.get_electric_field_amplitude(times) / atomic_electric_field,
            efield.get_vector_potential_amplitude_numeric_cumulative(times)
            / (atomic_momentum / proton_charge),
            line_labels=(
                fr"$ {ion.vis.LATEX_EFIELD}(t) $",
                fr"$ e \, {ion.LATEX_AFIELD}(t) $",
            ),
            x_label=r"$ t $",
            x_unit="fsec",
            y_label=fr"$ {ion.vis.LATEX_EFIELD}(t), \; e \, {ion.LATEX_AFIELD}(t) $",
            **PLOT_KWARGS,
        )
