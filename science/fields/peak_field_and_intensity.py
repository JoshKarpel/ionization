import logging
import os

from tqdm import tqdm

import numpy as np

import simulacra as si
from simulacra.units import *
import ionization as ion

# import matplotlib.pyplot as plt

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

log = si.utils.LogManager(
    "simulacra",
    "ionization",
    stdout_level=logging.INFO,
    file_logs=False,
    file_dir=OUT_DIR,
    file_level=logging.DEBUG,
)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=3)


def pulse_peaks_values_2d(
    pulse_width_min,
    pulse_width_max,
    fluence_min,
    fluence_max,
    pulse_type=ion.potentials.SincPulse,
    phase=0,
    points=200,
):
    pulse_widths = np.linspace(pulse_width_min, pulse_width_max, points)
    fluences = np.geomspace(fluence_min, fluence_max, points)

    pulse_width_mesh, fluence_mesh = np.meshgrid(pulse_widths, fluences, indexing="ij")
    max_efield_mesh = np.empty_like(pulse_width_mesh)
    max_intensity_mesh = np.empty_like(pulse_width_mesh)

    for ii, pulse_width in enumerate(tqdm(pulse_widths)):
        for jj, fluence in enumerate(fluences):
            pulse = ion.potentials.SincPulse(
                pulse_width=pulse_width, fluence=fluence, phase=phase
            )
            if pulse_type != ion.potentials.SincPulse:
                pulse = pulse_type(
                    pulse_width=pulse_width,
                    fluence=fluence,
                    phase=phase,
                    omega_carrier=pulse.omega_carrier,
                )

            times = np.linspace(-5 * pulse_width, 5 * pulse_width, 2000)
            max_efield = np.max(np.abs(pulse.get_electric_field_amplitude(times)))

            max_efield_mesh[ii, jj] = max_efield
            max_intensity_mesh[ii, jj] = epsilon_0 * c * (max_efield ** 2)

    common_kwargs = dict(
        x_label=r"Pulse Width $\tau$",
        x_unit="asec",
        y_label=fr"Fluence $H$",
        y_unit="Jcm2",
        y_log_axis=True,
    )

    obi_field = (pi * epsilon_0 / (proton_charge ** 3)) * (rydberg ** 2)
    obi_intensity = c * epsilon_0 * (obi_field ** 2)

    postfix = f"{pulse_width_min / asec:3f}as_to_{pulse_width_max / asec:3f}as__{fluence_min / Jcm2:3f}jcm2_to_{fluence_max / Jcm2:3f}jcm2_phase={phase / pi:2f}pi___{pulse_type.__name__}"

    si.vis.xyz_plot(
        f"2d_pulse_efield__{postfix}",
        pulse_width_mesh,
        fluence_mesh,
        max_efield_mesh,
        z_unit="atomic_electric_field",
        z_log_axis=True,
        z_label=fr"Max $\left| {ion.vis.LATEX_EFIELD} \right|$ vs. $\tau$ and $H$ for {pulse_type.__name__}",
        contours=[
            0.01 * atomic_electric_field,
            0.05 * atomic_electric_field,
            0.1 * atomic_electric_field,
            obi_field,
            0.5 * atomic_electric_field,
            1 * atomic_electric_field,
        ],
        contour_kwargs={"colors": "white", "linewidths": 0.5},
        **common_kwargs,
        **PLOT_KWARGS,
    )
    si.vis.xyz_plot(
        f"2d_pulse_intensity__{postfix}",
        pulse_width_mesh,
        fluence_mesh,
        max_intensity_mesh,
        z_unit="atomic_intensity",
        z_log_axis=True,
        z_label=fr"Max $P$ vs. $\tau$ and $H$ for {pulse_type.__name__}",
        contours=[
            0.01 * atomic_intensity,
            0.05 * atomic_intensity,
            0.1 * atomic_intensity,
            obi_intensity,
            0.5 * atomic_intensity,
            1 * atomic_intensity,
        ],
        contour_kwargs={"colors": "white", "linewidths": 0.5},
        **common_kwargs,
        **PLOT_KWARGS,
    )


if __name__ == "__main__":
    with log as logger:
        points = 300
        for pulse_type in (ion.potentials.SincPulse, ion.potentials.GaussianPulse):
            pulse_peaks_values_2d(
                50 * asec,
                1000 * asec,
                0.001 * Jcm2,
                30 * Jcm2,
                pulse_type=pulse_type,
                points=points,
            )
            # pulse_peaks_values_2d(50 * asec, 1000 * asec, 0.01 * Jcm2, 20 * Jcm2, pulse_type = pulse_type, points = points)
            # pulse_peaks_values_2d(50 * asec, 1000 * asec, 0.01 * Jcm2, 50 * Jcm2, pulse_type = pulse_type, points = points)
            pulse_peaks_values_2d(
                50 * asec,
                5000 * asec,
                0.001 * Jcm2,
                100 * Jcm2,
                pulse_type=pulse_type,
                points=points,
            )
