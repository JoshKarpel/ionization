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


def monochromatic_keldysh_1d(
    photon_energy_min,
    photon_energy_max,
    electric_field_amplitudes,
    ionization_potential=-rydberg,
):
    energies = np.linspace(photon_energy_min, photon_energy_max, 1e4)
    gammas = []
    labels = []
    for amplitude in tqdm(electric_field_amplitudes):
        efields = (
            ion.SineWave.from_photon_energy(photon_energy, amplitude=amplitude)
            for photon_energy in energies
        )
        gammas.append(
            np.array(
                list(
                    efield.keldysh_parameter(ionization_potential) for efield in efields
                )
            )
        )
        labels.append(
            fr"${ion.vis.LATEX_EFIELD}_0 = {amplitude / atomic_electric_field:3f} \; \mathrm{{a.u.}}$"
        )

    si.vis.xy_plot(
        f"1d_monochromatic_keldysh__{photon_energy_min / eV:3f}eV_to_{photon_energy_max / eV:3f}eV",
        energies,
        *gammas,
        line_labels=labels,
        x_label=r"Photon Energy $E$",
        x_unit="eV",
        y_label=r"Keldysh Parameter $\gamma$",
        y_lower_limit=0,
        y_upper_limit=10,
        y_pad=0,
        hlines=[1],
        hline_kwargs=[{"linestyle": "--", "color": "black"}],
        title="Keldysh Parameter vs. Photon Energy",
        **PLOT_KWARGS,
    )


def monochromatic_keldysh_2d(
    photon_energy_min,
    photon_energy_max,
    electric_field_amplitude_min,
    electric_field_amplitude_max,
    ionization_potential=-rydberg,
    points=200,
):
    photon_energies = np.linspace(photon_energy_min, photon_energy_max, points)
    electric_field_amplitudes = np.linspace(
        electric_field_amplitude_min, electric_field_amplitude_max, points
    )

    photon_energy_mesh, electric_field_amplitude_mesh = np.meshgrid(
        photon_energies, electric_field_amplitudes, indexing="ij"
    )
    gamma_mesh = np.empty_like(photon_energy_mesh)

    for ii, photon_energy in enumerate(tqdm(photon_energies)):
        for jj, electric_field_amplitude in enumerate(electric_field_amplitudes):
            gamma_mesh[ii, jj] = ion.SineWave.from_photon_energy(
                photon_energy, amplitude=electric_field_amplitude
            ).keldysh_parameter(ionization_potential)

    si.vis.xyz_plot(
        f"2d_monochromatic_keldysh__{photon_energy_min / eV:3f}eV_to_{photon_energy_max / eV:3f}eV__{electric_field_amplitude_min / atomic_electric_field:3f}aef_to_{electric_field_amplitude_max / atomic_electric_field:3f}aef",
        photon_energy_mesh,
        electric_field_amplitude_mesh,
        gamma_mesh,
        x_label=r"Photon Energy $E$",
        x_unit="eV",
        y_label=fr"Electric Field Amplitude ${ion.vis.LATEX_EFIELD}_0$",
        y_unit="atomic_electric_field",
        z_log_axis=True,
        z_lower_limit=0.01,
        z_upper_limit=10,
        z_label=fr"Keldysh Parameter $\gamma$ vs. $E$ and ${ion.vis.LATEX_EFIELD}_0$",
        contours=(0.01, 0.1, 0.5, 1),
        contour_kwargs={"colors": "white", "linewidths": 0.5},
        **PLOT_KWARGS,
    )


def pulse_keldysh_2d(
    pulse_width_min,
    pulse_width_max,
    fluence_min,
    fluence_max,
    ionization_potential=-rydberg,
    pulse_type=ion.potentials.SincPulse,
    points=200,
):
    pulse_widths = np.linspace(pulse_width_min, pulse_width_max, points)
    fluences = np.geomspace(fluence_min, fluence_max, points)

    pulse_width_mesh, fluence_mesh = np.meshgrid(pulse_widths, fluences, indexing="ij")
    gamma_mesh = np.empty_like(pulse_width_mesh)

    for ii, pulse_width in enumerate(tqdm(pulse_widths)):
        for jj, fluence in enumerate(fluences):
            pulse = ion.potentials.SincPulse(pulse_width=pulse_width, fluence=fluence)
            if pulse_type != ion.potentials.SincPulse:
                pulse = pulse_type(
                    pulse_width=pulse_width,
                    fluence=fluence,
                    omega_carrier=pulse.omega_carrier,
                )
            gamma_mesh[ii, jj] = pulse.keldysh_parameter(ionization_potential)

    si.vis.xyz_plot(
        f"2d_pulse_keldysh__{pulse_width_min / asec:3f}as_to_{pulse_width_max / asec:3f}as__{fluence_min / Jcm2:3f}jcm2_to_{fluence_max / Jcm2:3f}jcm2___{pulse_type.__name__}",
        pulse_width_mesh,
        fluence_mesh,
        gamma_mesh,
        x_label=r"Pulse Width $\tau$",
        x_unit="asec",
        y_label=fr"Fluence $H$",
        y_unit="Jcm2",
        y_log_axis=True,
        z_log_axis=True,
        z_lower_limit=0.01,
        z_upper_limit=10,
        z_label=fr"Keldysh Parameter $\gamma$ vs. $\tau$ and $H$ for {pulse_type.__name__}",
        contours=(0.01, 0.1, 0.5, 1),
        contour_kwargs={"colors": "white", "linewidths": 0.5},
        **PLOT_KWARGS,
    )


if __name__ == "__main__":
    with log as logger:
        # monochromatic_keldysh_1d(.01 * eV, 50 * eV, atomic_electric_field * np.array([.01, .05, .1, .5, 1, 5]))

        monochromatic_keldysh_2d(
            0 * eV,
            5 * eV,
            0 * atomic_electric_field,
            0.5 * atomic_electric_field,
            points=300,
        )
        monochromatic_keldysh_2d(
            0 * eV,
            10 * eV,
            0 * atomic_electric_field,
            3 * atomic_electric_field,
            points=300,
        )
        monochromatic_keldysh_2d(
            0 * eV,
            50 * eV,
            0 * atomic_electric_field,
            5 * atomic_electric_field,
            points=300,
        )
        monochromatic_keldysh_2d(
            0 * eV,
            30 * eV,
            0 * atomic_electric_field,
            3 * atomic_electric_field,
            points=300,
        )
        monochromatic_keldysh_2d(
            0 * eV,
            100 * eV,
            0 * atomic_electric_field,
            3 * atomic_electric_field,
            points=300,
        )

        for pulse_type in (
            ion.potentials.SincPulse,
            ion.potentials.GaussianPulse,
            ion.potentials.SechPulse,
        ):
            pulse_keldysh_2d(
                50 * asec,
                1000 * asec,
                0.001 * Jcm2,
                30 * Jcm2,
                pulse_type=pulse_type,
                points=300,
            )
            pulse_keldysh_2d(
                50 * asec,
                1000 * asec,
                0.01 * Jcm2,
                50 * Jcm2,
                pulse_type=pulse_type,
                points=300,
            )
            pulse_keldysh_2d(
                50 * asec,
                1000 * asec,
                0.01 * Jcm2,
                20 * Jcm2,
                pulse_type=pulse_type,
                points=300,
            )
            pulse_keldysh_2d(
                50 * asec,
                5000 * asec,
                0.01 * Jcm2,
                100 * Jcm2,
                pulse_type=pulse_type,
                points=500,
            )
