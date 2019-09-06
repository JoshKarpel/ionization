#!/usr/bin/env python
import datetime
import logging
import os

import numpy as np
import numpy.fft as nfft

import simulacra as si
import simulacra.units as u

import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)
SIM_LIB = os.path.join(OUT_DIR, "SIMLIB")

LOGMAN = si.utils.LogManager("simulacra", "ionization", stdout_level=logging.CRITICAL)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)


def run_from_simlib(spec):
    with LOGMAN:
        sim = si.utils.find_or_init_sim(spec, search_dir=SIM_LIB)

        if sim.status != si.Status.FINISHED:
            sim.run_simulation()

        return sim


def fft_field(field, times):
    dt = np.abs(times[1] - times[0])
    freqs = nfft.fftshift(nfft.fftfreq(len(times), dt))
    df = np.abs(freqs[1] - freqs[0])
    fft = tuple(nfft.fftshift(nfft.fft(nfft.fftshift(field), norm="ortho") / df))

    return freqs, fft


if __name__ == "__main__":
    with LOGMAN as logger:
        pw = 100 * u.asec
        tw = 30 * pw
        tb = 35 * pw

        flu = 1 * u.Jcm2

        window = ion.potentials.LogisticWindow(window_time=tw, window_width=pw / 5)

        uncorrected_cos_pulse = ion.potentials.SincPulse(
            pulse_width=pw, phase=0, fluence=flu, window=window
        )
        uncorrected_sin_pulse = ion.potentials.SincPulse(
            pulse_width=pw, phase=u.pi / 2, fluence=flu, window=window
        )

        spec_kwargs = dict(
            time_initial=-tb,
            time_final=tb,
            time_step=1 * u.asec,
            r_bound=100 * u.bohr_radius,
            r_points=100 * 10,
            l_bound=500,
            use_numeric_eigenstates=False,
            numeric_eigenstate_max_energy=20 * u.eV,
            numeric_eigenstate_max_angular_momentum=10,
            store_data_every=10,
            checkpoints=True,
            checkpoint_dir=SIM_LIB,
            checkpoint_every=datetime.timedelta(minutes=1),
        )

        specs = [
            ion.SphericalHarmonicSpecification(
                "uncorrected_cos",
                electric_potential=uncorrected_cos_pulse,
                electric_potential_dc_correction=False,
                electric_potential_fluence_correction=False,
                **spec_kwargs,
            ),
            ion.SphericalHarmonicSpecification(
                "uncorrected_sin",
                electric_potential=uncorrected_sin_pulse,
                electric_potential_dc_correction=False,
                electric_potential_fluence_correction=False,
                **spec_kwargs,
            ),
            ion.SphericalHarmonicSpecification(
                "dc_corrected_cos",
                electric_potential=uncorrected_cos_pulse,
                electric_potential_dc_correction=True,
                electric_potential_fluence_correction=False,
                **spec_kwargs,
            ),
            ion.SphericalHarmonicSpecification(
                "dc_corrected_sin",
                electric_potential=uncorrected_sin_pulse,
                electric_potential_dc_correction=True,
                electric_potential_fluence_correction=False,
                **spec_kwargs,
            ),
            ion.SphericalHarmonicSpecification(
                "fluence_corrected_cos",
                electric_potential=uncorrected_cos_pulse,
                electric_potential_dc_correction=False,
                electric_potential_fluence_correction=True,
                **spec_kwargs,
            ),
            ion.SphericalHarmonicSpecification(
                "fluence_corrected_sin",
                electric_potential=uncorrected_sin_pulse,
                electric_potential_dc_correction=False,
                electric_potential_fluence_correction=True,
                **spec_kwargs,
            ),
            ion.SphericalHarmonicSpecification(
                "fluence_and_dc_corrected_cos",
                electric_potential=uncorrected_cos_pulse,
                electric_potential_dc_correction=True,
                electric_potential_fluence_correction=True,
                **spec_kwargs,
            ),
            ion.SphericalHarmonicSpecification(
                "fluence_and_dc_corrected_sin",
                electric_potential=uncorrected_sin_pulse,
                electric_potential_dc_correction=True,
                electric_potential_fluence_correction=True,
                **spec_kwargs,
            ),
        ]

        for spec in specs:
            print()
            sim = spec.clone().to_simulation()
            # print(sim.name)
            # print(sim.info())
            print(
                sim.name,
                sim.spec.electric_potential.get_fluence_numeric(sim.times) / u.Jcm2,
                sim.spec.electric_potential.get_electric_field_integral_numeric(
                    sim.times
                )
                / (u.atomic_electric_field * u.atomic_time),
            )
            # print()

        sim = specs[-2].to_simulation()
        print(sim.spec.electric_potential.get_fluence_numeric(sim.times) / u.Jcm2)
        print(sim.spec.electric_potential)
        print(sim.spec.electric_potential.amplitude_correction_ratio)
        # print(sim.spec.electric_potential.electric_potential)
        print(sim.spec.electric_potential.electric_potential[0])
        print(sim.spec.electric_potential.electric_potential[0].fluence / u.Jcm2)
        corrected = sim.spec.electric_potential.get_electric_field_amplitude(sim.times)
        uncorrected = sim.spec.electric_potential.electric_potential.get_electric_field_amplitude(
            sim.times
        )
        print(corrected)
        print(uncorrected)
        print(corrected / uncorrected)
        print(np.sqrt(1.0 / 0.9997937024015667))

        # sims = si.utils.multi_map(run_from_simlib, specs, processes = 2)

        # for sim in sims:
        #     sim.plot_wavefunction_vs_time(**PLOT_KWARGS)
