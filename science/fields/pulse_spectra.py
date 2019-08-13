import logging
import os

import numpy as np
import numpy.fft as nfft
import scipy.integrate as integ

import simulacra as si
from simulacra.units import *

import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

logman = si.utils.LogManager("simulacra", "ionization", stdout_level=logging.INFO)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=3)

PULSE_TYPES = (ion.SincPulse, ion.GaussianPulse, ion.SechPulse)

if __name__ == "__main__":
    with logman as logger:
        t_bound = 400
        p_bound = 30
        freq_window = 10000 * THz

        pw = 200 * asec
        flu = 1 * Jcm2
        phase = pi / 2

        times = np.linspace(-t_bound * pw, t_bound * pw, 2 ** 14)
        dt = np.abs(times[1] - times[0])

        pulses = [
            ion.SincPulse.from_omega_min(
                pulse_width=pw,
                fluence=flu,
                phase=phase,
                window=ion.SymmetricExponentialTimeWindow(
                    window_time=30 * pw, window_width=0.2 * pw
                ),
            ),
            ion.GaussianPulse(
                pulse_width=pw,
                fluence=flu,
                phase=phase,
                window=ion.SymmetricExponentialTimeWindow(
                    window_time=5 * pw, window_width=0.2 * pw
                ),
            ),
            # ion.SechPulse(pulse_width = pw, fluence = flu, omega_carrier = dummy.omega_carrier, phase = dummy.phase,
            #               window = dummy.window),
        ]

        # pulses = list(ion.DC_correct_electric_potential(pulse, times) for pulse in pulses)
        fields = tuple(pulse.get_electric_field_amplitude(times) for pulse in pulses)

        # gaussian_max = np.nanmax(np.abs(fields[1]))
        # gaussian_fwhm_time = pulses[1].time_fwhm
        # gaussian_hwhm_time = gaussian_fwhm_time / 2

        si.vis.xy_plot(
            "efields_vs_time",
            times,
            *fields,
            line_labels=(pulse.__class__.__name__ for pulse in pulses),
            x_label=r"$ t $",
            x_unit="asec",
            y_label=fr"$ {ion.LATEX_EFIELD}(t) $",
            y_unit="atomic_electric_field",
            x_lower_limit=-p_bound * pw,
            x_upper_limit=p_bound * pw,
            title=fr"Electric Fields at $\tau = {uround(pw, asec)} \, \mathrm{{as}}, \, H = {uround(flu, Jcm2)} \, \mathrm{{J/cm^2}}$",
            # hlines = (gaussian_max, gaussian_max / 2), hline_kwargs = ({'linestyle': '--'}, {'linestyle': '--'}),
            # vlines = (-gaussian_hwhm_time, gaussian_hwhm_time), vline_kwargs = ({'linestyle': '--'}, {'linestyle': '--'}),
            **PLOT_KWARGS,
        )

        freqs = nfft.fftshift(nfft.fftfreq(len(times), dt))
        df = np.abs(freqs[1] - freqs[0])
        ffts = tuple(
            nfft.fftshift(nfft.fft(nfft.fftshift(field), norm="ortho") / df)
            for field in fields
        )

        si.vis.xy_plot(
            "amplitude_spectra_real",
            freqs,
            *(np.real(fft) for fft in ffts),
            line_labels=(pulse.__class__.__name__ for pulse in pulses),
            x_label=r"$ f $",
            x_unit="THz",
            x_lower_limit=-freq_window,
            x_upper_limit=freq_window,
            y_label=fr"$ {ion.LATEX_EFIELD}(f) $",
            title=fr"Real Amplitude Spectra at $\tau = {uround(pw, asec)} \, \mathrm{{as}}, \, H = {uround(flu, Jcm2)} \, \mathrm{{J/cm^2}}$",
            **PLOT_KWARGS,
        )
        si.vis.xy_plot(
            "amplitude_spectra_imag",
            freqs,
            *(np.imag(fft) for fft in ffts),
            line_labels=(pulse.__class__.__name__ for pulse in pulses),
            x_label=r"$ f $",
            x_unit="THz",
            x_lower_limit=-freq_window,
            x_upper_limit=freq_window,
            y_label=fr"$ {ion.LATEX_EFIELD}(f) $",
            title=fr"Imaginary Amplitude Spectra at $\tau = {uround(pw, asec)} \, \mathrm{{as}}, \, H = {uround(flu, Jcm2)} \, \mathrm{{J/cm^2}}$",
            **PLOT_KWARGS,
        )
        si.vis.xy_plot(
            "amplitude_spectra_real_zoom",
            freqs,
            *(np.real(fft) for fft in ffts),
            line_labels=(pulse.__class__.__name__ for pulse in pulses),
            x_label=r"$ f $",
            x_unit="THz",
            # x_lower_limit = -freq_window, x_upper_limit = freq_window,
            y_label=fr"$ {ion.LATEX_EFIELD}(f) $",
            title=fr"Real Amplitude Spectra at $\tau = {uround(pw, asec)} \, \mathrm{{as}}, \, H = {uround(flu, Jcm2)} \, \mathrm{{J/cm^2}}$",
            x_lower_limit=2450 * THz,
            x_upper_limit=2550 * THz,
            **PLOT_KWARGS,
        )

        # gaussian_max_freq = np.nanmax(np.abs(ffts[1]))
        # gaussian_center_freq= pulses[1].frequency_carrier
        # gaussian_fwhm_freq = pulses[1].frequency_fwhm
        # gaussian_hwhm_freq = gaussian_fwhm_freq / 2

        si.vis.xy_plot(
            "abs_amplitude_spectra",
            freqs,
            *(np.abs(f) for f in ffts),
            line_labels=(pulse.__class__.__name__ for pulse in pulses),
            x_label=r"$ f $",
            x_unit="THz",
            x_lower_limit=-freq_window,
            x_upper_limit=freq_window,
            y_label=fr"$ \left| {ion.LATEX_EFIELD}(f) \right| $ ($\mathrm{{a.u. / THz}}$)",
            y_unit=atomic_electric_field / THz,
            title=fr"Amplitude Spectra at $\tau = {uround(pw, asec)} \, \mathrm{{as}}, \, H = {uround(flu, Jcm2)} \, \mathrm{{J/cm^2}}$",
            # hlines = (gaussian_max_freq, gaussian_max_freq / 2), hline_kwargs = ({'linestyle': '--'}, {'linestyle': '--'}),
            # vlines = (-gaussian_hwhm_freq + gaussian_center_freq, gaussian_hwhm_freq + gaussian_center_freq), vline_kwargs = ({'linestyle': '--'}, {'linestyle': '--'}),
            **PLOT_KWARGS,
        )

        # gaussian_max_power = np.nanmax(np.abs(epsilon_0 * c * (np.abs(ffts[1]) ** 2) / len(times)))

        si.vis.xy_plot(
            "power_spectra",
            freqs,
            *(epsilon_0 * c * (np.abs(f) ** 2) / len(times) for f in ffts),
            line_labels=(pulse.__class__.__name__ for pulse in pulses),
            x_label=r"$ f $",
            x_unit="THz",
            x_lower_limit=-freq_window,
            x_upper_limit=freq_window,
            y_label=fr"$ \left| {ion.LATEX_EFIELD}(f) \right|^2 $  ($\mathrm{{J / cm^2 / THz}}$)",
            y_unit=Jcm2 / THz,
            title=fr"Power Density Spectra at $\tau = {uround(pw, asec)} \, \mathrm{{as}}, \, H = {uround(flu, Jcm2)} \, \mathrm{{J/cm^2}}$",
            # hlines = (gaussian_max_power, gaussian_max_power / 2), hline_kwargs = ({'linestyle': '--'}, {'linestyle': '--'}),
            # vlines = (-gaussian_hwhm_freq / np.sqrt(2) + gaussian_center_freq, gaussian_hwhm_freq / np.sqrt(2) + gaussian_center_freq), vline_kwargs = ({'linestyle': '--'}, {'linestyle': '--'}),
            **PLOT_KWARGS,
        )

        for pulse, f in zip(pulses, ffts):
            print(pulse)
            print(pulse.get_fluence_numeric(times) / Jcm2)
            # print(pulse.fluence / Jcm2)
            print(
                integ.simps(y=epsilon_0 * c * (np.abs(f) ** 2) / len(times), x=freqs)
                / Jcm2
            )

            print("-" * 80)

        print(len(times))

        for pulse in pulses:
            print(pulse.info())
