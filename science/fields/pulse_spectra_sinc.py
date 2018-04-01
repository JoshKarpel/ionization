#!/usr/bin/env python

import logging
import os

import numpy as np
import scipy.integrate as integ
import numpy.fft as nfft

import simulacra as si
import simulacra.units as u

import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

LOGMAN = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.INFO)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)

if __name__ == '__main__':
    with LOGMAN as logger:
        pw = 200 * u.asec
        pulse = ion.potentials.SincPulse(
            pulse_width = pw,
            fluence = 1 * u.Jcm2,
            phase = u.pi / 3,
            window = ion.potentials.SymmetricExponentialWindow(
                window_time = 30 * pw,
                window_width = pw / 5,
            )
        )

        t_bound = 35 * pw
        times = np.linspace(-t_bound, t_bound, int(2 * t_bound / u.asec))
        print('time steps', len(times))

        dc_corrected_pulse = ion.potentials.DC_correct_electric_potential(pulse, times)

        si.vis.xy_plot(
            'electric_field_vs_time',
            times,
            pulse.get_electric_field_amplitude(times),
            dc_corrected_pulse.get_electric_field_amplitude(times),
            dc_corrected_pulse[-1].get_electric_field_amplitude(times),
            line_labels = [
                r'original',
                r'corrected',
                r'correction',
            ],
            line_kwargs = [
                {'linestyle': '-'},
                {'linestyle': '--'},
                {'linestyle': ':'},
            ],
            x_unit = 'asec',
            y_unit = 'atomic_electric_field',
            **PLOT_KWARGS,
        )

        si.vis.xy_plot(
            'electric_field_vs_time_abs_log',
            times,
            np.abs(pulse.get_electric_field_amplitude(times)),
            np.abs(dc_corrected_pulse.get_electric_field_amplitude(times)),
            np.abs(dc_corrected_pulse[-1].get_electric_field_amplitude(times)),
            line_labels = [
                r'original',
                r'corrected',
                r'correction',
            ],
            line_kwargs = [
                {'linestyle': '-'},
                {'linestyle': '--'},
                {'linestyle': ':'},
            ],
            x_unit = 'asec',
            y_unit = 'atomic_electric_field',
            y_log_axis = True,
            y_lower_limit = 1e-15 * u.atomic_electric_field,
            **PLOT_KWARGS,
        )

        dt = np.abs(times[1] - times[0])
        print(f'dt = {u.uround(dt, u.asec)} as')
        freqs = nfft.fftshift(nfft.fftfreq(len(times), dt))
        df = np.abs(freqs[1] - freqs[0])
        print(f'df = {u.uround(df, u.THz)} THz')

        fft = nfft.fftshift(
            nfft.fft(
                nfft.fftshift(
                    pulse.get_electric_field_amplitude(times)
                ),
                norm = 'ortho',
            )
        ) / df
        fft_corrected = nfft.fftshift(
            nfft.fft(
                nfft.fftshift(
                    dc_corrected_pulse.get_electric_field_amplitude(times)
                ),
                norm = 'ortho',
            )
        ) / df

        fft_bound = 6000 * u.THz

        si.vis.xy_plot(
            'ffts_raw',
            freqs,
            np.real(fft),
            np.imag(fft),
            np.real(fft_corrected),
            np.imag(fft_corrected),
            line_kwargs = [
                {'color': 'C0', 'linestyle': '-'},
                {'color': 'C1', 'linestyle': '-'},
                {'color': 'C2', 'linestyle': '--'},
                {'color': 'C3', 'linestyle': '--'},
            ],
            x_unit = 'THz',
            x_lower_limit = -fft_bound,
            x_upper_limit = fft_bound,
            **PLOT_KWARGS,
        )

        si.vis.xy_plot(
            'ffts_raw_zoom',
            freqs,
            np.real(fft),
            np.imag(fft),
            np.real(fft_corrected),
            np.imag(fft_corrected),
            line_kwargs = [
                {'color': 'C0', 'linestyle': '-'},
                {'color': 'C1', 'linestyle': '-'},
                {'color': 'C2', 'linestyle': '--'},
                {'color': 'C3', 'linestyle': '--'},
            ],
            x_unit = 'THz',
            x_lower_limit = -fft_bound / 10,
            x_upper_limit = fft_bound / 10,
            **PLOT_KWARGS,
        )

        fft_fluence = df * u.c * u.epsilon_0 * integ.simps(np.abs(fft) ** 2) / len(times)
        print(f'fft fluence {u.uround(fft_fluence, u.Jcm2, 6)} J/cm^2')
        corrected_fft_fluence = df * u.c * u.epsilon_0 * integ.simps(np.abs(fft_corrected) ** 2) / len(times)
        print(f'corrected fft fluence {u.uround(corrected_fft_fluence, u.Jcm2, 6)} J/cm^2')
