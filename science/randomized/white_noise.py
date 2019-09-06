import logging
import os

import numpy as np
import numpy.random as rand
import numpy.fft as nfft
import scipy.signal as signal

import simulacra as si
from simulacra.units import *

import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

LOGMAN = si.utils.LogManager("simulacra", "ionization", stdout_level=logging.DEBUG)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=5)


# https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


if __name__ == "__main__":
    with LOGMAN as logger:
        pw = 200 * asec

        t_bound = 30

        sinc_pulse = ion.potentials.SincPulse(pulse_width=pw)

        times = np.linspace(-t_bound * pw, t_bound * pw, int(pw / asec) * 2 * t_bound)
        dt = np.abs(times[1] - times[0])

        print(times / asec)
        print(len(times))

        white_noise_field = rand.normal(0, 1, len(times))

        si.vis.xy_plot(
            "white_noise_field",
            times,
            white_noise_field,
            x_unit="asec",
            x_label=r"$t$",
            **PLOT_KWARGS,
        )

        white_noise_autocorrelation = np.correlate(
            white_noise_field, white_noise_field, "same"
        )

        si.vis.xy_plot(
            "white_noise_autocorrelation",
            times,
            white_noise_autocorrelation,
            x_unit="asec",
            x_label=r"$\Delta t$",
            **PLOT_KWARGS,
        )

        # sample_rate = len(times) / np.abs(times[-1] - times[0])
        # lowcut = sinc_pulse.frequency_min
        # highcut = sinc_pulse.frequency_max
        #
        # print(sample_rate / (1 / asec))
        # print(lowcut / THz, highcut / THz)
        #
        # b, a = butter_bandpass(lowcut, highcut, sample_rate, order = 6)
        # w, h = signal.freqz(b, a, worN = 2000)
        # si.vis.xy_plot(
        #     'filter_response',
        #     sample_rate * w / twopi,
        #     np.abs(h),
        #     x_unit = 'THz', x_label = r'$f$',
        #     x_upper_limit = 50000 * THz,
        #     y_label = 'Gain',
        #     **PLOT_KWARGS,
        # )

        # filtered_field = butter_bandpass_filter(white_noise_field, lowcut, highcut, sample_rate, order = 6)

        fft_freq = nfft.fftfreq(len(times), d=dt)
        fft_white_noise = nfft.fft(white_noise_field, norm="ortho")

        lowcut = sinc_pulse.frequency_min
        highcut = sinc_pulse.frequency_max
        keep_indices = np.greater_equal(np.abs(fft_freq), lowcut) * np.less_equal(
            np.abs(fft_freq), highcut
        )

        si.vis.xy_plot(
            "white_noise_fft",
            nfft.fftshift(fft_freq),
            nfft.fftshift(np.real(fft_white_noise)),
            nfft.fftshift(np.imag(fft_white_noise)),
            nfft.fftshift(np.abs(fft_white_noise)),
            nfft.fftshift(keep_indices),
            line_labels=["real", "imag", "abs", "filter"],
            x_unit="THz",
            x_label=r"$f$",
            **PLOT_KWARGS,
        )
        si.vis.xy_plot(
            "white_noise_fft_zoom",
            nfft.fftshift(fft_freq),
            nfft.fftshift(np.real(fft_white_noise)),
            nfft.fftshift(np.imag(fft_white_noise)),
            nfft.fftshift(np.abs(fft_white_noise)),
            nfft.fftshift(keep_indices),
            line_labels=["real", "imag", "abs", "filter"],
            x_unit="THz",
            x_label=r"$f$",
            x_lower_limit=-1.5 * highcut,
            x_upper_limit=1.5 * highcut,
            **PLOT_KWARGS,
        )

        fft_filtered = np.where(keep_indices, fft_white_noise, 0)
        filtered_field = nfft.ifft(fft_filtered, norm="ortho")
        filtered_autocorrelation = np.correlate(filtered_field, filtered_field, "same")

        si.vis.xy_plot(
            "filtered_field",
            times,
            white_noise_field,
            filtered_field,
            line_labels=["white", "filtered"],
            x_unit="asec",
            x_label=r"$t$",
            **PLOT_KWARGS,
        )

        si.vis.xy_plot(
            "filtered_autocorrelation",
            times,
            white_noise_autocorrelation,
            filtered_autocorrelation,
            line_labels=["white", "filtered"],
            x_unit="asec",
            x_label=r"$\Delta t$",
            **PLOT_KWARGS,
        )

        si.vis.xy_plot(
            "filtered_autocorrelation__just_filtered",
            times,
            # white_noise_autocorrelation,
            filtered_autocorrelation,
            # line_labels = ['white', 'filtered'],
            x_unit="asec",
            x_label=r"$\Delta t$",
            **PLOT_KWARGS,
        )
