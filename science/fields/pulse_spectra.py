import logging
import os

import numpy as np
import numpy.fft as fft
import scipy.integrate as integ

import simulacra as si
from simulacra.units import *

import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

logman = si.utils.LogManager(
    'simulacra', 'ionization',
    stdout_level = logging.INFO
)

PLT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 3,
)

PULSE_TYPES = (
    ion.SincPulse,
    ion.GaussianPulse,
    ion.SechPulse,
)

if __name__ == '__main__':
    with logman as logger:
        t_bound = 100
        p_bound = 30
        freq_window = 8000 * THz

        pw = 200 * asec
        flu = 1 * Jcm2

        times = np.linspace(-t_bound * pw, t_bound * pw, 2 ** 14)
        dt = np.abs(times[1] - times[0])

        dummy = ion.SincPulse(pulse_width = pw, fluence = flu, phase = 0,
                              window = ion.SymmetricExponentialTimeWindow(window_time = p_bound * pw, window_width = .2 * pw))
        pulses = [
            dummy,
            ion.GaussianPulse(pulse_width = pw, fluence = flu, omega_carrier = dummy.omega_carrier, phase = dummy.phase,
                              window = dummy.window),
            ion.SechPulse(pulse_width = pw, fluence = flu, omega_carrier = dummy.omega_carrier, phase = dummy.phase,
                          window = dummy.window),
        ]

        fields = tuple(pulse.get_electric_field_amplitude(times) for pulse in pulses)

        si.vis.xy_plot(
            'efields_vs_time',
            times,
            *fields,
            line_labels = (pulse.__class__.__name__ for pulse in pulses),
            x_label = r'$ t $', x_unit = 'asec',
            y_label = fr'$ {ion.LATEX_EFIELD}(t) $', y_unit = 'atomic_electric_field',
            title = fr'Electric Fields at $\tau = {uround(pw, asec)} \, \mathrm{{as}}, \, H = {uround(flu, Jcm2)} \, \mathrm{{J/cm^2}}$',
            **PLT_KWARGS,
        )

        freqs = fft.fftshift(fft.fftfreq(len(times), dt))
        df = np.abs(freqs[1] - freqs[0])
        # ffts = tuple(fft.fftshift(fft.fft(field) / df) for field in fields)
        ffts = tuple(fft.fftshift(fft.fft(field, norm = 'ortho') / df) for field in fields)

        si.vis.xy_plot(
            'amplitude_spectra',
            freqs,
            *ffts,
            line_labels = (pulse.__class__.__name__ for pulse in pulses),
            x_label = r'$ f $', x_unit = 'THz',
            x_lower_limit = -freq_window, x_upper_limit = freq_window,
            y_label = fr'$ {ion.LATEX_EFIELD}(f) $',
            title = fr'Amplitude Spectra at $\tau = {uround(pw, asec)} \, \mathrm{{as}}, \, H = {uround(flu, Jcm2)} \, \mathrm{{J/cm^2}}$',
            **PLT_KWARGS,
        )

        si.vis.xy_plot(
            'abs_amplitude_spectra',
            freqs,
            *(np.abs(f) for f in ffts),
            line_labels = (pulse.__class__.__name__ for pulse in pulses),
            x_label = r'$ f $', x_unit = 'THz',
            x_lower_limit = -freq_window, x_upper_limit = freq_window,
            y_label = fr'$ \left| {ion.LATEX_EFIELD}(f) \right| $ ($\mathrm{{a.u. / THz}}$)',
            y_unit = atomic_electric_field / THz,
            title = fr'Amplitude Spectra at $\tau = {uround(pw, asec)} \, \mathrm{{as}}, \, H = {uround(flu, Jcm2)} \, \mathrm{{J/cm^2}}$',
            **PLT_KWARGS,
        )

        si.vis.xy_plot(
            'power_spectra',
            freqs,
            *(epsilon_0 * c * (np.abs(f) ** 2) / len(times) for f in ffts),
            line_labels = (pulse.__class__.__name__ for pulse in pulses),
            x_label = r'$ f $', x_unit = 'THz',
            x_lower_limit = -freq_window, x_upper_limit = freq_window,
            y_label = fr'$ \left| {ion.LATEX_EFIELD}(f) \right|^2 $  ($\mathrm{{J / cm^2 / THz}}$)',
            y_unit = Jcm2 / THz,
            title = fr'Power Density Spectra at $\tau = {uround(pw, asec)} \, \mathrm{{as}}, \, H = {uround(flu, Jcm2)} \, \mathrm{{J/cm^2}}$',
            **PLT_KWARGS,
        )

        for pulse, f in zip(pulses, ffts):
            print(pulse.__class__.__name__)
            print(pulse.get_fluence_numeric(times) / Jcm2)
            print(pulse.fluence / Jcm2)
            print(integ.simps(y = epsilon_0 * c * (np.abs(f) ** 2) / len(times),
                              x = freqs) / Jcm2)

            print('-' * 80)

        print(len(times))
