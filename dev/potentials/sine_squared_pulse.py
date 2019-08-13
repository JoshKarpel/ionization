import logging
import os

import numpy as np
import numpy.fft as nfft
import simulacra as si
import simulacra.units as u

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)

if __name__ == "__main__":
    with si.utils.LogManager(
        "simulacra", "ionization", stdout_logs=True, stdout_level=logging.INFO
    ):
        # nc < 2 has negative-frequency components
        nc = 2
        pulse = potentials.CosSquaredPulse(
            amplitude=1 * u.atomic_electric_field,
            wavelength=800 * u.nm,
            number_of_cycles=nc,
        )

        times = np.linspace(-5 * pulse.total_time, 5 * pulse.total_time, 1e5)

        si.vis.xy_plot(
            "cos_squared_pulse",
            times,
            pulse.get_electric_field_envelope(times),
            pulse.get_electric_field_amplitude(times) / pulse.amplitude,
            pulse.get_vector_potential_amplitude_numeric_cumulative(times)
            * u.proton_charge
            / u.atomic_momentum,
            line_labels=["Envelope", "Pulse", "Vector Potential"],
            x_unit="asec",
            x_label=r"Time $t$",
            y_label=r"Fields",
            legend_kwargs={"loc": "lower left"},
            **PLOT_KWARGS,
        )

        dt = np.abs(times[1] - times[0])
        print(f"dt = {u.uround(dt, u.asec)} as")
        freqs = nfft.fftshift(nfft.fftfreq(len(times), dt))
        df = np.abs(freqs[1] - freqs[0])
        print(f"df = {u.uround(df, u.THz)} THz")

        fft = (
            nfft.fftshift(
                nfft.fft(
                    nfft.fftshift(pulse.get_electric_field_amplitude(times)),
                    norm="ortho",
                )
            )
            / df
        )

        fft_bound = 5 * pulse.frequency_carrier
        si.vis.xy_plot(
            "ffts_raw",
            freqs,
            np.real(fft),
            np.imag(fft),
            line_labels=["real", "imag"],
            line_kwargs=[
                {"color": "C0", "linestyle": "-"},
                {"color": "C1", "linestyle": "--"},
            ],
            x_unit="THz",
            x_label=r"$f$",
            x_lower_limit=-fft_bound,
            x_upper_limit=fft_bound,
            vlines=[-pulse.frequency_carrier, pulse.frequency_carrier],
            vline_kwargs=[
                {"color": "black", "alpha": 0.5, "linestyle": ":"},
                {"color": "black", "alpha": 0.5, "linestyle": ":"},
            ],
            **PLOT_KWARGS,
        )
