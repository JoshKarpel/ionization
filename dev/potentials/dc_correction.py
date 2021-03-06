import logging
import os

import numpy as np
import scipy.optimize as optimize
import simulacra as si
from simulacra.units import *

import ionization as ion


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=3)

EA_FIELD_PLOT_KWARGS = dict(
    line_labels=(fr"${str_efield}(t)$", fr"$q{str_afield}(t)$"),
    x_label=r"$t$",
    x_unit="asec",
    y_label=fr"${str_efield}(t), \, q{str_afield}(t)$",
)

EA_LOG_PLOT_KWARGS = dict(y_log_axis=True, y_upper_limit=2, y_lower_limit=1e-20)

if __name__ == "__main__":
    with si.utils.LogManager(
        "simulacra", "ionization", stdout_level=logging.DEBUG
    ) as logger:
        pw = 100 * asec

        window = ion.potentials.LogisticWindow(
            window_time=25 * pw, window_width=0.2 * pw
        ) + ion.potentials.RectangularTimeWindow(on_time=-31 * pw, off_time=31 * pw)

        ref_sinc = ion.potentials.SincPulse(pulse_width=pw)
        print(ref_sinc)

        pulse = ion.potentials.SincPulse(
            pulse_width=pw,
            fluence=1 * Jcm2,
            phase=0,
            # omega_carrier = ref_sinc.omega_carrier,
            window=window,
        )

        print(pulse)

        t = np.linspace(-40 * pw, 40 * pw, 1e5)
        total_t = np.abs(t[-1] - t[0])

        uncorrected_pulse_amp = pulse.get_electric_field_amplitude(t)
        uncorrected_pulse_vpot = proton_charge * (
            -pulse.get_electric_field_integral_numeric_cumulative(t)
        )

        print("uncorrected A final:", uncorrected_pulse_vpot[-1] / atomic_momentum)

        si.vis.xy_plot(
            f"uncorrected_pulse",
            t,
            uncorrected_pulse_amp / atomic_electric_field,
            uncorrected_pulse_vpot / atomic_momentum,
            **EA_FIELD_PLOT_KWARGS,
            **PLOT_KWARGS,
        )

        si.vis.xy_plot(
            f"uncorrected_pulse__log",
            t,
            np.abs(uncorrected_pulse_amp / atomic_electric_field),
            np.abs(uncorrected_pulse_vpot / atomic_momentum),
            **EA_FIELD_PLOT_KWARGS,
            **EA_LOG_PLOT_KWARGS,
            **PLOT_KWARGS,
        )

        ### CORRECTION 1 ###

        correction_field = ion.potentials.Rectangle(
            start_time=t[0],
            end_time=t[-1],
            amplitude=-pulse.get_electric_field_integral_numeric_cumulative(t)[-1]
            / total_t,
        )
        print(correction_field)

        corrected_pulse = pulse + correction_field
        print(corrected_pulse)

        corrected_pulse_amp = corrected_pulse.get_electric_field_amplitude(t)
        corrected_pulse_vpot = proton_charge * (
            -corrected_pulse.get_electric_field_integral_numeric_cumulative(t)
        )

        print("rect-corrected A final:", corrected_pulse_vpot[-1] / atomic_momentum)

        si.vis.xy_plot(
            f"rect-corrected_pulse",
            t,
            corrected_pulse_amp / atomic_electric_field,
            corrected_pulse_vpot / atomic_momentum,
            **EA_FIELD_PLOT_KWARGS,
            **PLOT_KWARGS,
        )

        si.vis.xy_plot(
            f"rect-corrected_pulse__log",
            t,
            np.abs(corrected_pulse_amp / atomic_electric_field),
            np.abs(corrected_pulse_vpot / atomic_momentum),
            **EA_FIELD_PLOT_KWARGS,
            **EA_LOG_PLOT_KWARGS,
            **PLOT_KWARGS,
        )

        ### CORRECTION 2 ###

        def func_to_minimize(amp, original_pulse):
            test_correction_field = ion.potentials.Rectangle(
                start_time=t[0],
                end_time=t[-1],
                amplitude=amp,
                window=original_pulse.window,
            )
            test_pulse = original_pulse + test_correction_field

            return np.abs(
                test_pulse.get_electric_field_integral_numeric_cumulative(t)[-1]
            )

        correction_amp = optimize.minimize_scalar(func_to_minimize, args=(pulse,))

        print(correction_amp)

        correction_field = ion.potentials.Rectangle(
            start_time=t[0],
            end_time=t[-1],
            amplitude=correction_amp.x,
            window=pulse.window,
        )
        print(correction_field)

        corrected_pulse = pulse + correction_field
        print(corrected_pulse)

        corrected_pulse_amp = corrected_pulse.get_electric_field_amplitude(t)
        corrected_pulse_vpot = proton_charge * (
            -corrected_pulse.get_electric_field_integral_numeric_cumulative(t)
        )

        print("opt-rect-corrected A final:", corrected_pulse_vpot[-1] / atomic_momentum)

        si.vis.xy_plot(
            f"opt-rect-corrected_pulse",
            t,
            corrected_pulse_amp / atomic_electric_field,
            corrected_pulse_vpot / atomic_momentum,
            **EA_FIELD_PLOT_KWARGS,
            **PLOT_KWARGS,
        )

        si.vis.xy_plot(
            f"opt-rect-corrected_pulse__log",
            t,
            np.abs(corrected_pulse_amp / atomic_electric_field),
            np.abs(corrected_pulse_vpot / atomic_momentum),
            **EA_FIELD_PLOT_KWARGS,
            **EA_LOG_PLOT_KWARGS,
            **PLOT_KWARGS,
        )
