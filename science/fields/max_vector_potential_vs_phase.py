import logging
import os

from tqdm import tqdm

import numpy as np
import simulacra as si
from simulacra.units import *

import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

LOGMAN = si.utils.LogManager("simulacra", "ionization", stdout_level=logging.DEBUG)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)

PULSE_TYPES = (
    ion.SincPulse,
    ion.GaussianPulse,
    # ion.SechPulse,
)

if __name__ == "__main__":
    with LOGMAN as logger:
        t_bound = 35
        p_bound = 30

        pw = 200 * asec
        flu = 1 * Jcm2
        phases = np.linspace(0, twopi, 1e3)

        times = np.linspace(-t_bound * pw, t_bound * pw, 2 ** 14)
        dt = np.abs(times[1] - times[0])

        window = window = ion.SymmetricExponentialTimeWindow(
            window_time=p_bound * pw, window_width=0.2 * pw
        )

        max_vector_potential = {
            pulse_type: np.empty_like(phases) for pulse_type in PULSE_TYPES
        }
        avg_abs_vector_potential = {
            pulse_type: np.empty_like(phases) for pulse_type in PULSE_TYPES
        }
        rms_vector_potential = {
            pulse_type: np.empty_like(phases) for pulse_type in PULSE_TYPES
        }

        for pulse_type in PULSE_TYPES:
            for ii, phase in enumerate(tqdm(phases)):
                pulse = pulse_type.from_omega_min(
                    pulse_width=pw, fluence=flu, phase=phase, window=window
                )
                pulse = ion.DC_correct_electric_potential(pulse, times)

                vp = pulse.get_vector_potential_amplitude_numeric_cumulative(times)
                max_vector_potential[pulse_type][ii] = np.max(np.abs(vp))
                avg_abs_vector_potential[pulse_type][ii] = np.mean(np.abs(vp))
                rms_vector_potential[pulse_type][ii] = np.sqrt(np.mean(vp ** 2))

        # MAX ABS

        si.vis.xy_plot(
            "max_vector_potential_vs_phase",
            phases,
            *(proton_charge * max_vp for pulse, max_vp in max_vector_potential.items()),
            line_labels=(pulse.__name__ for pulse in max_vector_potential),
            x_label=r"$ \varphi $",
            x_unit="rad",
            y_label=rf"$ \max_t \; \left|{ion.LATEX_AFIELD}_{{\varphi}}(t)\right| $",
            y_unit="atomic_momentum",
            **PLOT_KWARGS,
        )

        si.vis.xy_plot(
            "max_vector_potential_vs_phase__rel",
            phases,
            *(
                max_vp / max_vector_potential[pulse][0]
                for pulse, max_vp in max_vector_potential.items()
            ),
            line_labels=(pulse.__name__ for pulse in max_vector_potential),
            x_label=r"$ \varphi $",
            x_unit="rad",
            y_label=rf"$ \max_t \; \left|{ion.LATEX_AFIELD}_{{\varphi}}(t)\right| / \max_t \left|{ion.LATEX_AFIELD}_{{\varphi = 0}}(t)\right| $",
            **PLOT_KWARGS,
        )

        # AVG ABS

        si.vis.xy_plot(
            "avg_abs_vector_potential_vs_phase",
            phases,
            *(
                proton_charge * avg_abs_vp
                for pulse, avg_abs_vp in avg_abs_vector_potential.items()
            ),
            line_labels=(pulse.__name__ for pulse in avg_abs_vector_potential),
            x_label=r"$ \varphi $",
            x_unit="rad",
            y_label=rf"$ \mathrm{{avg}}_t \; \left|{ion.LATEX_AFIELD}_{{\varphi}}(t)\right| $",
            y_unit="atomic_momentum",
            **PLOT_KWARGS,
        )

        si.vis.xy_plot(
            "avg_abs_vector_potential_vs_phase__rel",
            phases,
            *(
                avg_abs_vp / avg_abs_vector_potential[pulse][0]
                for pulse, avg_abs_vp in avg_abs_vector_potential.items()
            ),
            line_labels=(pulse.__name__ for pulse in avg_abs_vector_potential),
            x_label=r"$ \varphi $",
            x_unit="rad",
            y_label=rf"$ \mathrm{{avg}}_t \; \left|{ion.LATEX_AFIELD}_{{\varphi}}(t)\right| / \mathrm{{avg}}_t \left|{ion.LATEX_AFIELD}_{{\varphi = 0}}(t)\right| $",
            **PLOT_KWARGS,
        )

        # RMS

        si.vis.xy_plot(
            "rms_vector_potential_vs_phase",
            phases,
            *(proton_charge * rms_vp for pulse, rms_vp in rms_vector_potential.items()),
            line_labels=(pulse.__name__ for pulse in rms_vector_potential),
            x_label=r"$ \varphi $",
            x_unit="rad",
            y_label=rf"$ \mathrm{{RMS}}_t \; \left|{ion.LATEX_AFIELD}_{{\varphi}}(t)\right| $",
            y_unit="atomic_momentum",
            **PLOT_KWARGS,
        )

        si.vis.xy_plot(
            "rms_vector_potential_vs_phase__rel",
            phases,
            *(
                rms_vp / rms_vector_potential[pulse][0]
                for pulse, rms_vp in rms_vector_potential.items()
            ),
            line_labels=(pulse.__name__ for pulse in rms_vector_potential),
            x_label=r"$ \varphi $",
            x_unit="rad",
            y_label=rf"$ \mathrm{{RMS}}_t \; \left|{ion.LATEX_AFIELD}_{{\varphi}}(t)\right| / \mathrm{{RMS}}_t \left|{ion.LATEX_AFIELD}_{{\varphi = 0}}(t)\right| $",
            **PLOT_KWARGS,
        )
