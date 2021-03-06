import logging
import os

import numpy as np
import simulacra as si
from simulacra.units import *

import ionization as ion
import ide as ide

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

log = si.utils.LogManager(
    "simulacra", "ionization", stdout_logs=True, stdout_level=logging.DEBUG
)


def run(spec):
    with log as logger:
        sim = spec.to_sim()
        sim.run()
        sim.info().log()

        sim.plot_b2_vs_time(target_dir=OUT_DIR)

    return sim


if __name__ == "__main__":
    with log as logger:
        pulse_widths = np.linspace(50, 600, 200) * asec
        t_bound = 5

        flu = 10 * Jcm2
        phase = 0

        L = bohr_radius
        m = electron_mass
        q = electron_charge

        prefactor = -((q / m) ** 2) / (4 * (L ** 2))
        tau_alpha = 2 * m * (L ** 2) / hbar

        specs = []
        for pw in pulse_widths:
            reference_sinc = ion.potentials.SincPulse(pulse_width=pw)
            efield = ion.potentials.GaussianPulse(
                pulse_width=pw,
                fluence=flu,
                phase=phase,
                omega_carrier=reference_sinc.omega_carrier,
                window=ion.potentials.LogisticWindow(
                    window_time=(t_bound * 0.9) * pw, window_width=0.2 * pw
                ),
            )

            specs.append(
                ide.potentials.VelocityGaugeIntegroDifferentialEquationSpecification(
                    f"{efield.__class__.__name__}_pw={pw / asec:3f}as_flu={flu / Jcm2:3f}Jcm2",
                    time_initial=-t_bound * pw,
                    time_final=t_bound * pw,
                    time_step=0.1 * asec,
                    electric_potential=efield,
                    prefactor=prefactor,
                    kernel=ide.gaussian_kernel_VEL,
                    kernel_kwargs=dict(tau_alpha=tau_alpha, width=L),
                    evolution_method="TRAP",
                    pulse_width=pw,
                    phase=phase,
                    flu=flu,
                )
            )

        PLOT_KWARGS = dict(target_dir=OUT_DIR)

        results = si.utils.multi_map(run, specs, processes=4)

        for log in (True, False):
            si.vis.xy_plot(
                f"ion.potentials.zation_vs_Pulse_width__flu={flu / Jcm2:3f}Jcm2_phase={phase:.3f}__log={log}",
                [r.spec.pulse_width for r in results],
                [np.abs(r.a[-1]) ** 2 for r in results],
                x_label=r"Pulse Width $  \tau  $",
                x_unit="asec",
                y_label=r"$  \left| a_{\mathrm{final}} \right|^2  $",
                y_log_axis=log,
                y_upper_limit=1,
                **PLOT_KWARGS,
            )
