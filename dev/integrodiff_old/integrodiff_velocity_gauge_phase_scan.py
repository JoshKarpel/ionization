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
    "simulacra", "ionization", stdout_logs=True, stdout_level=logging.INFO
)


def run(spec):
    with log as logger:
        sim = spec.to_sim()

        sim.run()

        sim.info().log()

        sim.plot_b2_vs_time(log=True, target_dir=OUT_DIR)

    return sim


if __name__ == "__main__":
    with log as logger:
        # pulse_widths = np.linspace(50, 600, 50) * asec
        pw = 400 * asec
        phases = np.linspace(0, pi, 50)
        t_bound = 5

        flu = 10 * Jcm2

        L = bohr_radius
        m = electron_mass
        q = electron_charge

        prefactor = -((q / m) ** 2) / (4 * (L ** 2))
        tau_alpha = 2 * m * (L ** 2) / hbar

        specs = []
        for phase in phases:
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
                ide.VelocityGaugeIntegroDifferentialEquationSpecification(
                    f"{efield.__class__.__name__}_pw={pw / asec:3f}as_flu={flu / Jcm2:3f}Jcm2_phase={phase:.3f}",
                    time_initial=-t_bound * pw,
                    time_final=t_bound * pw,
                    time_step=0.5 * asec,
                    electric_potential=efield,
                    prefactor=prefactor,
                    kernel=ide.gaussian_kernel_VEL,
                    kernel_kwargs=dict(tau_alpha=tau_alpha, width=L),
                    evolution_method="RK4",
                    pulse_width=pw,
                    phase=phase,
                    flu=flu,
                )
            )

        PLOT_KWARGS = dict(target_dir=OUT_DIR)

        results = si.utils.multi_map(run, specs, processes=2)

        for log in (True, False):
            if not log:
                y_lower_limit = 0
            else:
                y_lower_limit = None

            si.vis.xy_plot(
                f"ionization_vs_phase__pw={pw / asec:3f}as_flu={flu / Jcm2:3f}Jcm2__log={log}",
                [r.spec.phase for r in results],
                [np.abs(r.a[-1]) ** 2 for r in results],
                x_label=r"CEP $\varphi$ ($\pi$)",
                x_unit="rad",
                y_label=r"$  \left| a_{\mathrm{final}} \right|^2  $",
                y_log_axis=log,
                y_upper_limit=1,
                y_lower_limit=y_lower_limit,
                **PLOT_KWARGS,
            )

            si.vis.xy_plot(
                f"ionization_vs_phase__pw={pw / asec:3f}as_flu={flu / Jcm2:3f}Jcm2__log={log}__rel",
                [r.spec.phase for r in results],
                [
                    (np.abs(r.a[-1]) ** 2) / (np.abs(results[0].a[-1]) ** 2)
                    for r in results
                ],
                x_label=r"CEP $\varphi$ ($\pi$)",
                x_unit="rad",
                y_label=r"$  \left| a_{\mathrm{final}}(\varphi) \right|^2 / \left| a_{\mathrm{final}}(\varphi = 0) \right|^2  $",
                y_log_axis=log,
                **PLOT_KWARGS,
            )
