import itertools as it
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

        return sim


if __name__ == "__main__":
    with log as logger:
        pw = 100 * asec
        t_bound = 10

        L = bohr_radius
        m = electron_mass
        q = electron_charge

        sinc = ion.potentials.SincPulse(pulse_width=pw)
        efield = ion.potentials.GaussianPulse(
            pulse_width=pw, fluence=20 * Jcm2, omega_carrier=sinc.omega_carrier
        )
        # efield = ion.potentials.SincPulse(pulse_width = pw, fluence = 1 * Jcm2)

        # efield = ion.potentials.Rectangle(start_time = -100 * asec, end_time = 100 * asec, amplitude = .01 * atomic_electric_field)

        prefactor = -((q / m) ** 2) / (4 * (L ** 2))
        tau_alpha = 2 * m * (L ** 2) / hbar

        specs = []
        for method in ("FE", "BE", "TRAP", "RK4"):
            specs.append(
                ide.VelocityGaugeIntegroDifferentialEquationSpecification(
                    method,
                    time_initial=-t_bound * pw,
                    time_final=t_bound * pw,
                    time_step=1 * asec,
                    electric_potential=efield,
                    prefactor=prefactor,
                    kernel=ide.gaussian_kernel_VEL,
                    kernel_kwargs=dict(tau_alpha=tau_alpha, width=L),
                    evolution_method=method,
                )
            )

        PLOT_KWARGS = dict(target_dir=OUT_DIR)

        results = si.utils.multi_map(run, specs, processes=4)

        for r in results:
            print(r.info())
            r.plot_fields_vs_time(**PLOT_KWARGS)
            r.plot_b2_vs_time(**PLOT_KWARGS)

        for log, rel in it.product((True, False), repeat=2):
            plot_name = "comparison"
            if log:
                plot_name += "__log"

            if rel:
                plot_name += "__rel"
                y = [(np.abs(r.a) ** 2) / (np.abs(results[-1].a) ** 2) for r in results]
                y_lab = (
                    r"$ \left| a(t) \right|^2 / \left| a_{\mathrm{RK4}}(t) \right|^2$"
                )
            else:
                y = [np.abs(r.a) ** 2 for r in results]
                y_lab = r"$ \left| a(t) \right|^2 $"

            si.vis.xy_plot(
                plot_name,
                results[0].times,
                *y,
                line_labels=[r.name for r in results],
                x_label=r"Time $t$",
                x_unit="asec",
                y_label=y_lab,
                y_log_axis=log,
                **PLOT_KWARGS
            )
