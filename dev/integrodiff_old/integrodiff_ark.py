import logging
import os

import numpy as np
import simulacra as si
from simulacra.units import *

import ionization as ion
import ide as ide

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

if __name__ == "__main__":
    with si.utils.LogManager(
        "simulacra", "ionization", stdout_logs=True, stdout_level=logging.DEBUG
    ) as logger:
        pw = 50
        flu = 20

        electric_field = ion.potentials.SincPulse(
            pulse_width=pw * asec, fluence=flu * Jcm2
        )

        q = electron_charge
        m = electron_mass_reduced
        L = bohr_radius

        tau_alpha = 4 * m * (L ** 2) / hbar
        prefactor = -np.sqrt(pi) * (L ** 2) * ((q / hbar) ** 2)

        dt = 1
        t_bound = 1000

        sim = ide.AdaptiveIntegroDifferentialEquationSpecification(
            "ark",
            time_initial=-t_bound * asec,
            time_final=t_bound * asec,
            time_step=dt * asec,
            prefactor=prefactor,
            f=electric_field.get_electric_field_amplitude,
            kernel=ide.gaussian_kernel_LEN,
            kernel_kwargs=dict(tau_alpha=tau_alpha),
            evolution_method="ARK4",
            integration_method="simpson",
            maximum_time_step=1 * asec,
        ).to_sim()

        logger.debug(sim.info())
        sim.run()
        logger.debug(sim.info())

        sim.plot_b2_vs_time(
            target_dir=OUT_DIR,
            y_axis_label=r"$   \left| a_{\alpha}(t) \right|^2  $",
            field_axis_label=r"${}(t)$".format(str_efield),
            field_scale="AEF",
        )

        si.vis.xy_plot(
            "time_step",
            sim.times,
            sim.time_steps_by_times,
            x_axis_label=r"Time $t$",
            x_unit="asec",
            y_axis_label=r"Time Step $\Delta t$",
            y_unit="asec",
            y_log_axis=True,
            target_dir=OUT_DIR,
        )

        print(sim.time_index)
        print(sim.computed_time_steps)
