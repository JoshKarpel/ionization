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
        # electric_field = ion.potentials.Rectangle(start_time = -500 * asec, end_time = 500 * asec, amplitude = 1 * atomic_electric_field)
        electric_field = ion.potentials.SincPulse(
            pulse_width=100 * asec, fluence=1 * Jcm2
        )

        q = electron_charge
        m = electron_mass_reduced
        L = bohr_radius

        tau_alpha = 4 * m * (L ** 2) / hbar
        prefactor = -np.sqrt(pi) * (L ** 2) * ((q / hbar) ** 2)

        dt = 20
        t_bound = 1000

        spec = ide.IntegroDifferentialEquationSpecification(
            "ide_test__{}__dt={}as".format(electric_field.__class__.__name__, dt),
            time_initial=-t_bound * asec,
            time_final=t_bound * asec,
            time_step=dt * asec,
            integral_prefactor=prefactor,
            electric_potential=electric_field.get_electric_field_amplitude,
            kernel=ide.gaussian_kernel_LEN,
            kernel_kwargs=dict(tau_alpha=tau_alpha),
        )

        sim = spec.to_sim()

        sim.run_simulation()

        # print(sim.y)
        # print(np.abs(sim.y) ** 2)
        # print('tau alpha (as)', tau_alpha / asec)

        sim.plot_b2_vs_time(
            target_dir=OUT_DIR,
            y_axis_label=r"$   \left| a_{\alpha}(t) \right|^2  $",
            field_axis_label=r"${}(t)$".format(str_efield),
            field_scale="AEF",
        )
