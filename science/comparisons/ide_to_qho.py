import logging
import os
import itertools

import numpy as np
import simulacra as si
from simulacra.units import *

import ionization as ion
import ide as ide

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)
SIM_LIB = os.path.join(OUT_DIR, "simlib")

logman = si.utils.LogManager(
    "simulacra", "ionization", stdout_logs=True, stdout_level=logging.DEBUG
)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=3)


def run(spec):
    with logman as logger:
        sim = si.utils.find_or_init_sim(spec, search_dir=SIM_LIB)

        sim.info().log()
        if not sim.status == si.Status.FINISHED:
            sim.run_simulation()
            sim.save(target_dir=SIM_LIB)
            sim.info().log()

        sim.plot_wavefunction_vs_time(**PLOT_KWARGS)

        return sim


if __name__ == "__main__":
    with logman as logger:
        pulse_widths = np.array([200]) * asec
        fluences = np.array([0.001, 0.01, 0.1, 1, 10]) * Jcm2
        phases = [0, pi / 2]

        for pw, flu, phase in itertools.product(pulse_widths, fluences, phases):
            t_bound = 30

            efield = ion.potentials.SincPulse(
                pulse_width=pw,
                fluence=flu,
                phase=phase,
                window=ion.potentials.LogisticWindow(
                    window_time=(t_bound - 2) * pw, window_width=0.2 * pw
                ),
            )

            test_width = 1 * bohr_radius
            test_charge = 1 * electron_charge
            test_mass = 1 * electron_mass
            potential_depth = 36.831335 * eV

            internal_potential = ion.HarmonicOscillator.from_ground_state_energy_and_mass(
                ground_state_energy=rydberg, mass=electron_mass
            )

            shared_kwargs = dict(
                test_width=test_width,
                test_charge=test_charge,
                test_mass=test_mass,
                potential_depth=potential_depth,
                electric_potential=efield,
                time_initial=-t_bound * pw,
                time_final=t_bound * pw,
                time_step=1 * asec,
                electric_potential_dc_correction=True,
                x_bound=200 * bohr_radius,
                x_points=2 ** 12,
                mask=ion.RadialCosineMask(
                    inner_radius=180 * bohr_radius, outer_radius=200 * bohr_radius
                ),
                use_numeric_eigenstates=True,
                numeric_eigenstate_max_energy=10 * eV,
                time_step_minimum=0.05 * asec,
                time_step_maximum=1 * asec,
                error_on="da/dt",
                epsilon=1e-6,
                analytic_eigenstate_type=ion.QHOState,
                checkpoints=True,
                checkpoint_dir=SIM_LIB,
            )

            prefix = (
                f"pw={pw / asec:2f}as_flu={flu / Jcm2:4f}jcm2_phase={phase / pi:3f}pi"
            )

            specs = [
                ion.LineSpecification(
                    prefix + "__line_len",
                    internal_potential=internal_potential,
                    initial_state=ion.QHOState.from_potential(
                        internal_potential, mass=test_mass
                    ),
                    store_data_every=5,
                    evolution_gauge="LEN",
                    **shared_kwargs,
                ),
                ion.LineSpecification(
                    prefix + "__line_vel",
                    internal_potential=internal_potential,
                    initial_state=ion.QHOState.from_potential(
                        internal_potential, mass=test_mass
                    ),
                    store_data_every=5,
                    evolution_gauge="VEL",
                    **shared_kwargs,
                ),
                ide.IntegroDifferentialEquationSpecification(
                    prefix + "__ide_len",
                    integral_prefactor=ide.gaussian_prefactor_LEN(
                        test_width, test_charge
                    ),
                    kernel=ide.gaussian_kernel_LEN,
                    kernel_kwargs={
                        "tau_alpha": ide.gaussian_tau_alpha_LEN(test_width, test_mass)
                    },
                    evolution_gauge="LEN",
                    evolution_method="ARK4",
                    **shared_kwargs,
                ),
                ide.IntegroDifferentialEquationSpecification(
                    prefix + "__ide_vel",
                    integral_prefactor=ide.gaussian_prefactor_VEL(
                        test_width, test_charge, test_mass
                    ),
                    kernel=ide.gaussian_kernel_VEL,
                    kernel_kwargs={
                        "tau_alpha": ide.gaussian_tau_alpha_VEL(test_width, test_mass),
                        "width": test_width,
                    },
                    evolution_gauge="VEL",
                    evolution_method="ARK4",
                    **shared_kwargs,
                ),
            ]

            results = si.utils.multi_map(run, specs, processes=4)
