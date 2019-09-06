import logging
import os

import numpy as np

import simulacra as si
from simulacra.units import *
import ionization as ion
import ide as ide

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

logman = si.utils.LogManager(
    "simulacra", "ionization", stdout_logs=True, stdout_level=logging.DEBUG
)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)


def run(spec):
    with logman as logger:
        sim = spec.to_sim()

        logger.debug(sim.info())
        sim.run()
        logger.debug(sim.info())

        sim.plot_b2_vs_time(**PLOT_KWARGS)

        return sim


if __name__ == "__main__":
    with logman as logger:
        td = np.linspace(0, 500 * asec, 1000)

        test_mass = 1 * electron_mass_reduced
        test_charge = 1 * electron_charge
        test_width = 1 * bohr_radius

        kernel_hydrogen = ide.hydrogen_kernel_LEN(td)

        si.vis.xy_plot(
            "hydrogen_kernel",
            td,
            np.abs(kernel_hydrogen),
            np.real(kernel_hydrogen),
            np.imag(kernel_hydrogen),
            line_labels=[r"Abs", r"Re", r"Im"],
            x_label=r"$t-t'$",
            x_unit="asec",
            y_label=r"$K(t-t')$",
            vlines=[93 * asec, 150 * asec],
            **PLOT_KWARGS,
        )

        omega = ion.HydrogenBoundState(1, 0).energy / hbar
        kp = ide.hydrogen_kernel_LEN(td) * np.exp(1j * omega * td)

        si.vis.xy_plot(
            "hydrogen_kernel_with_extra_term",
            td,
            np.abs(kp) / np.abs(kp)[0],
            np.real(kp) / np.abs(kp)[0],
            np.imag(kp) / np.abs(kp)[0],
            # np.real(np.exp(1j * omega * td)),
            # np.imag(np.exp(1j * omega * td)),
            line_labels=[
                r"Abs",
                r"Re",
                r"Im",
                # r'Re extra',
                # r'Im extra',
            ],
            x_label=r"$t-t'$",
            x_unit="asec",
            y_label=r"$K(t-t')$",
            vlines=[93 * asec, 150 * asec],
            legend_kwargs={"loc": "upper right"},
            **PLOT_KWARGS,
        )

        pulse = ion.potentials.GaussianPulse.from_number_of_cycles(
            pulse_width=200 * asec, fluence=1 * Jcm2, phase=0
        )

        shared_kwargs = dict(
            time_initial=-pulse.pulse_width * 3,
            time_final=pulse.pulse_width * 3,
            time_step=0.1 * asec,
            electric_potential=pulse,
            electric_potential_dc_correction=True,
        )

        specs = [
            ide.IntegroDifferentialEquationSpecification(
                "hydrogen",
                kernel=ide.hydrogen_kernel_LEN,
                integral_prefactor=ide.hydrogen_prefactor_LEN(test_charge),
                **shared_kwargs,
            )
        ]

        # results = si.utils.multi_map(run, specs, processes = 2)
