import logging
import os

import simulacra as si
from simulacra.units import *

import ionization as ion


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)


def run(spec):
    with si.utils.LogManager(
        "simulacra",
        "ionization",
        stdout_level=logging.DEBUG,
        file_logs=False,
        file_dir=OUT_DIR,
        file_level=logging.DEBUG,
    ) as logger:
        sim = spec.to_sim()

        sim.run()

        sim.plot_wavefunction_vs_time(target_dir=OUT_DIR)

        return sim


if __name__ == "__main__":
    with si.utils.LogManager(
        "simulacra",
        "ionization",
        stdout_level=logging.INFO,
        file_logs=False,
        file_dir=OUT_DIR,
        file_level=logging.DEBUG,
    ) as logger:
        source = ion.potentials.SincPulse(
            pulse_width=200 * asec, fluence=1 * (J / (cm ** 2))
        )
        pulse_widths = [290, 310, 390, 410]
        # pulse_widths = [50, 100, 200, 400, 600, 800]
        t_step = 2 * asec

        mask = ion.RadialCosineMask(
            inner_radius=100 * bohr_radius, outer_radius=150 * bohr_radius
        )

        specs = []
        for phase in ("cos", "sin"):
            for pw in pulse_widths:
                sinc = ion.potentials.SincPulse.from_amplitude_density(
                    pulse_width=pw * asec,
                    amplitude_density=source.amplitude_omega,
                    phase=phase,
                )

                specs.append(
                    ion.SphericalHarmonicSpecification(
                        "pw={}asec_phase={}".format(pw, phase),
                        r_bound=150 * bohr_radius,
                        r_points=600,
                        l_bound=100,
                        electric_potential=sinc,
                        mask=mask,
                        time_initial=-15 * pw * asec,
                        time_final=15 * pw * asec,
                        time_step=t_step,
                    )
                )

        si.utils.multi_map(run, specs)
