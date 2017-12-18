import logging
import os

import numpy as np

import simulacra as si
import simulacra.units as u

import ionization as ion
import ionization.ide as ide

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

LOGMAN = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.DEBUG)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)

if __name__ == '__main__':
    with LOGMAN as logger:
        pulse = ion.potentials.GaussianPulse.from_number_of_cycles(
            pulse_width = 200 * u.asec,
            number_of_cycles = 4,
            fluence = 1 * u.Jcm2,
            phase = 0,
        )

        sim = ide.IntegroDifferentialEquationSpecification(
            'full_hydrogen_kernel_test',
            time_initial = -5 * pulse.pulse_width,
            time_final = 5 * pulse.pulse_width,
            time_step = 1 * u.asec,
            kernel = ide.FullHydrogenKernel(),
        ).to_simulation()

        vp = sim.interpolated_vector_potential
        zero_vp = lambda x: 0  # no vp, effectively no interaction

        kernel = sim.spec.kernel

        times = np.linspace(sim.spec.time_initial, sim.spec.time_final, int((sim.spec.time_final - sim.spec.time_initial) / sim.spec.time_step))
        print(times)
        print(len(times))

        p = np.linspace(0, 5, 1000) * u.atomic_momentum

        si.vis.xy_plot(
            'matrix_element_aligned',
            p,
            kernel.z_dipole_matrix_element(p, 0),
            x_unit = 'atomic_momentum',
            x_label = r'$p$',
            y_label = r'$d_z(p)$',
            **PLOT_KWARGS,
        )
