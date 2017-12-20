import logging
import os

import numpy as np
import scipy.interpolate as interp

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

        kernel = sim.spec.kernel

        times = np.linspace(sim.spec.time_initial, sim.spec.time_final, int((sim.spec.time_final - sim.spec.time_initial) / sim.spec.time_step))
        zero_vp = lambda x: np.zeros_like(x)  # no vp, effectively no interaction
        zero_vp = interp.CubicSpline(x = times, y = zero_vp(times))
        # print(times)
        # print(len(times))

        # p = np.linspace(0, 5, 1000) * u.atomic_momentum

        si.vis.xy_plot(
            'matrix_element_aligned',
            kernel.p,
            (4 * u.pi / 3) * (kernel.p ** 2) * np.abs(kernel.z_dipole_matrix_element_per_momentum(kernel.p, 0)) ** 2,
            x_unit = 'atomic_momentum',
            x_label = r'$p$',
            y_label = r'$\frac{4 \pi}{3} \, p^2 \, \left| d_z(p) \right|^2$',
            x_upper_limit = 5 * u.atomic_momentum,
            **PLOT_KWARGS,
        )

        integrand_vs_p = kernel.integrand_vs_p(times[500], times[500], zero_vp)
        si.vis.xy_plot(
            'integrand_vs_p_at_t_t',
            kernel.p,
            np.abs(integrand_vs_p),
            np.real(integrand_vs_p),
            np.imag(integrand_vs_p),
            line_kwargs = [
                dict(color = 'black', linestyle = '-'),
                dict(color = 'red', linestyle = '--'),
                dict(color = 'purple', linestyle = ':'),
            ],
            x_unit = 'atomic_momentum',
            x_label = r'$p$',
            y_label = r'$\int_{\Omega} d\Omega \, p^2 \, \left| d_z(p) \right|^2$',
            x_upper_limit = 5 * u.atomic_momentum,
            **PLOT_KWARGS,
        )
