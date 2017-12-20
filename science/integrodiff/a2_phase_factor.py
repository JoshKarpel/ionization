import logging
import os
import collections
import itertools

import numpy as np

import simulacra as si
import simulacra.units as u

import ionization as ion
import ionization.ide as ide

import matplotlib as mpl
import matplotlib.pyplot as plt

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)
SIM_LIB = os.path.join(OUT_DIR, 'SIMLIB')

LOGMAN = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.DEBUG)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)

ANIM_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_width = 5,
    aspect_ratio = 1.5,
    fig_dpi_scale = 3,
)


def run(spec):
    with LOGMAN as logger:
        sim = si.utils.find_or_init_sim(spec, search_dir = SIM_LIB)

        if sim.status != si.Status.FINISHED:
            sim.run_simulation()
            sim.save(target_dir = SIM_LIB)

    return sim


class BauerGaussianPulse(ion.potentials.UniformLinearlyPolarizedElectricPotential):
    """Gaussian pulse as defined in Bauer1999. Phase = 0 is a sine-like pulse."""

    def __init__(self,
                 amplitude = 0.3 * u.atomic_electric_field,
                 omega = .2 * u.atomic_angular_frequency,
                 number_of_cycles = 6,
                 phase = 0,
                 **kwargs):
        super().__init__(**kwargs)

        self.amplitude = amplitude
        self.omega = omega
        self.number_of_cycles = number_of_cycles
        self.phase = phase

        self.pulse_center = number_of_cycles * u.pi / self.omega
        self.sigma2 = (self.pulse_center ** 2 / (4 * np.log(20)))

    @property
    def cycle_time(self):
        return 2 * self.pulse_center / self.number_of_cycles

    def get_electric_field_envelope(self, t):
        return np.exp(-((t - self.pulse_center) ** 2) / (4 * self.sigma2))

    def get_electric_field_amplitude(self, t):
        """Return the electric field amplitude at time t."""
        amp = self.get_electric_field_envelope(t) * np.sin((self.omega * t) + self.phase)

        return amp * self.amplitude * super().get_electric_field_amplitude(t)


def get_pulse_identifier(pulse):
    return f'E={u.uround(pulse.amplitude, u.atomic_electric_field, 1)}_Nc={pulse.number_of_cycles}_omega={u.uround(pulse.omega, u.atomic_angular_frequency, 1)}'


def animate_kernel_over_time(pulse, kernel):
    ident = f'{kernel.__class__.__name__}___{get_pulse_identifier(pulse)}'
    spec = ide.IntegroDifferentialEquationSpecification(
        ident,
        electric_potential = pulse,
        kernel = kernel,
        time_initial = -.3 * pulse.pulse_center,
        time_final = 2.3 * pulse.pulse_center,
    )
    sim = run(spec)

    times = sim.times

    time_unit, time_unit_tex = u.get_unit_value_and_latex_from_unit('asec')

    figman = si.vis.FigureManager(
        f'kernel_over_time__{ident}',
        save_on_exit = False,
        close_after_exit = False,
        **ANIM_KWARGS
    )

    with figman as figman:
        fig = figman.fig
        grid_spec = mpl.gridspec.GridSpec(2, 1, height_ratios = [4, 1], hspace = 0.07)
        ax_kernel = plt.subplot(grid_spec[0])
        ax_fields = plt.subplot(grid_spec[1], sharex = ax_kernel)
        ax_solution = ax_kernel.twinx()

        efield_line, = ax_fields.plot(
            times / time_unit,
            pulse.get_electric_field_amplitude(times) / u.atomic_electric_field,
        )
        vp_line, = ax_fields.plot(
            times / time_unit,
            pulse.get_vector_potential_amplitude_numeric_cumulative(times) * u.proton_charge / u.atomic_momentum,
        )

        phase_factor_line, = ax_kernel.plot(
            [],
            [],
            color = 'C9',
            alpha = 0.5,
            animated = True,
        )

        kernel_line_re, = ax_kernel.plot(
            [],
            [],
            color = 'C3',
            alpha = 0.8,
            animated = True,
        )
        kernel_line_im, = ax_kernel.plot(
            [],
            [],
            color = 'C4',
            alpha = 0.8,
            animated = True,
        )
        kernel_line_abs, = ax_kernel.plot(
            [],
            [],
            color = 'black',
            animated = True,
        )

        solution_line, = ax_solution.plot(
            sim.times / time_unit,
            sim.b2,
            color = 'black',
            alpha = 0.5,
        )

        time_line_upper = ax_solution.axvline(
            0,
            linestyle = ':',
            color = 'black',
            alpha = 0.6,
            animated = True,
        )
        time_line_lower = ax_fields.axvline(
            0,
            linestyle = ':',
            color = 'black',
            alpha = 0.6,
            animated = True,
        )

        ax_kernel.set_xlim(times[0] / time_unit, times[-1] / time_unit)
        ax_kernel.set_ylim(-1, 1)

        ax_solution.set_ylim(0, 1)

        ax_kernel.tick_params(
            labelleft = False,
            labelright = True,
            labeltop = True,
            labelbottom = False,
            bottom = True,
            top = True,
            left = True,
            right = True,
        )
        ax_fields.tick_params(
            labelleft = True,
            labelright = True,
            labeltop = False,
            labelbottom = True,
            bottom = True,
            top = True,
            left = True,
            right = True,
        )
        ax_solution.tick_params(
            labelleft = True,
            labelright = False,
            labeltop = False,
            labelbottom = False,
            bottom = True,
            top = False,
            left = True,
            right = False,
        )

        ax_solution.grid(True, **collections.ChainMap({'linestyle': '--'}, si.vis.GRID_KWARGS))
        ax_kernel.grid(True, **si.vis.GRID_KWARGS)
        ax_fields.grid(True, **si.vis.GRID_KWARGS)

        ax_fields.set_xlabel(fr'$t \; {time_unit_tex}$')
        ax_kernel.set_ylabel(r"$K_b(t, t') \; (a_0^2)$")
        ax_solution.set_ylabel(r'$\left|b(t)\right|^2$')

        def update_func(index):
            current_time = times[index]
            previous_time = times[:index + 1]

            y = sim.evaluate_kernel(current_time, previous_time)
            x = previous_time

            kernel_line_re.set_data(x / time_unit, np.real(y) / (u.bohr_radius ** 2))
            kernel_line_im.set_data(x / time_unit, np.imag(y) / (u.bohr_radius ** 2))
            kernel_line_abs.set_data(x / time_unit, np.abs(y) / (u.bohr_radius ** 2))

            if hasattr(kernel, '_vector_potential_phase_factor_integral'):
                phase_factor = np.real(kernel._vector_potential_phase_factor(current_time, previous_time, sim.interpolated_vector_potential))
                phase_factor_line.set_data(x / time_unit, phase_factor)

            time_line_upper.set_xdata(current_time / time_unit)
            time_line_lower.set_xdata(current_time / time_unit)

        args = list(range(len(times)))

        si.vis.animate(
            figman,
            update_func,
            args,
            artists = [
                kernel_line_re,
                kernel_line_im,
                kernel_line_abs,
                phase_factor_line,
                time_line_upper,
                time_line_lower,
            ],
            length = 60,
        )

        figman.cleanup()


if __name__ == '__main__':
    with LOGMAN as logger:
        pulses = [
            BauerGaussianPulse(amplitude = amplitude, number_of_cycles = number_of_cycles, omega = omega)
            for amplitude in np.array([.3, .5]) * u.atomic_electric_field
            for number_of_cycles in [6, 12]
            for omega in np.array([.2]) * u.atomic_angular_frequency
        ]
        kernels = [
            ide.LengthGaugeHydrogenKernel(),
            ide.ApproximateLengthGaugeHydrogenKernelWithContinuumContinuumInteraction(),
        ]

        for pulse, kernel in itertools.product(pulses, kernels):
            animate_kernel_over_time(pulse, kernel)
