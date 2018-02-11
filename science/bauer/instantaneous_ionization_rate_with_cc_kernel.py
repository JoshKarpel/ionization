"""
Bauer1999, Bauer2016 reference a 2.4 E^2 ionization rate. Can we get that?
"""

import logging
import os
import functools
import datetime
import itertools

import numpy as np
import scipy.integrate as integ
import scipy.optimize as optim

import simulacra as si
import simulacra.units as u

import ionization as ion
import ionization.tunneling as tunneling

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)
SIM_LIB = os.path.join(OUT_DIR, 'SIMLIB')

LOGMAN = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.INFO)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)

ANIMATOR_KWARGS = dict(
    target_dir = OUT_DIR,
    fig_dpi_scale = 1,
    length = 30,
    fps = 30,
)


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


prefactor = -((u.electron_charge / u.hbar) ** 2)


def instantaneous_tunneling_rate_from_cc_kernel(pulse, times):
    sim = ion.ide.IntegroDifferentialEquationSpecification(
        'dummy',
        time_initial = times[0],
        time_final = times[-1],
        electric_potential = pulse,
        # kernel = ion.ide.LengthGaugeHydrogenKernel(),
        kernel = ion.ide.ApproximateLengthGaugeHydrogenKernelWithContinuumContinuumInteraction(),
    ).to_sim()

    kernel_integrals = np.empty_like(times, dtype = np.complex128)
    for current_time_idx, current_time in enumerate(times):
        previous_times = times[:current_time_idx + 1]
        kernel = sim.evaluate_kernel(current_time, previous_times)
        kernel_integrals[current_time_idx] = integ.simps(
            y = kernel,
            x = previous_times,
        )

    gamma = 2 * prefactor * kernel_integrals * (pulse.get_electric_field_amplitude(times) ** 2)
    return gamma


def landau_if_below_critical_field(field_amplitude):
    critical_field = .0835 * u.atomic_electric_field
    return np.where(
        np.abs(field_amplitude) <= critical_field,
        tunneling.landau_tunneling_rate(field_amplitude),
        0
    )


if __name__ == '__main__':
    with LOGMAN as logger:
        empirical_prefactor = 1.2

        amplitudes = np.array([.3, .5]) * u.atomic_electric_field
        number_of_cycleses = [6, 12]
        omegas = np.array([.2]) * u.atomic_angular_frequency

        t_extra_lower = 2
        t_extra_upper = 1
        x_unit, x_unit_name = u.fsec, 'fsec'

        for amplitude, number_of_cycles, omega in itertools.product(amplitudes, number_of_cycleses, omegas):
            pulse = BauerGaussianPulse(amplitude = amplitude, number_of_cycles = number_of_cycles, omega = omega)
            times = np.linspace(-t_extra_lower * pulse.pulse_center, (2 + t_extra_upper) * pulse.pulse_center, 5000)

            efield_vs_time = pulse.get_electric_field_amplitude(times)

            figman = si.vis.xy_plot(
                f'instantaneous_ionization_rate__{get_pulse_identifier(pulse)}',
                times,
                efield_vs_time / u.atomic_electric_field,
                pulse.get_vector_potential_amplitude_numeric_cumulative(times) * u.proton_charge / u.atomic_momentum,
                line_kwargs = [
                    {'alpha': 0.6},
                    {'alpha': 0.6},
                ],
                x_label = '$t$',
                x_unit = x_unit_name,
                y_label = 'Field Amplitudes',
                font_size_axis_labels = 12,
                title = ' '.join(get_pulse_identifier(pulse).split('_')).title(),
                save_on_exit = False,
                close_after_exit = False,
                **PLOT_KWARGS
            )

            fig = figman.fig
            ax_fields = fig.gca()
            ax_rate = ax_fields.twinx()

            # ax_rate.plot(
            #     times / x_unit,
            #     -tunneling.landau_tunneling_rate(efield_vs_time) * u.atomic_time,
            #     color = 'C2',
            #     linestyle = '--',
            #     alpha = 0.5,
            # )

            # ax_rate.plot(
            #     times / x_unit,
            #     -landau_if_below_critical_field(efield_vs_time) * u.atomic_time,
            #     color = 'C2',
            #     linestyle = '-',
            # )

            cc_tunneling_rate = instantaneous_tunneling_rate_from_cc_kernel(pulse, times)
            # ax_rate.plot(
            #     times / x_unit,
            #     np.imag(cc_tunneling_rate) * u.atomic_time,
            #     color = 'C3',
            #     linestyle = '-',
            #     alpha = 0.5,
            # )
            ax_rate.plot(
                times / x_unit,
                np.real(cc_tunneling_rate) * u.atomic_time,
                color = 'black',
                linestyle = '-',
            )

            ax_rate.plot(
                times / x_unit,
                -empirical_prefactor * ((efield_vs_time / u.atomic_electric_field) ** 2),
                color = 'C4'
            )

            ax_rate.set_ylabel(r'$\Gamma(t)$ (1/a.u.)', fontsize = 12)
            gamma_max = 6 * (np.max(np.abs(efield_vs_time / u.atomic_electric_field))) ** 2
            ax_rate.set_ylim(-gamma_max, gamma_max)
            ax_rate.grid(True, **si.vis.GRID_KWARGS)

            ax_fields.grid(False, axis = 'y')
            ax_fields.set_xlim(0 * pulse.pulse_center / x_unit, 2 * pulse.pulse_center / x_unit)
            ax_fields.set_ylim(-2, 2)

            legend_handles = [
                mlines.Line2D(
                    [], [],
                    color = 'C0',
                    alpha = 0.6,
                    linewidth = 1,
                    label = r'$\mathcal{E}$',
                ),
                mlines.Line2D(
                    [], [],
                    color = 'C1',
                    alpha = 0.6,
                    linewidth = 1,
                    label = r'$\mathcal{A}$',
                ),
                mlines.Line2D(
                    [], [],
                    color = 'black',
                    alpha = 1,
                    linewidth = 1,
                    label = r'$\Gamma_b^{\mathrm{cc}}(t) \mathcal{E}^2$',
                ),
                mlines.Line2D(
                    [], [],
                    color = 'C4',
                    alpha = 1,
                    linewidth = 1,
                    label = r'$2.4 \, \mathcal{E}^2$',
                ),
            ]

            ax_fields.legend(
                loc = 'upper left',
                fontsize = 10,
                handles = legend_handles,
            )

            figman.save()
            figman.cleanup()
