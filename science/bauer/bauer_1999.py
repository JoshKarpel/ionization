"""
Bauer1999, Bauer2016 reference a 2.4 E^2 ionization rate. Can we get that?
"""

import logging
import os
import functools
import datetime
import itertools

import numpy as np

import simulacra as si
from simulacra.units import *

import ionization as ion
import ionization.integrodiff as ide

import matplotlib.pyplot as plt

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


class BauerGaussianPulse(ion.UniformLinearlyPolarizedElectricPotential):
    """Gaussian pulse as defined in Bauer1999. Phase = 0 is a sine-like pulse."""

    def __init__(self, amplitude = 0.3 * atomic_electric_field, omega = .2 * atomic_angular_frequency, number_of_cycles = 6, phase = 0, **kwargs):
        super().__init__(**kwargs)

        self.amplitude = amplitude
        self.omega = omega
        self.number_of_cycles = number_of_cycles
        self.phase = phase

        self.pulse_center = number_of_cycles * pi / self.omega
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


def run(spec):
    with LOGMAN as logger:
        sim = si.utils.find_or_init_sim(spec, search_dir = SIM_LIB)

        if sim.status != si.Status.FINISHED:
            sim.run_simulation()
            sim.save(target_dir = SIM_LIB)

        sim.plot_wavefunction_vs_time(**PLOT_KWARGS)

        si.vis.xy_plot(
            f'{sim.file_name}__initial_state_overlap__tdse',
            sim.data_times / sim.spec.electric_potential.cycle_time / 2
            if sim.spec.electric_potential.number_of_cycles == 6
            else sim.data_times / sim.spec.electric_potential.cycle_time,
            sim.state_overlaps_vs_time[sim.spec.initial_state],
            x_label = r'$t$ (cycles)',
            x_lower_limit = 0,
            x_upper_limit = 6,
            y_label = r'$\Gamma(t)$',
            **PLOT_KWARGS
        )

    return sim


def get_pulse_identifier(pulse):
    return f'E={uround(pulse.amplitude, atomic_electric_field, 1)}_Nc={pulse.number_of_cycles}_omega={uround(pulse.omega, atomic_angular_frequency, 1)}'


if __name__ == '__main__':
    with LOGMAN as logger:
        amplitudes = np.array([.3, .5]) * atomic_electric_field
        number_of_cycleses = [6, 12]
        omegas = np.array([.2]) * atomic_angular_frequency

        r_bound = 120 * bohr_radius
        mask_inner = 100 * bohr_radius
        mask_outer = r_bound
        r_points = 1200
        l_points = 2000
        mesh_identifier = f'R={uround(r_bound, bohr_radius)}_Nr={r_points}_L={l_points}'

        specs = []
        for amplitude, number_of_cycles, omega in itertools.product(amplitudes, number_of_cycleses, omegas):
            pulse = BauerGaussianPulse(amplitude = amplitude, number_of_cycles = number_of_cycles, omega = omega)
            pulse_identifier = get_pulse_identifier(pulse)

            times = np.linspace(0, pulse.pulse_center * 2, 1000)

            si.vis.xy_plot(
                f'field__{pulse_identifier}',
                times,
                pulse.get_electric_field_amplitude(times),
                pulse.get_electric_field_envelope(times) * pulse.amplitude,
                x_unit = 'fsec',
                y_unit = 'atomic_electric_field',
                **PLOT_KWARGS
            )

            specs.append(ion.SphericalHarmonicSpecification(
                f'tdse__{mesh_identifier}__{pulse_identifier}',
                r_bound = r_bound,
                r_points = r_points,
                l_points = l_points,
                time_initial = times[0],
                time_final = times[-1],
                time_step = 1 * asec,
                electric_potential = pulse,
                mask = ion.RadialCosineMask(inner_radius = mask_inner, outer_radius = mask_outer),
                use_numeric_eigenstates = True,
                numeric_eigenstate_max_energy = 20 * eV,
                numeric_eigenstate_max_angular_momentum = 20,
                checkpoints = True,
                checkpoint_dir = SIM_LIB,
                checkpoint_every = datetime.timedelta(minutes = 1),
            ))

        sims = si.utils.multi_map(run, specs, processes = 2)

        longest_pulse = max((sim.spec.electric_potential for sim in sims), key = lambda p: 2 * p.pulse_center)

        si.vis.xxyy_plot(
            f'pulse_ionization_comparison__{mesh_identifier}',
            # [
            #     sim.data_times / sim.spec.electric_potential.cycle_time / 2
            #     if sim.spec.electric_potential.number_of_cycles == 6
            #     else sim.data_times / sim.spec.electric_potential.cycle_time
            #     for sim in sims
            # ],
            [
                sim.data_times for sim in sims
            ],
            [
                sim.state_overlaps_vs_time[sim.spec.initial_state]
                for sim in sims
            ],
            line_labels = [
                get_pulse_identifier(sim.spec.electric_potential).replace('_', ' ').replace('E', '$\mathcal{E}_0$').replace('Nc', '$N$').replace('omega', '$\omega$')
                for sim in sims
            ],
            # x_label = r'$t$',
            # x_unit = 'fsec',
            x_label = r'$t$ (cycles)',
            x_unit = 2 * longest_pulse.pulse_center / longest_pulse.number_of_cycles,
            x_lower_limit = 0,
            x_upper_limit = longest_pulse.pulse_center,
            y_label = r'$\Gamma(t)$',
            y_lower_limit = 0,
            y_upper_limit = 1,
            y_pad = 0,
            font_size_legend = 10,
            legend_kwargs = dict(
                loc = 'upper left',
                bbox_to_anchor = (-.1, -.25),
                borderaxespad = 0,
                ncol = 2,
            ),
            **PLOT_KWARGS,
        )
