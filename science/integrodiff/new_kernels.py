import logging
import os

import numpy as np
import scipy.integrate as integ

import simulacra as si
import simulacra.units as u

import ionization as ion
import ionization.ide as ide

import matplotlib.pyplot as plt

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

LOGMAN = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.DEBUG)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)


class BauerGaussianPulse(ion.potentials.UniformLinearlyPolarizedElectricPotential):
    """Gaussian pulse as defined in Bauer1999. Phase = 0 is a sine-like pulse."""

    def __init__(self, amplitude = 0.3 * u.atomic_electric_field, omega = .2 * u.atomic_angular_frequency, number_of_cycles = 6, phase = 0, **kwargs):
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


def run(spec):
    with LOGMAN as logger:
        sim = spec.to_sim()

        sim.run()

        sim.plot_wavefunction_vs_time(**PLOT_KWARGS)

    return sim


if __name__ == '__main__':
    with LOGMAN as logger:
        KERNELS = [
            ide.LengthGaugeHydrogenKernel(),
            ide.ApproximateLengthGaugeHydrogenKernelWithContinuumContinuumInteraction(
                # integration_method = integ.quadrature,
            ),
        ]

        pulse = BauerGaussianPulse()
        # pulse = ion.potentials.GaussianPulse.from_number_of_cycles(
        #     pulse_width = 100 * u.asec,
        #     fluence = 2 * u.Jcm2,
        #     phase = 0,
        #     number_of_cycles = 3,
        # )

        shared_kwargs = dict(
            time_initial = 0,
            time_final = 2 * pulse.pulse_center,
            # time_initial = -3.5 * pulse.pulse_width,
            # time_final = 3.5 * pulse.pulse_width,
            time_step = 1 * u.asec,
            electric_potential = pulse,
            evolution_method = ide.ForwardEulerMethod(),
        )

        specs = []
        for kernel in KERNELS:
            specs.append(ide.IntegroDifferentialEquationSpecification(
                f'{kernel.__class__.__name__}',
                kernel = kernel,
                **shared_kwargs,
            ))

        results = si.utils.multi_map(run, specs, processes = 4)

        just = max(len(r.name) for r in results)
        for r in results:
            print(f'{r.name.rjust(just)} : {r.running_time}')

        si.vis.xxyy_plot(
            'kernel_comparison',
            [
                *[r.data_times for r in results]
            ],
            [
                *[r.b2 for r in results]
            ],
            line_labels = [r.spec.kernel.__class__.__name__ for r in results],
            x_unit = 'asec',
            x_label = '$t$',
            **PLOT_KWARGS,
        )
