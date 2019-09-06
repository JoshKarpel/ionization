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
from simulacra.units import *

import ionization as ion
import ide as ide

import matplotlib.patches as mpatches
import matplotlib.lines as mlines

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)
SIM_LIB = os.path.join(OUT_DIR, "SIMLIB")

LOGMAN = si.utils.LogManager("simulacra", "ionization", stdout_level=logging.INFO)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)

ANIMATOR_KWARGS = dict(target_dir=OUT_DIR, fig_dpi_scale=1, length=30, fps=30)


class BauerGaussianPulse(potentials.UniformLinearlyPolarizedElectricPotential):
    """Gaussian pulse as defined in Bauer1999. Phase = 0 is a sine-like pulse."""

    def __init__(
        self,
        amplitude=0.3 * atomic_electric_field,
        omega=0.2 * atomic_angular_frequency,
        number_of_cycles=6,
        phase=0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.amplitude = amplitude
        self.omega = omega
        self.number_of_cycles = number_of_cycles
        self.phase = phase

        self.pulse_center = number_of_cycles * pi / self.omega
        self.sigma2 = self.pulse_center ** 2 / (4 * np.log(20))

    @property
    def cycle_time(self):
        return 2 * self.pulse_center / self.number_of_cycles

    def get_electric_field_envelope(self, t):
        return np.exp(-((t - self.pulse_center) ** 2) / (4 * self.sigma2))

    def get_electric_field_amplitude(self, t):
        """Return the electric field amplitude at time t."""
        amp = self.get_electric_field_envelope(t) * np.sin(
            (self.omega * t) + self.phase
        )

        return amp * self.amplitude * super().get_electric_field_amplitude(t)


def run(spec):
    with LOGMAN as logger:
        sim = si.utils.find_or_init_sim(spec, search_dir=SIM_LIB)

        if sim.status != si.Status.FINISHED:
            sim.run_simulation()
            sim.save(target_dir=SIM_LIB)

        sim.plot_wavefunction_vs_time(**PLOT_KWARGS)

    return sim


def get_pulse_identifier(pulse):
    return f"E={pulse.amplitude / atomic_electric_field:1f}_Nc={pulse.number_of_cycles}_omega={pulse.omega / atomic_angular_frequency:1f}"


def calculate_landau_rate(field_amplitude):
    scaled_energy = np.abs(ion.states.HydrogenBoundState(1, 0).energy) / hartree
    scaled_field = np.abs(field_amplitude) / atomic_electric_field

    return np.where(
        np.isclose(scaled_field, 0),
        0,
        4
        * (((2 * scaled_energy) ** 2.5) / scaled_field)
        * np.exp(-2 * ((2 * scaled_energy) ** 1.5) / (3 * scaled_field))
        / atomic_time,
    )


def calculate_keldysh_rate(field_amplitude):
    scaled_energy = np.abs(states.HydrogenBoundState(1, 0).energy) / hartree
    scaled_field = np.abs(field_amplitude) / atomic_electric_field

    return np.where(
        np.isclose(scaled_field, 0),
        0,
        (np.sqrt(6 * pi) / (2 ** 1.25))
        * scaled_field
        * np.sqrt(scaled_energy / ((2 * scaled_field) ** 1.5))
        * np.exp(-2 * ((2 * scaled_energy) ** 1.5) / (3 * scaled_field))
        / atomic_time,
    )


def calculate_empirical_rate(field_amplitude, prefactor=2.4):
    return (
        prefactor
        * ((np.abs(field_amplitude) / atomic_electric_field) ** 2)
        / atomic_time
    )


def calculate_landau_critical_rate(prefactor=2.4):
    # amplitudes = np.geomspace(.05, 10, 1000) * atomic_electric_field
    critical_field = optim.brentq(
        lambda amp: calculate_landau_rate(amp)
        - calculate_empirical_rate(amp, prefactor=prefactor),
        0.01 * atomic_electric_field,
        0.5 * atomic_electric_field,
    )

    return critical_field


def do_bauer_empirical_ionization(pulse, times, prefactor=2.4):
    field_amplitudes = pulse.get_electric_field_amplitude(times)

    landau_rate = calculate_landau_rate(field_amplitudes)
    empirical_rate = calculate_empirical_rate(field_amplitudes, prefactor=prefactor)

    critical_field = calculate_landau_critical_rate(prefactor)

    ionization_rate = np.where(
        np.abs(field_amplitudes) > critical_field,
        empirical_rate,
        # 0,
        landau_rate,
    )

    return np.exp(-integ.cumtrapz(y=ionization_rate, x=times, initial=0))


def calculate_empirical_ionization_from_sim(sim, prefactor=2.4):
    pulse, times = sim.spec.electric_potential, sim.times

    return do_bauer_empirical_ionization(pulse, times, prefactor=prefactor)


def plot_ionization_rates():
    amplitudes = np.geomspace(0.05, 10, 1000) * atomic_electric_field

    si.vis.xy_plot(
        "ionization_rates_comparison",
        amplitudes,
        calculate_empirical_rate(amplitudes, prefactor=2.4),
        calculate_empirical_rate(amplitudes, prefactor=1.03),
        calculate_landau_rate(amplitudes),
        calculate_keldysh_rate(amplitudes),
        line_labels=[
            "$ 2.4 \, \mathcal{E}(t)^2 $",
            "$ 1.03 \, \mathcal{E}(t)^2 $",
            "$W_L$",
            "$W_K$",
        ],
        legend_kwargs=dict(
            loc="upper left", bbox_to_anchor=(-0.1, -0.25), borderaxespad=0, ncol=2
        ),
        x_unit="atomic_electric_field",
        x_label=r"$ \mathcal{E} $",
        x_log_axis=True,
        y_unit=1 / atomic_time,
        y_label=r"Ionization Rate $W$ (a.u.)",
        y_log_axis=True,
        y_lower_limit=0.001 / atomic_time,
        y_upper_limit=10 / atomic_time,
        y_log_pad=1,
        **PLOT_KWARGS,
    )


def run_tdse_sims(
    amplitudes=np.array([0.3, 0.5]) * atomic_electric_field,
    number_of_cycleses=(6, 12),
    omegas=np.array([0.2]) * atomic_angular_frequency,
    r_bound=100 * bohr_radius,
    mask_inner=75 * bohr_radius,
    mask_outer=100 * bohr_radius,
    r_points=500,
    l_points=300,
):
    mesh_identifier = f"R={r_bound / bohr_radius:3f}_Nr={r_points}_L={l_points}"

    specs = []
    for amplitude, number_of_cycles, omega in itertools.product(
        amplitudes, number_of_cycleses, omegas
    ):
        pulse = BauerGaussianPulse(
            amplitude=amplitude, number_of_cycles=number_of_cycles, omega=omega
        )
        pulse_identifier = get_pulse_identifier(pulse)

        times = np.linspace(-pulse.pulse_center, pulse.pulse_center * 3, 1000)

        si.vis.xy_plot(
            f"field__{pulse_identifier}",
            times,
            pulse.get_electric_field_amplitude(times),
            pulse.get_electric_field_envelope(times) * pulse.amplitude,
            x_unit="fsec",
            y_unit="atomic_electric_field",
            **PLOT_KWARGS,
        )

        specs.append(
            ion.SphericalHarmonicSpecification(
                f"tdse__{mesh_identifier}__{pulse_identifier}",
                r_bound=r_bound,
                r_points=r_points,
                l_points=l_points,
                time_initial=times[0],
                time_final=times[-1],
                time_step=1 * asec,
                electric_potential=pulse,
                mask=potentials.RadialCosineMask(
                    inner_radius=mask_inner, outer_radius=mask_outer
                ),
                use_numeric_eigenstates=True,
                numeric_eigenstate_max_energy=20 * eV,
                numeric_eigenstate_max_angular_momentum=5,
                checkpoints=True,
                checkpoint_dir=SIM_LIB,
                checkpoint_every=datetime.timedelta(minutes=1),
            )
        )

    return si.utils.multi_map(run, specs, processes=2), mesh_identifier


def run_ide_sims(tdse_sims):
    specs = []
    dt = 0.5 * asec
    for sim in tdse_sims:
        specs.append(
            ide.IntegroDifferentialEquationSpecification(
                sim.name.replace("tdse", "ide") + f"__dt={dt / asec:3f}",
                electric_potential=sim.spec.electric_potential,
                time_initial=sim.times[0],
                time_final=sim.times[-1],
                time_step=dt,
                kernel=ide.LengthGaugeHydrogenKernelWithContinuumContinuumInteraction(),
                checkpoints=True,
                checkpoint_dir=SIM_LIB,
                checkpoint_every=datetime.timedelta(minutes=1),
            )
        )

    ide_sims = si.utils.multi_map(run, specs, processes=2)

    return ide_sims, dt


def landau_if_below_critical_field(field_amplitude, prefactor=2.4):
    critical_field = calculate_landau_critical_rate(prefactor)
    return np.where(
        np.abs(field_amplitude) <= critical_field,
        calculate_landau_rate(field_amplitude),
        0,
    )


def run_ide_sims_with_decay(tdse_sims, prefactor=2.4):
    specs = []
    dt = 1 * asec
    for sim in tdse_sims:
        specs.append(
            ide.IntegroDifferentialEquationSpecification(
                sim.name.replace("tdse", "ide")
                + f"__dt={dt / asec:3f}__WITH_DECAY_prefactor={prefactor}",
                electric_potential=sim.spec.electric_potential,
                time_initial=sim.times[0],
                time_final=sim.times[-1],
                time_step=dt,
                kernel=ide.LengthGaugeHydrogenKernelWithContinuumContinuumInteraction(),
                checkpoints=True,
                checkpoint_dir=SIM_LIB,
                checkpoint_every=datetime.timedelta(minutes=1),
                tunneling_model=functools.partial(
                    landau_if_below_critical_field, prefactor=prefactor
                ),
            )
        )

    ide_sims = si.utils.multi_map(run, specs, processes=2)

    return ide_sims, dt


def make_comparison_plot(
    tdse_sims,
    ide_sims,
    ide_sims_with_decay,
    sims_to_empirical,
    mesh_identifier,
    prefactor,
    ide_time_step,
):
    longest_pulse = max(
        (sim.spec.electric_potential for sim in tdse_sims),
        key=lambda p: 2 * p.pulse_center,
    )

    x_data = []
    y_data = []
    line_labels = []
    line_kwargs = []
    colors = [f"C{n}" for n in range(len(tdse_sims))]
    styles = ["-", "--", "-.", ":"]
    for tdse_sim, ide_sim, ide_sim_with_decay, color in zip(
        tdse_sims, ide_sims, ide_sims_with_decay, colors
    ):
        empirical = sims_to_empirical[tdse_sim]

        x_data.append(tdse_sim.data_times)
        x_data.append(ide_sim.data_times)
        x_data.append(ide_sim_with_decay.data_times)
        x_data.append(tdse_sim.times)

        y_data.append(tdse_sim.state_overlaps_vs_time[tdse_sim.spec.initial_state])
        y_data.append(ide_sim.b2)
        y_data.append(ide_sim_with_decay.b2)
        y_data.append(empirical)

        for style in styles:
            line_kwargs.append({"linestyle": style, "color": color})

    color_patches = [
        mpatches.Patch(
            color=color,
            label=get_pulse_identifier(sim.spec.electric_potential)
            .replace("_", " ")
            .replace("E", "$\mathcal{E}_0$")
            .replace("Nc", "$N$")
            .replace("omega", "$\omega$"),
        )
        for color, sim in zip(colors, tdse_sims)
    ]

    sim_types = ["TDSE", "IDE", "IDE w/ $W_L$", "EMP"]
    style_patches = [
        mlines.Line2D(
            [], [], color="black", linestyle=style, linewidth=1, label=sim_type
        )
        for style, sim_type in zip(styles, sim_types)
    ]

    legend_handles = color_patches + style_patches

    si.vis.xxyy_plot(
        f"pulse_ionization_comparison_with_empirical_rates__prefactor={prefactor}__{mesh_identifier}__IDEdt={ide_time_step / asec:3f}",
        x_data,
        y_data,
        # line_labels = line_labels,
        line_kwargs=line_kwargs,
        x_label=r"$t$ (cycles)",
        x_unit=2 * longest_pulse.pulse_center / longest_pulse.number_of_cycles,
        x_lower_limit=0,
        x_upper_limit=longest_pulse.pulse_center,
        y_label=r"$\Gamma(t)$",
        y_lower_limit=0,
        y_upper_limit=1,
        y_pad=0,
        font_size_legend=9,
        legend_kwargs=dict(
            loc="upper left",
            bbox_to_anchor=(-0.1, -0.25),
            borderaxespad=0,
            ncol=2,
            handles=legend_handles,
        ),
        title=f"Prefactor = {prefactor}",
        **PLOT_KWARGS,
    )


if __name__ == "__main__":
    with LOGMAN as logger:
        print(calculate_landau_critical_rate(2.4) / atomic_electric_field)
        print(calculate_landau_critical_rate(2.33) / atomic_electric_field)

        tdse_sims, mesh_identifier = run_tdse_sims(
            amplitudes=np.array([0.3, 0.5]) * atomic_electric_field,
            number_of_cycleses=[6, 12],
            omegas=np.array([0.2]) * atomic_angular_frequency,
            r_bound=100 * bohr_radius,
            mask_inner=80 * bohr_radius,
            mask_outer=100 * bohr_radius,
            r_points=1000,
            l_points=500,
        )

        ide_sims, ide_time_step = run_ide_sims(tdse_sims)

        plot_ionization_rates()

        for prefactor in [2.4]:
            sim_to_empirical = {
                sim: calculate_empirical_ionization_from_sim(sim, prefactor=prefactor)
                for sim in tdse_sims
            }
            ide_sims_with_decay, ide_time_step = run_ide_sims_with_decay(
                tdse_sims, prefactor=prefactor
            )
            make_comparison_plot(
                tdse_sims,
                ide_sims,
                ide_sims_with_decay,
                sim_to_empirical,
                mesh_identifier,
                prefactor,
                ide_time_step,
            )
