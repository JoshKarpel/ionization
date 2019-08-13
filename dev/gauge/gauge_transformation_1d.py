"""
This script tests all of the evolution methods on each mesh.
"""

import logging
import os

import numpy as np

import simulacra as si
from simulacra.units import *

import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

logman = si.utils.LogManager("simulacra", "ionization", stdout_level=logging.INFO)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=3)


def plot_g_1d(name, g, x, **kwargs):
    g_real = np.real(g)
    g_imag = np.imag(g)
    g_abs = np.abs(g)
    norm = np.nanmax(g_abs)

    si.vis.xy_plot(
        name,
        x,
        g_real / norm,
        g_imag / norm,
        g_abs / norm,
        line_labels=("Real g", "Imag g", "Abs g"),
        line_kwargs=(None, None, {"linestyle": "--"}),
        x_unit="bohr_radius",
        y_lower_limit=-1,
        y_upper_limit=1,
        **kwargs,
    )


GAUGE_TO_OPP = {"LEN": "VEL", "VEL": "LEN"}


def wrapped_plot_g_1d(sim):
    if (
        sim.time_index % (sim.time_steps // 6) == 0
        or sim.time_index == sim.time_steps - 1
    ):
        print(f"index {sim.time_index}")
        plot_g_1d(
            f"{sim.time_index}_g__{sim.spec.evolution_gauge}",
            sim.mesh.g,
            sim.mesh.x_mesh,
            **PLOT_KWARGS,
        )
        plot_g_1d(
            f"{sim.time_index}_g__{GAUGE_TO_OPP[sim.spec.evolution_gauge]}_from_{sim.spec.evolution_gauge}",
            sim.mesh.gauge_transformation(leaving_gauge=sim.spec.evolution_gauge),
            sim.mesh.x_mesh,
            **PLOT_KWARGS,
        )


def run_sim(spec):
    with logman as logger:
        sim = spec.to_sim()

        sim.info().log()
        sim.run(callback=wrapped_plot_g_1d, progress_bar=True)
        sim.info().log()

        sim.plot_state_overlaps_vs_time(**PLOT_KWARGS)

        return sim


if __name__ == "__main__":
    with logman as logger:
        x_bound = 50 * bohr_radius
        spacing = 1 * eV
        amp = 0.001 * atomic_electric_field
        t_bound = 2
        max_n = 10

        potential = ion.HarmonicOscillator.from_energy_spacing_and_mass(
            spacing, electron_mass
        )

        efield = ion.SineWave.from_photon_energy(spacing, amplitude=amp)
        efield.window = ion.SymmetricExponentialTimeWindow(
            window_time=(t_bound - 1) * efield.period_carrier,
            window_width=0.1 * efield.period_carrier,
        )

        # efield = ion.Rectangle(amplitude = 1 * atomic_electric_field, start_time = 50 * asec, end_time = 250 * asec)

        line_spec_base = dict(
            x_bound=x_bound,
            x_points=2 ** 10,
            internal_potential=potential,
            electric_potential=efield,
            test_charge=electron_charge,
            initial_state=ion.QHOState.from_potential(potential, electron_mass),
            test_states=tuple(
                ion.QHOState.from_potential(potential, electron_mass, n)
                for n in range(max_n + 1)
            ),
            time_initial=-t_bound * efield.period_carrier,
            time_final=t_bound * efield.period_carrier,
            # time_initial = 0, time_final = 300 * asec,
            time_step=5 * asec,
            electric_potential_dc_correction=True,
            evolution_method="CN",
            animators=[
                animation.animators.RectangleAnimator(
                    # length = 10,
                    length=30,
                    fps=30,
                    target_dir=OUT_DIR,
                    axman_wavefunction=animation.animators.LineMeshAxis(
                        norm=si.vis.AbsoluteRenormalize()
                    ),
                    axman_lower=animation.animators.ElectricPotentialPlotAxis(
                        show_vector_potential=True
                    ),
                )
            ],
        )

        specs = []

        for gauge in ("LEN", "VEL"):
            specs.append(
                ion.LineSpecification(
                    f"{gauge}",
                    **line_spec_base,
                    evolution_gauge=gauge,
                    dipole_gauges=("LEN",),
                )
            )

    results = si.utils.multi_map(run_sim, specs)

    for r in results:
        print(r.electric_dipole_moment_expectation_value_vs_time)

    si.vis.xxyy_plot(
        "dipole_moment",
        (r.data_times for r in results),
        (r.electric_dipole_moment_expectation_value_vs_time for r in results),
        line_labels=(r.name for r in results),
        line_kwargs=({"linestyle": "-"}, {"linestyle": "--"}),
        x_label=r"Time $t$",
        x_unit="asec",
        y_label=r"Dipole Moment $d$",
        y_unit="atomic_electric_dipole_moment",
        **PLOT_KWARGS,
    )
