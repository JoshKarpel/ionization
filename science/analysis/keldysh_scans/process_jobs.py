import os
import itertools

import numpy as np

import simulacra as si
import simulacra.cluster as clu
from simulacra.units import *

import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

PLOT_KWARGS = dict(fig_width=si.vis.points_to_inches(500))


def get_keldysh_parameter(sim_result, selector="bandwidth"):
    return round(
        sim_result.electric_potential[0].keldysh_parameter(
            keldysh_omega_selector=selector
        ),
        3,
    )


if __name__ == "__main__":
    with si.utils.LogManager("simulacra", "ionization") as logger:
        states = list(ion.HydrogenBoundState(n, 0) for n in range(1, 10))
        transition_energies = set(
            np.abs(a.energy - b.energy) for a, b in itertools.product(states, repeat=2)
        )
        transition_frequencies = [energy / h for energy in transition_energies]
        print(sorted(transition_energies))
        for energy in sorted(transition_energies):
            print(energy / eV, energy / h / THz)

        jp_names = ["hyd__keldysh_scan__sinc.job", "hyd__keldysh_scan__gaussian.job"]
        for jp_name in jp_names:
            jp = clu.JobProcessor.load(jp_name)
            jp.job_dir_path = OUT_DIR

            if "sinc" in jp.name:
                selector = "bandwidth"
            elif "gaussian" in jp.name:
                selector = "carrier"

            target_dir = os.path.join(OUT_DIR, jp.name)

            print(jp)
            # jp.summarize()

            phases = sorted(list(jp.parameter_set("phase")))
            pulse_widths = sorted(list(jp.parameter_set("pulse_width")))
            keldysh_parameters = sorted(
                set(
                    get_keldysh_parameter(r, selector=selector)
                    for r in jp.data.values()
                )
            )
            print(phases)
            print(pulse_widths)
            print(keldysh_parameters)

            # results = jp.select_by_kwargs(phase = phases[0], pulse_width = pulse_widths[1], fluence = fluences[0])
            # print(results)
            # for r in results:
            #     print(r.file_name, r.phase, r.pulse_width / asec, r.fluence / Jcm2, r.electric_potential[0].omega_min, r.final_initial_state_overlap, r.final_bound_state_overlap)

            metrics = ["final_initial_state_overlap", "final_bound_state_overlap"]
            for metric in metrics:
                for keldysh_parameter in keldysh_parameters:
                    results = [
                        r
                        for r in jp.data.values()
                        if get_keldysh_parameter(r, selector=selector)
                        == keldysh_parameter
                    ]

                    metric_name = metric.replace("_", " ").title()

                    si.vis.xxyy_plot(
                        f"{metric}__gamma={keldysh_parameter}",
                        [
                            *[
                                [r.pulse_width for r in results if r.phase == phase]
                                for phase in phases
                            ]
                        ],
                        [
                            *[
                                [
                                    getattr(r, metric)
                                    for r in results
                                    if r.phase == phase
                                ]
                                for phase in phases
                            ]
                        ],
                        line_labels=[
                            fr"$\varphi = {uround(phase, pi)}\pi$" for phase in phases
                        ],
                        x_label=r"$\tau$",
                        x_unit="asec",
                        # y_label = metric_name,
                        title=fr"{metric_name}: $\gamma = {keldysh_parameter}$",
                        font_size_title=20,
                        font_size_axis_labels=16,
                        font_size_tick_labels=14,
                        font_size_legend=16,
                        **PLOT_KWARGS,
                        target_dir=target_dir,
                    )
