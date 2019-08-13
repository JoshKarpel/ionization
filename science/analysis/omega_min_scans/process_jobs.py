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


def get_omega_min(sim_result):
    return sim_result.electric_potential[0].omega_min


def get_omega_carrier(sim_result):
    return sim_result.electric_potential[0].omega_carrier


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

        jp_names = [
            "hyd__omega_min_scan__sinc.job",
            "hyd__omega_min_scan__gaussian__fixed_bounds.job",
        ]
        for jp_name in jp_names:
            jp = clu.JobProcessor.load(jp_name)
            jp.job_dir_path = OUT_DIR

            target_dir = os.path.join(OUT_DIR, jp.name)

            print(jp)
            # jp.summarize()

            phases = sorted(list(jp.parameter_set("phase")))
            pulse_widths = sorted(list(jp.parameter_set("pulse_width")))
            fluences = sorted(list(jp.parameter_set("fluence")))
            print(phases)
            print(pulse_widths)
            print(fluences)

            # results = jp.select_by_kwargs(phase = phases[0], pulse_width = pulse_widths[1], fluence = fluences[0])
            # print(results)
            # for r in results:
            #     print(r.file_name, r.phase, r.pulse_width / asec, r.fluence / Jcm2, r.electric_potential[0].omega_min, r.final_initial_state_overlap, r.final_bound_state_overlap)

            metrics = ["final_initial_state_overlap", "final_bound_state_overlap"]
            for metric in metrics:
                for pulse_width, fluence in itertools.product(pulse_widths, fluences):
                    selector = dict(pulse_width=pulse_width, fluence=fluence)

                    metric_name = metric.replace("_", " ").title()

                    si.vis.xxyy_plot(
                        f"{metric}__pw={uround(pulse_width, asec)}as_flu={uround(fluence, Jcm2)}",
                        [
                            *[
                                [
                                    get_omega_carrier(r) / twopi
                                    for r in jp.select_by_kwargs(
                                        phase=phase, **selector
                                    )
                                ]
                                for phase in phases
                            ]
                        ],
                        [
                            *[
                                [
                                    getattr(r, metric)
                                    for r in jp.select_by_kwargs(
                                        phase=phase, **selector
                                    )
                                ]
                                for phase in phases
                            ]
                        ],
                        line_labels=[
                            fr"$\varphi = {uround(phase, pi)}\pi$" for phase in phases
                        ],
                        x_label=r"$f_{\mathrm{carrier}}$",
                        x_unit="THz",
                        # y_label = metric_name,
                        title=fr"{metric_name}: $\tau = {uround(pulse_width, asec)} \, \mathrm{{as}}, H = {uround(fluence, Jcm2)} \, \mathrm{{J/cm^2}}$",
                        vlines=transition_frequencies,
                        font_size_title=20,
                        font_size_axis_labels=16,
                        font_size_tick_labels=14,
                        font_size_legend=16,
                        **PLOT_KWARGS,
                        target_dir=target_dir,
                    )
