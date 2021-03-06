import os

import simulacra as si
from simulacra.units import *


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

if __name__ == "__main__":
    with si.utils.LogManager("simulacra", "ionization") as logger:
        jp_lide = si.cluster.JobProcessor.load("compare_to_velocity.job")
        jp_vide = si.cluster.JobProcessor.load("vide_compare.job")

        results_lide = list(jp_lide.data.values())
        results_vide = list(jp_vide.data.values())

        PLOT_KWARGS = dict(x_unit="rad", target_dir=OUT_DIR)

        for log in (True, False):
            postfix = ""
            if log:
                postfix += "__log"

            si.vis.xy_plot(
                f"ionization_vs_phase__length" + postfix,
                [r.phase for r in results_lide],
                [r.final_bound_state_overlap for r in results_lide],
                y_log_axis=log,
                **PLOT_KWARGS,
            )

            si.vis.xy_plot(
                f"ionization_vs_phase__velocity" + postfix,
                [r.phase for r in results_vide],
                [r.final_bound_state_overlap for r in results_vide],
                y_log_axis=log,
                **PLOT_KWARGS,
            )

            si.vis.xy_plot(
                f"ionization_vs_phase__compare" + postfix,
                [r.phase for r in results_vide],
                [r.final_bound_state_overlap for r in results_lide],
                [r.final_bound_state_overlap for r in results_vide],
                line_labels=("Length", "Velocity"),
                y_log_axis=log,
                **PLOT_KWARGS,
            )

            rel_lide = results_lide[0].final_bound_state_overlap
            rel_vide = results_vide[0].final_bound_state_overlap

            si.vis.xy_plot(
                f"ionization_vs_phase__compare_rel" + postfix,
                [r.phase for r in results_vide],
                [r.final_bound_state_overlap / rel_lide for r in results_lide],
                [r.final_bound_state_overlap / rel_vide for r in results_vide],
                line_labels=("Length (Rel.)", "Velocity (Rel.)"),
                y_log_axis=log,
                **PLOT_KWARGS,
            )
