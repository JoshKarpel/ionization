import os

import simulacra as si
import simulacra.cluster as clu

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=2)

if __name__ == "__main__":
    with si.utils.LogManager("simulacra", "ionization") as logger:
        jp_names = [
            (
                "hyd__pw=200as_flu=1jcm2.job",
                "ide__sinc__pw=200as_flu=1jcm2__len__fast.job",
                "ide__sinc__pw=200as_flu=1jcm2__vel__fast.job",
            ),
            (
                "hyd__pw=200as_flu=10jcm2.job",
                "ide__sinc__pw=200as_flu=10jcm2__len__fast.job",
                "ide__sinc__pw=200as_flu=10jcm2__vel__fast.job",
            ),
        ]
        for jp_hyd_name, jp_ide_name_len, jp_ide_name_vel in jp_names:
            jp_hyd = clu.JobProcessor.load(jp_hyd_name)
            jp_ide_len = clu.JobProcessor.load(jp_ide_name_len)
            jp_ide_vel = clu.JobProcessor.load(jp_ide_name_vel)

            print(jp_hyd)
            print(jp_ide_len)
            print(jp_ide_vel)

            jp_hyd.job_dir_path = os.path.join(OUT_DIR, jp_hyd_name)
            jp_ide_len.job_dir_path = os.path.join(OUT_DIR, jp_ide_name_len)
            jp_ide_vel.job_dir_path = os.path.join(OUT_DIR, jp_ide_name_vel)

            # automatic plots
            # jp_hyd.make_pulse_parameter_scans_1d()
            # jp_ide.make_pulse_parameter_scans_1d()

            # more careful plots
            results_hyd = list(jp_hyd.data.values())
            results_ide_len = list(jp_ide_len.data.values())
            results_ide_vel = list(jp_ide_vel.data.values())

            metrics = [
                "final_norm",
                "final_bound_state_overlap",
                "final_initial_state_overlap",
            ]
            labels = [
                r"$ \left\langle \Psi | \Psi \right\rangle $",
                r"$ \left\langle \Psi | \psi_{\mathrm{bound}} \right\rangle $",
                r"$ \left\langle \Psi | \psi_{\mathrm{initial}} \right\rangle $",
                r"Len IDE Bound State",
                r"Vel IDE Bound State",
            ]

            base_title = f"metrics_vs_phase__{jp_hyd_name}"

            for log in (True, False):
                si.vis.xy_plot(
                    base_title + f"__log={log}",
                    list(r.phase for r in results_hyd),
                    *(
                        list(getattr(r, metric) for r in results_hyd)
                        for metric in metrics
                    ),
                    list(r.final_bound_state_overlap for r in results_ide_len),
                    list(r.final_bound_state_overlap for r in results_ide_vel),
                    line_kwargs=[None, None, None, None, {"linestyle": "--"}],
                    line_labels=labels,
                    x_label=r"CEP $\varphi$",
                    x_unit="rad",
                    y_log_axis=log,
                    y_log_pad=2,
                    **PLOT_KWARGS,
                )
