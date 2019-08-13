import logging
import os

import numpy as np

import simulacra as si
import simulacra.cluster as clu
from simulacra.units import *

import matplotlib.pyplot as plt

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

LOGMAN = si.utils.LogManager("simulacra", "ionization", stdout_level=logging.DEBUG)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)

METRICS = ["final_bound_state_overlap", "final_initial_state_overlap", "final_norm"]


def make_heatmaps_for_each_cep(jp):
    pulse_widths = jp.parameter_set("pulse_width")
    fluences = jp.parameter_set("fluence")
    ceps = jp.parameter_set("phase")

    for metric in METRICS:
        for cep in ceps:
            for log_z in (True, False):
                logger.debug(
                    f"Making plot for CEP {uround(cep, pi)}pi, metric {metric}"
                )

                results = {
                    (r.pulse_width, r.fluence): getattr(r, metric)
                    for r in jp.select_by_kwargs(phase=cep)
                }

                x = np.array(sorted(pulse_widths))
                y = np.array(sorted(fluences))
                x_mesh, y_mesh = np.meshgrid(x, y, indexing="ij")

                z_mesh = np.empty_like(x_mesh)
                for i, pw in enumerate(x):
                    for j, flu in enumerate(y):
                        z_mesh[i, j] = results[pw, flu]

                si.vis.xyz_plot(
                    f'ION_MAP__{metric}_{"logZ" if log_z else ""}_cep={uround(cep, pi, 2)}pi',
                    x_mesh,
                    y_mesh,
                    z_mesh,
                    x_unit="asec",
                    y_unit="Jcm2",
                    z_lower_limit=1e-6 if log_z else 0,
                    z_upper_limit=1,
                    z_log_axis=log_z,
                    x_label=r"$\tau$",
                    y_label=r"$H$",
                    z_label=metric,
                    title="Pulse Width vs. Fluence Ionization Map",
                    **PLOT_KWARGS,
                )


def make_modulation_depth_heatmaps(jp):
    pulse_widths = jp.parameter_set("pulse_width")
    fluences = jp.parameter_set("fluence")

    for metric in METRICS:
        for log_z in (True, False):
            logger.debug(f"Making modulation depth plot for metric {metric}")

            results_cos = {
                (r.pulse_width, r.fluence): getattr(r, metric)
                for r in jp.select_by_kwargs(phase=0)
            }
            results_sin = {
                (r.pulse_width, r.fluence): getattr(r, metric)
                for r in jp.select_by_kwargs(phase=pi / 2)
            }

            def m(pw, flu):
                c = results_cos[pw, flu]
                s = results_sin[pw, flu]

                return (c - s) / (c + s)

            x = np.array(sorted(pulse_widths))
            y = np.array(sorted(fluences))
            x_mesh, y_mesh = np.meshgrid(x, y, indexing="ij")

            z_mesh = np.empty_like(x_mesh)
            for i, pw in enumerate(x):
                for j, flu in enumerate(y):
                    z_mesh[i, j] = m(pw, flu)

            si.vis.xyz_plot(
                f'MOD_DEPTH__{metric}{"_logZ" if log_z else ""}',
                x_mesh,
                y_mesh,
                z_mesh,
                x_unit="asec",
                y_unit="Jcm2",
                z_lower_limit=-1,
                z_upper_limit=1,
                z_log_axis=log_z,
                x_label=r"$\tau$",
                y_label=r"$H$",
                z_label=f"Modulation Depth\n{metric}",
                title="Pulse Width vs. Fluence Modulation Map",
                colormap=plt.get_cmap("RdBu_r"),
                sym_log_norm_epsilon=0.1,
                **PLOT_KWARGS,
            )


if __name__ == "__main__":
    with LOGMAN as logger:
        jp = clu.JobProcessor.load(
            os.path.join(
                os.getcwd(), "job_processors", "hyd__heatmap_PW_vs_FLU_2ceps.job"
            )
        )

        make_heatmaps_for_each_cep(jp)
        make_modulation_depth_heatmaps(jp)
