import logging
import os

from tqdm import tqdm

import numpy as np

import simulacra as si
import simulacra.cluster as clu
import simulacra.units as u

import matplotlib.pyplot as plt

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

LOGMAN = si.utils.LogManager("simulacra", "ionization", stdout_level=logging.DEBUG)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)


def make_modulation_depth_heatmaps(jp):
    pulse_widths = jp.parameter_set("pulse_width")
    fluences = jp.parameter_set("fluence")

    for metric in jp.ionization_metrics:
        for log_z in (True, False):
            logger.debug(f"Making modulation depth plot for metric {metric}")

            results_cos = {
                (r.pulse_width, r.fluence): getattr(r, metric)
                for r in jp.select_by_kwargs(phase=0)
            }
            results_sin = {
                (r.pulse_width, r.fluence): getattr(r, metric)
                for r in jp.select_by_kwargs(phase=u.pi / 2)
            }

            def modulation_depth(pw, flu):
                c = results_cos[pw, flu]
                s = results_sin[pw, flu]

                return (c - s) / (c + s)

            x = np.array(sorted(pulse_widths))
            y = np.array(sorted(fluences))
            x_mesh, y_mesh = np.meshgrid(x, y, indexing="ij")

            z_mesh = np.empty_like(x_mesh)
            for i, pw in enumerate(x):
                for j, flu in enumerate(y):
                    z_mesh[i, j] = modulation_depth(pw, flu)

            si.vis.xyz_plot(
                f"{jp.name}_modulation_depth_heatmap__{metric}"
                + ("_logZ" if log_z else ""),
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
        jp_dir = os.path.join(os.path.dirname(__file__), "job_processors")
        jp_paths = [os.path.join(jp_dir, jp_name) for jp_name in os.listdir(jp_dir)]

        for jp_path in tqdm(jp_paths):
            print(jp_path)
            jp = clu.JobProcessor.load(jp_path)
            print(jp)

            try:
                make_modulation_depth_heatmaps(jp)
            except KeyError:
                pass
