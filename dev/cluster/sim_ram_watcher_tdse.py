import logging
import os
import psutil
import subprocess
import time

import numpy as np

import simulacra as si
import simulacra.units as u

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)


def run_sim_in_proc(R, ppR, L, T, maxE, maxL):
    process = subprocess.Popen(
        [
            "python",
            "sim_ram_runner_tdse.py",
            str(R),
            str(ppR),
            str(L),
            str(T),
            str(maxE),
            str(maxL),
        ]
    )
    return process


if __name__ == "__main__":
    with si.utils.LogManager(
        "simulacra", "ionization", stdout_logs=True, stdout_level=logging.INFO
    ) as logger:
        R = 200
        ppR = 20
        L = 1000
        T = 50
        maxE = 20
        maxL = 10

        times = []
        rss = []
        vms = []
        proc = run_sim_in_proc(R, ppR, L, T, maxE, maxL)
        ps_proc = psutil.Process(proc.pid)

        while proc.poll() is None:
            try:
                t, mem_info = time.time(), ps_proc.memory_info()
                times.append(t)
                rss.append(mem_info.rss)
                vms.append(mem_info.vms)
            except psutil.NoSuchProcess:
                pass

            time.sleep(1 * u.usec)

        times = np.array(times) - times[0]

        si.vis.xy_plot(
            f"rss_vs_time__R={R}_ppR={ppR}_L={L}_T={T}_E={maxE}_L={maxL}",
            times,
            rss,
            vms,
            line_labels=["RSS", "VMS"],
            x_label="Time",
            x_unit="s",
            y_label="Memory Usage (MB)",
            y_unit=(1024 ** 2),
            # x_lower_limit = 25,
            # x_upper_limit = 38,
            **PLOT_KWARGS,
        )

        print("max rss", max(rss))
        print("max vms", max(vms))
