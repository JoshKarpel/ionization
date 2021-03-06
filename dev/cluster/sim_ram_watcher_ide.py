import logging
import os
import psutil
import subprocess
import time

import numpy as np

import simulacra as si
from simulacra.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)


PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)


def run_sim_in_proc(dt):
    process = subprocess.Popen(["python", "sim_ram_runner_ide.py", str(dt)])
    return process


if __name__ == "__main__":
    with si.utils.LogManager(
        "simulacra", "ionization", stdout_logs=True, stdout_level=logging.INFO
    ) as logger:
        dt = 0.01

        times = []
        rss = []
        vms = []
        proc = run_sim_in_proc(dt)
        ps_proc = psutil.Process(proc.pid)

        while proc.poll() is None:
            try:
                t, mem_info = time.time(), ps_proc.memory_info()
                times.append(t)
                rss.append(mem_info.rss)
                vms.append(mem_info.vms)
            except psutil.NoSuchProcess:
                pass

            time.sleep(10 * usec)

        times = np.array(times) - times[0]

        steps = 100 * 20 / dt

        si.vis.xy_plot(
            f"rss_vs_time__dt={dt}as_steps={round(steps)}",
            times,
            rss,
            vms,
            line_labels=["RSS", "VMS"],
            x_label="Time",
            x_unit="s",
            y_label="Memory Usage (MB)",
            y_unit=(1024 ** 2),
            **PLOT_KWARGS,
        )
