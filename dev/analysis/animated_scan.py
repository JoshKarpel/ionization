import os
import time

import simulacra as si
from simulacra.units import *


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

if __name__ == "__main__":
    with si.utils.LogManager("simulacra", "ionization"):
        jp = clu.JobProcessor.load("ide__sinc__25pw_20ph_20flu__v2.job")

        jp.plots_dir = OUT_DIR
        jp.summary_dir = OUT_DIR

        for pw in jp.parameter_set("pulse_width"):
            x = sorted(jp.parameter_set("phase"))
            fluences = sorted(jp.parameter_set("fluence"))

            def f(x, fluence):
                results = jp.select_by_kwargs(**{"fluence": fluence, "pulse_width": pw})

                return [r.final_bound_state_overlap for r in results]

            si.vis.xyt_plot(
                f"phase_fluence__pw={pw / asec:.3f}as",
                x,
                fluences,
                f,
                x_unit="rad",
                t_unit="Jcm2",
                t_fmt_string=r"$H = {}$",
                y_log_axis=False,
                length=10,
                target_dir=OUT_DIR,
            )

            time.sleep(1)
