import os

import simulacra as si
import simulacra.cluster as clu
from simulacra.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=2)

if __name__ == "__main__":
    with si.utils.LogManager("simulacra", "ionization") as logger:
        jp_names = ["dense_fluences__hyd__100-200-400as__sine_and_cosine.job"]
        for jp_name in jp_names:
            jp = clu.JobProcessor.load(jp_name)
            jp.job_dir_path = OUT_DIR

            print(jp)
            # jp.summarize()

            phases = list(jp.parameter_set("phase"))
            pulse_widths = sorted(list(jp.parameter_set("pulse_width")))
            print(phases)
            print(pulse_widths)

            results = jp.select_by_kwargs(phase=phases[0], pulse_width=pulse_widths[1])
            print(results)
            for r in results:
                print(
                    r.file_name,
                    r.phase,
                    r.pulse_width / asec,
                    r.fluence / Jcm2,
                    r.final_initial_state_overlap,
                    r.final_bound_state_overlap,
                )
