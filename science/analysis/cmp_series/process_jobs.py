import os
import itertools

import simulacra as si
from simulacra.units import *

import ionization as iclu

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=3)

if __name__ == "__main__":
    with si.utils.LogManager("simulacra", "ionization") as logger:
        jp_names = ["hyd__sinc", "fsw__sinc", "ide__sinc__len", "ide__sinc__vel"]

        jps = list(
            iclu.PulseJobProcessor.load(f"cmp__{jp_name}.job") for jp_name in jp_names
        )

        for jp in jps:
            print(jp)

        # pick up parameter sets from the first job (they should all be identical, by construction)
        selectors = list(
            dict(pulse_width=pw, fluence=flu)
            for pw, flu in itertools.product(
                jps[0].parameter_set("pulse_width"), jps[0].parameter_set("fluence")
            )
        )

        # for s in selectors:
        #     print(s)

        metrics = ["final_initial_state_overlap"]

        for metric, selector in itertools.product(metrics, selectors):
            x_by_job = [
                [r.phase for r in jp.select_by_kwargs(**selector)] for jp in jps
            ]
            y_by_job = [
                [getattr(r, metric) for r in jp.select_by_kwargs(**selector)]
                for jp in jps
            ]

            # print(x_by_job)
            # print(y_by_job)

            for log in [True, False]:
                si.vis.xxyy_plot(
                    f'{metric}__pw={uround(selector["pulse_width"], asec)}as_flu={uround(selector["fluence"], Jcm2)}jcm2'
                    + ("__log" if log else ""),
                    x_by_job,
                    y_by_job,
                    line_labels=[
                        n.replace("__", " ").replace("sinc", "").upper()
                        for n in jp_names
                    ],
                    x_label=r"$ \varphi $",
                    x_unit="rad",
                    y_label=metric.replace("_", " ").title(),
                    y_log_axis=log,
                    y_log_pad=2,
                    title=fr'$ \tau = {uround(selector["pulse_width"], asec)} \, \mathrm{{as}}$, $H = {uround(selector["fluence"], Jcm2)} \, \mathrm{{J/cm^2}}$',
                    **PLOT_KWARGS,
                )
