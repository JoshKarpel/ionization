import os

import simulacra as si
from simulacra.units import *

import ionization as ion
import ionization as iclu

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)
JP_DIR = os.path.join(os.getcwd(), "job_processors")

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=3)

if __name__ == "__main__":
    with si.utils.LogManager("simulacra", "ionization") as logger:
        # jp_path = os.path.join(JP_DIR, 'hyd__amp_scan_v2_fast.job')
        jp_path = os.path.join(JP_DIR, "emergency_ide_scan_3__amplitude_scan.job")
        jp = iclu.PulseJobProcessor.load(jp_path)

        print(jp)

        pulse_widths = sorted(jp.parameter_set("pulse_width"))
        phases = sorted(jp.parameter_set("phase"))
        amplitudes = sorted(jp.parameter_set("amplitude"))

        print(len(jp.data))

        print(len(pulse_widths), len(phases), len(amplitudes))

        # metrics = ['final_initial_state_overlap', 'final_bound_state_overlap']
        metrics = ["final_bound_state_overlap"]
        # metric_names = ['Initial State Overlap', 'Bound State Overlap']
        metric_names = ["Bound State Overlap"]

        for metric, metric_name in zip(metrics, metric_names):
            for idx, amplitude in enumerate(
                amplitudes
            ):  # lexical sort is not good enough
                print(f"AMPLITUDE {amplitude / atomic_electric_field:3f}")
                results = {
                    (r.pulse_width, r.phase): r
                    for r in jp.select_by_kwargs(amplitude=amplitude)
                }

                pws = sorted(set(pw for pw, cep in results))
                ceps = sorted(set(cep for pw, cep in results))

                print(len(pws))

                if len(pws) < 50:
                    continue

                for log_y in (True, False):
                    si.vis.xxyy_plot(
                        metric.upper()
                        + ("_logY_" if log_y else "")
                        + f"_{idx}_"
                        + f"__amp={amplitude / atomic_electric_field:3f}aef",
                        [
                            *[
                                [results[pw, cep].pulse_width for pw in pws]
                                for cep in ceps
                            ]
                        ],
                        [
                            *[
                                [getattr(results[pw, cep], metric) for pw in pws]
                                for cep in ceps
                            ]
                        ],
                        line_labels=[
                            fr"$\varphi = {phase / pi:3f}\pi$" for phase in ceps
                        ],
                        x_label=r"$\tau$",
                        x_unit="asec",
                        y_label=metric_name,
                        title=fr"Constant Amplitude Scan: ${ion.vis.LATEX_EFIELD}_0 = {amplitude / atomic_electric_field:3f} \, \mathrm{{a.u.}}$",
                        y_lower_limit=None if log_y else 0,
                        y_log_axis=log_y,
                        **PLOT_KWARGS,
                    )
