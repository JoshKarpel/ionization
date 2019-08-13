import logging
import os
import itertools

import simulacra as si
import simulacra.cluster as clu
from simulacra.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)
JP_DIR = os.path.join(os.getcwd(), "job_processors")

LOGMAN = si.utils.LogManager("simulacra", "ionization", stdout_level=logging.DEBUG)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)

METRICS = ["final_initial_state_overlap"]


def make_scans(jp_name, SCAN_ATTR):
    jp = clu.JobProcessor.load(os.path.join(JP_DIR, jp_name))

    grouping_attrs = ["pulse_width", "fluence", "phase"]
    units = dict(zip(grouping_attrs, ["asec", "Jcm2", "rad"]))
    symbols = dict(zip(grouping_attrs, [r"\tau", r"H", r"\varphi"]))

    plot_kwargs = {**PLOT_KWARGS, "target_dir": os.path.join(OUT_DIR, jp.name)}

    for metric in METRICS:
        for plot_attr, line_attr, scan_attr in itertools.permutations(grouping_attrs):
            if scan_attr != SCAN_ATTR:
                continue
            for plot_attr_value in jp.parameter_set(plot_attr):
                line_attr_values = sorted(jp.parameter_set(line_attr))
                scan_attr_values = sorted(jp.parameter_set(scan_attr))
                results = {
                    (getattr(r, line_attr), getattr(r, scan_attr)): r
                    for r in jp.select_by_kwargs(**{plot_attr: plot_attr_value})
                }

                plot_unit, plot_unit_tex = get_unit_value_and_latex_from_unit(
                    units[plot_attr]
                )
                line_unit, line_unit_tex = get_unit_value_and_latex_from_unit(
                    units[line_attr]
                )

                for log_y in [False, True]:
                    si.vis.xxyy_plot(
                        ("logY_" if log_y else "")
                        + f"{plot_attr}={uround(plot_attr_value, units[plot_attr])}_{line_attr}_{scan_attr}",
                        [
                            *[
                                [
                                    getattr(
                                        results[line_attr_value, scan_attr_value],
                                        scan_attr,
                                    )
                                    for scan_attr_value in scan_attr_values
                                ]
                                for line_attr_value in line_attr_values
                            ]
                        ],
                        [
                            *[
                                [
                                    getattr(
                                        results[line_attr_value, scan_attr_value],
                                        metric,
                                    )
                                    for scan_attr_value in scan_attr_values
                                ]
                                for line_attr_value in line_attr_values
                            ]
                        ],
                        line_labels=[
                            rf"$ {symbols[line_attr]} = {uround(line_attr_value, units[line_attr])} \, {line_unit_tex} $"
                            for line_attr_value in line_attr_values
                        ],
                        legend_on_right=True,
                        x_unit=units[scan_attr],
                        x_label=rf"${symbols[scan_attr]}$",
                        y_label=metric.replace("_", " ").title(),
                        title=rf"$ {symbols[plot_attr]} = {uround(plot_attr_value, units[plot_attr])} \, {plot_unit_tex} $",
                        y_log_axis=log_y,
                        y_lower_limit=None if log_y else 0,
                        y_upper_limit=1.1,
                        y_log_pad=1.1,
                        **plot_kwargs,
                    )


if __name__ == "__main__":
    with LOGMAN as logger:
        # make_scans('ide__cep_scan__fast.job', 'phase')
        # make_scans('ide__pw_scan__fast.job', 'pulse_width')
        make_scans("emergency_ide_scan.job", "pulse_width")
        make_scans("emergency_ide_scan_2.job", "pulse_width")
        make_scans("emergency_ide_scan_3__amplitude_scan.job", "pulse_width")
