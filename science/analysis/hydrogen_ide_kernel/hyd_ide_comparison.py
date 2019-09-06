import logging
import os
import itertools

import numpy as np

import simulacra as si
from simulacra.units import *

import ionization as clu

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

LOGMAN = si.utils.LogManager("simulacra", "ionization", stdout_level=logging.DEBUG)

PLOT_KWARGS = dict(
    target_dir=OUT_DIR,
    # img_format = 'png',
    # fig_dpi_scale = 6,
)


def version_1():
    jp_hyd = clu.PulseJobProcessor.load(
        "job_processors/hyd__pw_scan_v2__50-1000as_3flus_3phis__sinc.job"
    )
    jp_ide = clu.PulseJobProcessor.load("job_processors/ide__K_hyd__pw_scan__fast.job")

    print(jp_hyd)
    print(jp_ide)

    phases = sorted(jp_ide.parameter_set("phase"))
    fluences = sorted(jp_ide.parameter_set("fluence"))
    ide_alpha = 0.6

    styles = ["-", ":", "--"]
    colors = ["C0", "C1", "C2", "C3", "C4"]

    phase_to_style = dict(zip(phases, styles))
    fluence_to_color = dict(zip(fluences, colors))
    color_patches = [
        mpatches.Patch(
            color=color, label=fr"$ H = {fluence / Jcm2:3f} \, \mathrm{{J/cm^2}} $"
        )
        for fluence, color in fluence_to_color.items()
    ]

    phases_latex = [r"0", r"\pi / 4", r"\pi / 2"]
    style_patches = [
        mlines.Line2D(
            [],
            [],
            color="black",
            linestyle=style,
            linewidth=3,
            label=fr"$ \varphi = {phase_latex} $",
        )
        for phase, style, phase_latex in zip(phases, styles, phases_latex)
    ]

    legend_handles = color_patches + style_patches

    results_by_phase_and_fluence__hyd = {
        (phase, fluence): jp_hyd.select_by_kwargs(phase=phase, fluence=fluence)
        for phase in phases
        for fluence in fluences
    }
    results_by_phase_and_fluence__ide = {
        (phase, fluence): jp_ide.select_by_kwargs(phase=phase, fluence=fluence)
        for phase in phases
        for fluence in fluences
    }

    metrics = ["final_initial_state_overlap", "final_bound_state_overlap"]
    extra_line_kwargs = dict(linewidth=1, markevery=10)

    line_kwargs = [
        {
            "linestyle": phase_to_style[phase],
            "color": fluence_to_color[fluence],
            "marker": "D",
            "markersize": 2.5,
            **extra_line_kwargs,
        }
        for phase, fluence in results_by_phase_and_fluence__hyd.keys()
    ]
    line_kwargs += [
        {
            "linestyle": phase_to_style[phase],
            "color": fluence_to_color[fluence],
            "marker": "+",
            "markersize": 4,
            **extra_line_kwargs,
        }
        for phase, fluence in results_by_phase_and_fluence__ide.keys()
    ]

    for log_y in [True, False]:
        postfix = ""
        if log_y:
            postfix += "_logY"

        for metric in metrics:
            si.vis.xxyy_plot(
                f"hyd_ide_comparison__pulse_width_scan__sinc__{metric}" + postfix,
                [
                    *[
                        [r.pulse_width for r in results]
                        for results in results_by_phase_and_fluence__hyd.values()
                    ],
                    *[
                        [r.pulse_width for r in results]
                        for results in results_by_phase_and_fluence__ide.values()
                    ],
                ],
                [
                    *[
                        [getattr(r, metric) for r in results]
                        for results in results_by_phase_and_fluence__hyd.values()
                    ],
                    *[
                        [getattr(r, metric) for r in results]
                        for results in results_by_phase_and_fluence__ide.values()
                    ],
                ],
                line_kwargs=line_kwargs,
                title="Pulse Width Scan Comparison: Sinc Pulse",
                title_offset=1.075,
                x_label=r"Pulse Width $\tau$",
                x_unit="asec",
                y_label=metric.replace("final_", "").replace("_", " ").title(),
                y_log_axis=log_y,
                y_log_pad=2,
                legend_on_right=True,
                legend_kwargs={"handles": legend_handles},
                **PLOT_KWARGS,
                x_upper_limit=500 * asec,
                y_lower_limit=1e-5 if log_y else 0,
            )


def version_2():
    jp_hyd = clu.PulseJobProcessor.load(
        "job_processors/hyd__pw_scan_v2__50-1000as_3flus_3phis__sinc.job"
    )
    jp_ide = clu.PulseJobProcessor.load("job_processors/ide__K_hyd__pw_scan__fast.job")

    phases = sorted(jp_hyd.parameter_set("phase").union(jp_ide.parameter_set("phase")))
    fluences = sorted(
        jp_hyd.parameter_set("fluence").union(jp_ide.parameter_set("fluence"))
    )

    styles = ["-", ":", "--"]
    colors = ["C0", "C1", "C2", "C3", "C4"]

    phase_to_style = dict(zip(phases, styles))
    fluence_to_color = dict(zip(fluences, colors))
    color_patches = [
        mpatches.Patch(
            color=color, label=fr"$ H = {fluence / Jcm2:3f} \, \mathrm{{J/cm^2}} $"
        )
        for fluence, color in fluence_to_color.items()
    ]

    phases_latex = [r"0", r"\pi / 4", r"\pi / 2"]
    style_patches = [
        mlines.Line2D(
            [],
            [],
            color="black",
            linestyle=style,
            linewidth=3,
            label=fr"$ \varphi = {phase_latex} $",
        )
        for phase, style, phase_latex in zip(phases, styles, phases_latex)
    ]

    legend_handles = color_patches + style_patches

    results_by_phase_and_fluence__hyd = {
        (phase, fluence): jp_hyd.select_by_kwargs(phase=phase, fluence=fluence)
        for phase in phases
        for fluence in fluences
    }
    results_by_phase_and_fluence__ide = {
        (phase, fluence): jp_ide.select_by_kwargs(phase=phase, fluence=fluence)
        for phase in phases
        for fluence in fluences
    }

    metrics = ["final_bound_state_overlap", "final_initial_state_overlap"]
    extra_line_kwargs = dict(linewidth=1)

    line_kwargs_hyd = {
        (phase, fluence): {
            "linestyle": phase_to_style[phase],
            "color": fluence_to_color[fluence],
            **extra_line_kwargs,
        }
        for phase, fluence in results_by_phase_and_fluence__hyd.keys()
    }
    line_kwargs_ide = {
        (phase, fluence): {
            "linestyle": phase_to_style[phase],
            "color": fluence_to_color[fluence],
            **extra_line_kwargs,
        }
        for phase, fluence in results_by_phase_and_fluence__ide.keys()
    }

    for metric, log_y in itertools.product(metrics, [False, True]):
        postfix = ""
        if log_y:
            postfix += "_logY"

        with si.vis.FigureManager(
            f"hyd_ide_comparison__pulse_width_scan__sinc__{metric}" + postfix,
            aspect_ratio=1,
            **PLOT_KWARGS,
        ) as figman:
            fig = figman.fig

            gridspec = plt.GridSpec(2, 1, hspace=0.06)
            ax_hyd = fig.add_subplot(gridspec[0])
            ax_ide = fig.add_subplot(gridspec[1], sharey=ax_hyd)

            # TICKS, LEGEND, LABELS, and TITLE
            ax_hyd.tick_params(
                labeltop=True,
                labelright=True,
                labelbottom=False,
                labelleft=True,
                bottom=False,
            )
            ax_ide.tick_params(
                labeltop=False,
                labelright=True,
                labelbottom=True,
                labelleft=True,
                top=False,
            )

            legend = ax_hyd.legend(
                bbox_to_anchor=(1.15, 1),
                loc="upper left",
                borderaxespad=0.0,
                handles=legend_handles,
            )

            metric_label = metric.replace("final_", "").replace("_", " ").title()
            ax_hyd.set_ylabel(f"TDSE\n{metric_label}")
            ax_ide.set_ylabel(f"IDE\n{metric_label}")
            ax_ide.set_xlabel(fr"Pulse Width $ \tau \; \mathrm{{(as)}} $")
            suptitle = fig.suptitle("Pulse Width Scan Comparison: Sinc Pulse")
            suptitle.set_x(0.6)
            # suptitle.set_y(1)

            # LINES
            for phase, fluence in results_by_phase_and_fluence__hyd.keys():
                results = results_by_phase_and_fluence__hyd[(phase, fluence)]
                ax_hyd.plot(
                    [r.pulse_width / asec for r in results],
                    [getattr(r, metric) for r in results],
                    **line_kwargs_hyd[(phase, fluence)],
                )
            for phase, fluence in results_by_phase_and_fluence__ide.keys():
                results = results_by_phase_and_fluence__ide[(phase, fluence)]
                ax_ide.plot(
                    [r.pulse_width / asec for r in results],
                    [getattr(r, metric) for r in results],
                    **line_kwargs_ide[(phase, fluence)],
                )

            # LIMITS AND GRIDS
            grid_kwargs = si.vis.DEFAULT_GRID_KWARGS
            minor_grid_kwargs = si.vis.MINOR_GRID_KWARGS
            for ax in [ax_hyd, ax_ide]:
                ax.grid(True, which="major", **grid_kwargs)
                if log_y:
                    ax.set_yscale("log")
                    ax.grid(True, which="minor", axis="y", **minor_grid_kwargs)

                ax.set_xlim(0, 1000)
                ax.set_ylim(-0.025 if not log_y else 1e-5, 1.025)


def version_3():
    jp_hyd = clu.PulseJobProcessor.load(
        "job_processors/hyd__pw_scan_v2__50-1000as_3flus_3phis__sinc.job"
    )
    jp_ide = clu.PulseJobProcessor.load("job_processors/ide__K_hyd__pw_scan__fast.job")

    phases = sorted(jp_hyd.parameter_set("phase").union(jp_ide.parameter_set("phase")))
    fluences = sorted(
        jp_hyd.parameter_set("fluence").union(jp_ide.parameter_set("fluence"))
    )

    # styles = ['-', ':', '--']
    colors = ["C0", "C1", "C2", "C3", "C4"]

    # phase_to_style = dict(zip(phases, styles))
    fluence_to_color = dict(zip(fluences, colors))
    # color_patches = [mpatches.Patch(color = color, label = fr'$ H = {fluence / Jcm2:3f} \, \mathrm{{J/cm^2}} $')
    #                  for fluence, color in fluence_to_color.items()]

    # phases_latex = [r'0', r'\pi / 4', r'\pi / 2']
    # style_patches = [mlines.Line2D([], [], color = 'black', linestyle = style, linewidth = 3, label = fr'$ \varphi = {phase_latex} $')
    #                  for phase, style, phase_latex in zip(phases, styles, phases_latex)]

    # legend_handles = color_patches + style_patches

    results_by_phase_and_fluence__hyd = {
        (phase, fluence): jp_hyd.select_by_kwargs(phase=phase, fluence=fluence)
        for phase in phases
        for fluence in fluences
    }
    results_by_phase_and_fluence__ide = {
        (phase, fluence): jp_ide.select_by_kwargs(phase=phase, fluence=fluence)
        for phase in phases
        for fluence in fluences
    }

    hyd_cos = {
        fluence: results_by_phase_and_fluence__hyd[0, fluence] for fluence in fluences
    }
    hyd_sin = {
        fluence: results_by_phase_and_fluence__hyd[pi / 2, fluence]
        for fluence in fluences
    }
    ide_cos = {
        fluence: results_by_phase_and_fluence__ide[0, fluence] for fluence in fluences
    }
    ide_sin = {
        fluence: results_by_phase_and_fluence__ide[pi / 2, fluence]
        for fluence in fluences
    }

    metrics = ["final_bound_state_overlap", "final_initial_state_overlap"]
    extra_line_kwargs = dict(linewidth=1)

    line_kwargs = {
        fluence: {"color": fluence_to_color[fluence], **extra_line_kwargs}
        for fluence in fluences
    }

    for metric, log_y in itertools.product(metrics, [False, True]):
        postfix = ""
        if log_y:
            postfix += "_logY"

        with si.vis.FigureManager(
            f"hyd_ide_sine_cosine_diff__pulse_width_scan__sinc__{metric}" + postfix,
            aspect_ratio=1,
            **PLOT_KWARGS,
        ) as figman:
            fig = figman.fig

            gridspec = plt.GridSpec(2, 1, hspace=0.06)
            ax_hyd = fig.add_subplot(gridspec[0])
            ax_ide = fig.add_subplot(gridspec[1], sharey=ax_hyd)

            # TICKS, LABELS, and TITLE
            ax_hyd.tick_params(
                labeltop=True,
                labelright=True,
                labelbottom=False,
                labelleft=True,
                bottom=False,
            )
            ax_ide.tick_params(
                labeltop=False,
                labelright=True,
                labelbottom=True,
                labelleft=True,
                top=False,
            )

            metric_label = metric.replace("final_", "").replace("_", " ").title()
            ax_hyd.set_ylabel(f"TDSE\n{metric_label}\nModulation Depth $\Phi$")
            ax_ide.set_ylabel(f"IDE\n{metric_label}\nModulation Depth $\Phi$")
            ax_ide.set_xlabel(r"Pulse Width $ \tau \; \mathrm{{(as)}} $")
            suptitle = fig.suptitle("Pulse Width Scan Comparison: Sinc Pulse")
            suptitle.set_x(0.6)
            # suptitle.set_y(1)

            # LINES and LEGEND
            for fluence in fluences:
                cos = np.array(list(getattr(r, metric) for r in hyd_cos[fluence]))
                sin = np.array(list(getattr(r, metric) for r in hyd_sin[fluence]))
                y = (cos - sin) / (cos + sin)
                ax_hyd.plot(
                    [r.pulse_width / asec for r in hyd_cos[fluence]],
                    y,
                    label=rf"$ H = {fluence / Jcm2:3f} \, \mathrm{{J/cm^2}} $",
                    **line_kwargs[fluence],
                )
            for fluence in fluences:
                cos = np.array(list(getattr(r, metric) for r in ide_cos[fluence]))
                sin = np.array(list(getattr(r, metric) for r in ide_sin[fluence]))
                y = (cos - sin) / (cos + sin)
                ax_ide.plot(
                    [r.pulse_width / asec for r in ide_cos[fluence]],
                    y,
                    label=rf"$ H = {fluence / Jcm2:3f} \, \mathrm{{J/cm^2}} $",
                    **line_kwargs[fluence],
                )
            legend = ax_hyd.legend(
                bbox_to_anchor=(1.15, 1), loc="upper left", borderaxespad=0.0
            )

            # LIMITS AND GRIDS
            grid_kwargs = si.vis.DEFAULT_GRID_KWARGS
            minor_grid_kwargs = si.vis.MINOR_GRID_KWARGS
            for ax in [ax_hyd, ax_ide]:
                ax.grid(True, which="major", **grid_kwargs)
                if log_y:
                    ax.set_yscale("symlog", linthreshy=1e-3)
                    ax.grid(True, which="minor", axis="y", **minor_grid_kwargs)

                ax.set_xlim(0, 1000)
                # if not log_y:
                #     ax.set_ylim(-1, 10)
                # ax.set_ylim(-0.025 if not log_y else 1e-5, 1.025)


if __name__ == "__main__":
    with LOGMAN as logger:
        # version_1()
        version_2()
        version_3()
