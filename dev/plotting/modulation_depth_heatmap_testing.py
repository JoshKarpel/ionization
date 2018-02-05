import logging
import os
import itertools

import numpy as np

import simulacra as si
from simulacra.units import *

import ionization as ion
import ionization.cluster as iclu

import matplotlib.pyplot as plt
import matplotlib.colors

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

LOGMAN = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.DEBUG)

PLOT_KWARGS = [
    dict(
        target_dir = OUT_DIR,
        img_format = 'png',
        fig_dpi_scale = 6,
    ),
    dict(
        target_dir = OUT_DIR,
    ),
]


def diff_over_avg(cosine, sine):
    return 2 * (cosine - sine) / (cosine + sine)


def cos_over_sin(cosine, sine):
    return cosine / sine


def sin_over_cos(cosine, sine):
    return sine / cosine


def diff_ionized_over_avg_ionized(cosine, sine):
    return 2 * (cosine - sine) / (2 - cosine - sine)


def cos_over_sin_minus_sin_over_cos(cosine, sine):
    return (cosine / sine) - (sine / cosine)


def log_over_log(cosine, sine):
    return (np.log10(cosine) - np.log10(sine)) / (np.log10(cosine) + np.log10(sine))


def log_over_log_normed(cosine, sine):
    return (np.log10(cosine) - np.log10(sine)) / np.sqrt((np.log10(cosine) ** 2) + (np.log10(sine) ** 2))


def hyperbolic(cosine, sine):
    sign = -np.sign(cosine - sine)

    diff_abs = np.abs(cosine - sine)
    diff_frac = np.abs(cosine / sine)
    diff_frac_log = np.abs(np.log(np.abs(cosine / sine)))

    always_geq1_diff_frac = np.where(diff_frac >= 1, diff_frac, 1 / diff_frac)

    # diff_abs_norm = diff_abs / np.nanmax(diff_abs)
    # diff_frac_norm = diff_frac / np.nanmax(diff_frac)
    # diff_frac_log_norm = diff_frac_log / np.nanmax(diff_frac_log)

    # C = sign * np.abs((diff_abs_norm - diff_frac_norm) / (diff_abs_norm + diff_frac_norm))

    # print(diff_frac, diff_frac_log)
    C = sign * np.where(always_geq1_diff_frac <= 10, always_geq1_diff_frac / 10, diff_frac_log / np.max(diff_frac_log))
    # C /= np.nanmax(C)

    # C = sign * np.maximum(diff_frac, diff_frac_log)

    return C


FUNC_TO_TEX = {
    diff_over_avg: r'$ M = 2 \frac{C-S}{C+S} $',
    cos_over_sin: r'$ M = \frac{C}{S} $',
    sin_over_cos: r'$ M = \frac{S}{C} $',
    diff_ionized_over_avg_ionized: r'$ M = 2\frac{C - S}{2 - S - C}$',
    cos_over_sin_minus_sin_over_cos: r'$ M = \frac{C}{S} - \frac{S}{C} $',
    log_over_log: r'$ M = \frac{\log C - \log S}{\log C + \log S} $',
    log_over_log_normed: r'$ M = \frac{\log C - \log S}{\sqrt{(\log C)^2 + (\log S)^2}} $',
    hyperbolic: r'$ M = $ Crazy Stuff',
}

LOG_TO_FUNC = {
    True: np.geomspace,
    False: np.linspace,
}


def make_heatmap(modulation_depth_func, plt_kwargs):
    N = 200

    for log_xy, log_z in itertools.product((False, True), repeat = 2):
        cosine = LOG_TO_FUNC[log_xy](1e-6, 1 - 1e-6, N)
        sine = LOG_TO_FUNC[log_xy](1e-6, 1 - 1e-6, N)
        cosine_mesh, sine_mesh = np.meshgrid(cosine, sine, indexing = 'ij')
        z_mesh = modulation_depth_func(cosine_mesh, sine_mesh)

        log_str = iclu.format_log_str(
            (log_xy, log_z),
            ('XY', 'Z'),
        )

        figman = si.vis.xyz_plot(
            modulation_depth_func.__name__ + log_str,
            cosine_mesh, sine_mesh, z_mesh,
            x_label = r'$C$ (Cosine Remaining)',
            y_label = r'$S$ (Sine Remaining)',
            z_label = r'$M$',
            title = FUNC_TO_TEX[modulation_depth_func],
            colormap = plt.get_cmap('RdBu_r'),
            # colormap = plt.get_cmap('coolwarm'),
            x_lower_limit = 1e-6,
            y_lower_limit = 1e-6,
            x_upper_limit = 1,
            y_upper_limit = 1,
            x_log_axis = log_xy,
            y_log_axis = log_xy,
            z_log_axis = log_z,
            # z_lower_limit = -np.max(np.abs(z_mesh)),
            # z_upper_limit = np.max(np.abs(z_mesh)),
            save_on_exit = False,
            close_after_exit = False,
            **plt_kwargs,
        )

        ax = figman.fig.axes[0]

        text_kwargs = dict(
            transform = ax.transAxes,
            bbox = dict(color = 'green', alpha = 0.5),
        )

        ax.text(
            x = 1.1,
            y = -.2,
            s = 'Sin ionized\nCos intact',
            **text_kwargs
        )
        ax.text(
            x = 1.1,
            y = 1.1,
            s = 'Sin intact\nCos intact',
            **text_kwargs
        )
        ax.text(
            x = -.375,
            y = 1.1,
            s = 'Sin intact\nCos ionized',
            **text_kwargs
        )
        ax.text(
            x = -.375,
            y = -.2,
            s = 'Sin ionized\nCos ionized',
            **text_kwargs
        )

        figman.save()
        figman.cleanup()


# class HyperbolicColormap(matplotlib.colors.Colormap):
#     def __init__(self):
#         self.name = 'hyperbolic'
#         self.N = 256
#
#     def __call__(self, x, alpha = 1, bytes = False):
#         diff_abs = np.abs(x)
#         diff_

if __name__ == '__main__':
    with LOGMAN as logger:
        for func in FUNC_TO_TEX.keys():
            for plt_kwargs in PLOT_KWARGS:
                make_heatmap(func, plt_kwargs)

        x = np.linspace(0.0001, 1, 100)[1:]
        C_max = 20
        Cs = np.linspace(.01, 1, 10)
        curves = [C / x for C in Cs]
        print(curves)

        si.vis.xy_plot(
            'curves',
            x,
            *curves,
            x_label = 'abs diff',
            y_label = 'frac diff',
            line_labels = Cs,
            line_kwargs = [{'color': str(C / C_max)} for C in Cs],
            y_log_axis = True,
            **PLOT_KWARGS[0]
        )
