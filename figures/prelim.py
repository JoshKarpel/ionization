import functools as ft
import logging
import os
import sys

import matplotlib
import numpy as np
import scipy.optimize as optimize
from tqdm import tqdm

matplotlib.use('pgf')

import simulacra as si
import ionization as ion
import ionization.integrodiff as ide
from simulacra.units import *

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

logman = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.INFO)

pgf_with_latex = {  # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
    "text.usetex": True,  # use LaTeX to write all text
    "text.latex.unicode": True,
    "font.family": "serif",
    "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    # "axes.labelsize": 11,  # LaTeX default is 10pt font.
    # "font.size": 11,
    # "legend.fontsize": 10,  # Make the legend/label fonts a little smaller
    # "xtick.labelsize": 9,
    # "ytick.labelsize": 9,
    # "figure.figsize": si.vis._get_fig_dims(0.95),  # default fig size of 0.95 \textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",  # use utf8 fonts because your computer can handle it :)
        r"\usepackage[T1]{fontenc}",  # plots will be generated using this preamble
    ]
}
matplotlib.rcParams.update(pgf_with_latex)

import matplotlib.pyplot as plt

PPT_WIDTH = 13 + (1 / 3)
PPT_HEIGHT = 7.5

PPT_ASPECT_RATIO = PPT_HEIGHT / PPT_WIDTH
PPT_WIDTH_PTS = 960

FIGMAN_KWARGS = dict(
    fig_width_pts = PPT_WIDTH_PTS,
    aspect_ratio = PPT_ASPECT_RATIO,
)

PLOT_KWARGS = dict(
    img_format = 'png',
    fig_dpi_scale = 3,
    target_dir = OUT_DIR,
)


def run(spec):
    sim = spec.to_simulation()
    sim.run_simulation()
    return sim


def save_figure(filename):
    # si.vis.save_current_figure(filename, target_dir = OUT_DIR, img_format = 'pdf')
    # si.vis.save_current_figure(filename, target_dir = OUT_DIR, img_format = 'pgf')
    si.vis.save_current_figure(filename, target_dir = OUT_DIR, img_format = 'png')


def get_func_name():
    return sys._getframe(1).f_code.co_name


def title_bg():
    sim = ion.CylindricalSliceSpecification(
        'cylindrical_slice_mesh',
        electric_potential = ion.SineWave.from_photon_energy(20 * eV, 1 * atomic_electric_field),
        time_initial = 0, time_final = 200 * asec,
        z_bound = 40 * bohr_radius,
    ).to_simulation()
    sim.run_simulation(progress_bar = True)

    sim.mesh.plot_g(**FIGMAN_KWARGS, **PLOT_KWARGS)  # TODO: fix these at source
    sim.mesh.plot_g2(**FIGMAN_KWARGS, **PLOT_KWARGS)

    for which in ('g', 'g2'):
        with si.vis.FigureManager(
                f'title_bg__{which}',
                tight_layout = False,
                **FIGMAN_KWARGS,
                **PLOT_KWARGS,
        ) as figman:
            fig = figman.fig
            ax = fig.add_axes([0, 0, 1, 1])

            getattr(sim.mesh, f'attach_{which}_to_axis')(ax, plot_limit = 20 * bohr_radius)


def spherical_harmonic_mesh():
    sim = ion.SphericalHarmonicSpecification(
        'spherical_harmonic_mesh',
        electric_potential = ion.SineWave.from_photon_energy(20 * eV, 1 * atomic_electric_field),
        time_initial = 0, time_final = 200 * asec,
        r_points = 500,
        r_bound = 60 * bohr_radius,
        l_bound = 40,
    ).to_simulation()
    sim.run_simulation(progress_bar = True)

    sim.mesh.plot_g_repr(plot_limit = 25 * bohr_radius, **FIGMAN_KWARGS, **PLOT_KWARGS)
    sim.mesh.plot_g(plot_limit = 25 * bohr_radius, **FIGMAN_KWARGS, **PLOT_KWARGS)


if __name__ == '__main__':
    with logman as logger:
        figures = [
            title_bg,
            spherical_harmonic_mesh,
        ]

        for fig in tqdm(figures):
            fig()
