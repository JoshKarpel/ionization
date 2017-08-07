import os
import itertools

import numpy as np

import simulacra as si
from simulacra.units import *
import ionization as ion

import matplotlib as mpl

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

UNITS = ['bohr_radius']
BOUNDS = np.array([20]) * bohr_radius
SHADINGS = ['gouraud', 'flat']
CMAPS = [mpl.cm.get_cmap('inferno')]
AXES = [True]

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)


def make_plots(sim, postfix = ''):
    for unit, bound, shading, show_axes in itertools.product(UNITS, BOUNDS, SHADINGS, AXES):
        name_postfix = postfix + f'_unit={unit}_R={uround(bound, unit)}br_shading={shading}_axes={show_axes}'

        for cmap in CMAPS:
            sim.mesh.plot_g2(
                name_postfix = name_postfix + f'_cmap={cmap.name}',
                colormap = cmap,
                distance_unit = unit,
                plot_limit = bound,
                shading = shading,
                show_axes = show_axes,
                **PLOT_KWARGS
            )

        sim.mesh.plot_g(
            name_postfix = name_postfix + '_cmap=richardson',
            distance_unit = unit,
            plot_limit = bound,
            shading = shading,
            show_axes = show_axes,
            **PLOT_KWARGS
        )


if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization') as logger:
        max_n = 2
        angular_points = 50

        states = list(ion.HydrogenBoundState(n, l) for n in range(max_n + 1) for l in range(n))

        sim = ion.SphericalHarmonicSpecification(
            f'sph_harms',
            r_bound = np.max(BOUNDS),
            l_bound = angular_points,
            use_numeric_eigenstates = True,
            numeric_eigenstate_max_energy = 20 * eV,
            numeric_eigenstate_max_angular_momentum = max(s.l for s in states) + 1,
        ).to_simulation()

        for state in states:
            logger.info(f'Generating plots for {state} on {sim.mesh.__class__.__name__}')
            sim.mesh.g = sim.mesh.get_g_for_state(state)
            sim.time_index += 1  # hack to get around the watcher on g spatial reconstruction in spherical harmonic mesh
            make_plots(sim, postfix = f'__n={state.n}_l={state.l}')
