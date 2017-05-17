import os
import itertools

import numpy as np

import simulacra as si
from simulacra.units import *
import ionization as ion

import matplotlib as mpl


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

# units = ('bohr_radius', 'nm')
# bounds = np.array([10, 20, 50]) * bohr_radius
# cmaps = (mpl.cm.get_cmap('inferno'), mpl.cm.get_cmap('viridis'))

# units = ('bohr_radius',)
# bounds = np.array([10, 20]) * bohr_radius
# cmaps = (mpl.cm.get_cmap('inferno'),)

units = ('bohr_radius',)
bounds = np.array([10, 20, 50, 100]) * bohr_radius
shadings = ('flat', 'gouraud')
cmaps = (mpl.cm.get_cmap('inferno'),)


def make_plots(spec):
    sim = ion.ElectricFieldSimulation(spec)

    for unit, bound, shading in itertools.product(units, bounds, shadings):
        for cmap in cmaps:
            postfix = f'__{unit}__{uround(bound, unit)}__{cmap.name}__{shading}'
            sim.mesh.plot_g2(name_postfix = postfix, target_dir = OUT_DIR, colormap = cmap, distance_unit = unit, plot_limit = bound, shading = shading)
            # sim.mesh.plot_psi2(name_postfix = postfix, target_dir = OUT_DIR, colormap = cmap, distance_unit = unit, plot_limit = bound)
        sim.mesh.plot_g(name_postfix = f'__{unit}__{uround(bound, unit)}__richardson__{shading}', target_dir = OUT_DIR, distance_unit = unit, plot_limit = bound, shading = shading)


if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization') as logger:
        n = 5
        angular_points = 200

        states = (ion.HydrogenBoundState(n, l) for n in range(n + 1) for l in range(n))

        specs = []

        # line_potential = ion.FiniteSquareWell(potential_depth = 3 * eV, width = 1 * nm)
        # for initial_state in ion.FiniteSquareWellState.all_states_of_well_from_parameters(3 * eV, 1 * nm, electron_mass):
        #     specs.append(ion.LineSpecification('line_mesh__{}'.format(initial_state.n),
        #                                        initial_state = initial_state,
        #                                        x_bound = 30 * nm))

        for state in states:
            specs.append(ion.CylindricalSliceSpecification(f'cyl_slice__{state.n}_{state.l}',
                                                           initial_state = state,
                                                           z_bound = np.max(bounds), rho_bound = np.max(bounds)))

            specs.append(ion.SphericalSliceSpecification(f'sph_slice__{state.n}_{state.l}',
                                                         initial_state = state,
                                                         r_bound = np.max(bounds), theta_points = angular_points))

            specs.append(ion.SphericalHarmonicSpecification(f'sph_harms__{state.n}_{state.l}',
                                                            initial_state = state,
                                                            r_bound = np.max(bounds), l_bound = angular_points))

        si.utils.multi_map(make_plots, specs, processes = 5)
