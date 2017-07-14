import os

import numpy as np
import simulacra as si
from simulacra.units import *

import ionization as ion


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 5,
)

if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization') as logger:
        x = np.linspace(-20 * bohr_radius, 20 * bohr_radius, 1000)

        gaussian_well = ion.GaussianPotential(potential_extrema = -10 * eV, width = 5 * bohr_radius)

        variational_ground_state = ion.GaussianWellState.from_potential(gaussian_well, electron_mass)
        print('width', variational_ground_state.width / bohr_radius)
        print('energy', variational_ground_state.energy / eV)

        si.vis.xy_plot(
            'potential',
            x,
            gaussian_well(distance = x),
            x_label = r'$x$', x_unit = 'bohr_radius',
            y_label = r'$V(x)$', y_unit = 'eV',
            **PLOT_KWARGS,
        )

        sim = ion.LineSpecification(
            'gaussian_well',
            x_bound = 100 * bohr_radius,
            x_points = 2 ** 12,
            test_mass = electron_mass,
            internal_potential = gaussian_well,
            initial_state = variational_ground_state,
            use_numeric_eigenstates = True,
            analytic_eigenstate_type = ion.GaussianWellState
        ).to_simulation()

        si.vis.xy_plot(
            'ground_state_comparison',
            sim.mesh.x_mesh,
            variational_ground_state(sim.mesh.x_mesh),
            np.abs(sim.spec.test_states[0](sim.mesh.x_mesh)),
            line_labels = ['Variational', 'Numeric'],
            x_label = r'$x$', x_unit = 'bohr_radius',
            y_label = r'$\psi_0(x)$',
            **PLOT_KWARGS
        )

        for state in sim.spec.test_states:
            si.vis.xy_plot(
                si.utils.strip_illegal_characters(f'numeric__{"n" if state.bound else "E"}={state.n if state.bound else uround(state.energy, eV)}'),
                sim.mesh.x_mesh,
                np.abs(state(sim.mesh.x_mesh)) ** 2,
                x_label = r'$x$', x_unit = 'bohr_radius',
                y_label = fr'$\left| \psi_{{{"n" if state.bound else "E"}={state.n if state.bound else uround(state.energy, eV)}}}(x) \right|^2$',
                **PLOT_KWARGS
            )

        for state in sim.spec.test_states:
            print(f'n={state}: {uround(state.energy, eV)}')
