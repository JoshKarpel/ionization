import logging
import os

import numpy as np

import simulacra as si
import simulacra.units as u

import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

LOGMAN = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.INFO)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)


def run(spec):
    with LOGMAN as logger:
        sim = spec.to_sim()

        sim.run()

        return sim


if __name__ == '__main__':
    with LOGMAN as logger:
        spec = ion.mesh.CylindricalSliceSpecification(
            f'cyl_slice',
            operators = ion.mesh.CylindricalSliceLengthGaugeOperators(),
            evolution_method = ion.mesh.AlternatingDirectionImplicitCrankNicolson(),
            evolution_gauge = ion.Gauge.LENGTH,
            rho_bound = 20 * u.bohr_radius,
            rho_points = 20 * 10,
            z_bound = 20 * u.bohr_radius,
            z_points = 20 * 10 * 2,
            time_initial = 0,
            time_final = 1 * u.asec
        )

        sim = spec.to_sim()

        print(sim.info())
        # print(sim.mesh.operators)
        # print(sim.mesh.operators.kinetic_energy(sim.mesh))
        # print(sim.mesh.operators.total_hamiltonian(sim.mesh))
        for oper in sim.spec.evolution_method.get_evolution_operators(sim.mesh, 1 * u.asec):
            print(oper)
