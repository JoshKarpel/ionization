import numpy as np
import matplotlib.pyplot as plt

import simulacra as si
from simulacra.units import *
import ionization as ion
import ionization.ide as ide
from idesolver import IDESolver



def make_abs_plot(solver):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(solver.x / asec, np.abs(solver.y) ** 2, color = 'black', linewidth = 3)

    ax.grid(True)
    ax.set_xlim(solver.x[0] / asec, solver.x[-1] / asec)
    ax.set_ylim(0, 1.1)
    plt.show()


def make_error_plot(solver, exact):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    error = np.abs(solver.y - exact)

    ax.plot(solver.x, error)
    ax.set_yscale('log')
    ax.grid(True)

    plt.show()


def iterative():
    pw = 200 * asec
    omega = ion.HydrogenBoundState(1, 0).energy / hbar
    efield = ion.GaussianPulse.from_number_of_cycles(pulse_width = pw, fluence = 1 * Jcm2, phase = 0).get_electric_field_amplitude
    kern = ide.hydrogen_kernel_LEN

    t = np.linspace(-pw * 3, pw * 3, 200)

    solver = IDESolver(
        y_initial = 1,
        x = t,
        c = lambda x, y: -1j * omega * y,
        d = lambda x: -((electron_charge / hbar) ** 2) * efield(x),
        k = lambda x, s: efield(s) * kern(x - s),
        lower_bound = lambda x: t[0],
        upper_bound = lambda x: x,
        f = lambda y: y,
        global_error_tolerance = 1e-6,
    )
    solver.solve()

    return solver


if __name__ == '__main__':
    solver = iterative()

    print(solver.wall_time_elapsed)

    # make_comparison_plot(solver, exact)
    # make_error_plot(solver, exact)
    make_abs_plot(solver)
