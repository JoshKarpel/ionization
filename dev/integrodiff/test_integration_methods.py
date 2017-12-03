import logging
import os
import time

import numpy as np
import scipy.integrate as integ
import scipy.interpolate as interp

import simulacra as si
import simulacra.units as u

import ionization as ion
import ionization.ide as ide

import matplotlib.pyplot as plt

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

LOGMAN = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.DEBUG)

if __name__ == '__main__':
    with LOGMAN as logger:
        integrator = ide.InterpolatedIntegrator(
            integrator = integ.quad,
        )

        x = np.linspace(0, 1, 10000000)


        def y(x):
            time.sleep(.1)
            return (x ** 2) * (np.sin(x) ** 2)


        with si.utils.BlockTimer() as timer:
            quad = integ.quad(y, x[0], x[-1])[0]
        print('quad', quad, timer.proc_time_elapsed)

        with si.utils.BlockTimer() as timer:
            quadrature = integ.quadrature(y, x[0], x[-1])[0]
        print('quadrature', quadrature, timer.proc_time_elapsed)

        with si.utils.BlockTimer() as timer:
            trapz = integ.trapz(y = y(x), x = x)
        print('trapz', trapz, timer.proc_time_elapsed)

        with si.utils.BlockTimer() as timer:
            simps = integ.simps(y = y(x), x = x)
        print('simps', simps, timer.proc_time_elapsed)

        with si.utils.BlockTimer() as timer:
            interpolated = integrator(y(x), x)
        print('interpolated', interpolated, timer.proc_time_elapsed)
