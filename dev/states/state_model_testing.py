import os

import numpy as np

import simulacra as si
import simulacra.units as u
import ionization as ion

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization') as logger:
        # for n in range(1, 4):
        #     for l in range(n):
        #         for m in range(-l, l + 1):
        #             print(n, l, m)
        #
        #             s = ion.states.HydrogenBoundState(n = n, l = l, m = m)
        #             print(str(s))
        #             print(repr(s))
        #             print()
        #
        # s = ion.states.HydrogenBoundState(amplitude = .00124j)
        # print(s.ket)
        # print(s.bra)
        # print()
        #
        # print(s.normalized())
        #
        # s += ion.states.HydrogenBoundState(2, 1, 0, amplitude = -.5)
        #
        # print(s.states)
        # print(s)
        # print(s.norm)
        # print()
        #
        # s += ion.states.HydrogenBoundState(5, 2, 1, amplitude = .5j)
        #
        # print(s.states)
        # print(s)
        # print(s.norm)
        # print()
        #
        # n = s.normalized()
        # print(n)
        # print(n.norm)
        # print(repr(n))

        # a = ion.states.HydrogenBoundState()
        # b = ion.states.HydrogenBoundState(amplitude = 1j)
        #
        # c = (a + b).normalized()
        # print(c)
        # print(c.info())
        # print()
        #
        # d = ion.states.HydrogenBoundState(n = 2)
        #
        # e = (c + d).normalized()
        # print(e)
        # print(e.info())
        # print()
        #
        # f = ion.states.HydrogenCoulombState()
        #
        # g = (e + f).normalized()
        # print(g)
        # print(g.info())
        # print(g.tex)
        # print()
        #
        # num = ion.states.NumericSphericalHarmonicState(
        #     radial_wavefunction = 0,
        #     l = 1,
        #     m = 0,
        #     energy = -13.6 * u.eV,
        #     corresponding_analytic_state = ion.states.HydrogenBoundState(),
        #     binding = ion.states.Binding.BOUND)
        # print(num)
        # print(num.corresponding_analytic_state)
        # print(num.tex)
        # print(num.tex_ket)
        # print(num.tex_bra)
        # print(num.info())

        STATES = [
            ion.states.FreeSphericalWave(),
            ion.states.HydrogenBoundState(),
            ion.states.HydrogenCoulombState(),
            ion.states.NumericSphericalHarmonicState(
                g = 0,
                l = 1,
                m = 0,
                energy = 1 * u.eV,
                corresponding_analytic_state = ion.states.HydrogenCoulombState(),
                binding = ion.states.Binding.FREE,
            ),
            ion.states.OneDPlaneWave(),
            ion.states.QHOState.from_omega_and_mass(omega = u.atomic_angular_frequency),
            ion.states.FiniteSquareWellState(1 * u.eV, 1 * u.nm, u.electron_mass),
            ion.states.GaussianWellState(1 * u.eV, 1 * u.nm, u.electron_mass),
            ion.states.OneDSoftCoulombState(),
            ion.states.NumericOneDState(
                g = 0,
                energy = 1 * u.eV,
                corresponding_analytic_state = ion.states.OneDPlaneWave(),
                binding = ion.states.Binding.FREE,
            ),
            ion.states.ThreeDPlaneWave(0, 0, u.twopi / u.nm),
        ]

        for state in STATES:
            print(state.__class__.__name__)
            print('str:', str(state))
            print('repr:', repr(state))
            print('ket:', state.ket)
            print('bra:', state.bra)
            print('tex:', state.tex)
            print('tex_ket:', state.tex_ket)
            print('tex_bra:', state.tex_bra)
            print(state.info())
            print()
