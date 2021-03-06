import logging
import os
import time

import numpy as np
import scipy.integrate as integ
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg

import simulacra as si
from simulacra.units import *

import ionization as ion
import ide as ide

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

LOGMAN = si.utils.LogManager("simulacra", "ionization", stdout_level=logging.WARNING)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)


@si.utils.timed
def run_hyd_ide_sim(pulse, tb, dt=1 * asec):
    print("IDE", f"{dt / asec:3f}")
    sim = ide.IntegroDifferentialEquationSpecification(
        "idesim",
        electric_potential=pulse,
        kernel=ide.hydrogen_kernel_LEN,
        integral_prefactor=ide.hydrogen_prefactor_LEN(electron_charge),
        time_initial=-pulse.pulse_width * tb,
        time_final=pulse.pulse_width * tb,
        time_step=dt,
    ).to_sim()

    sim.run()

    return sim


def new_decomposition(pulse, tb, dt=1 * asec):
    total_time = 2 * pulse.pulse_width * tb
    pts = int(total_time / dt)
    print("IME", f"{dt / asec:3f}", pts)
    times = np.linspace(-pulse.pulse_width * tb, pulse.pulse_width * tb, pts)

    kicks = []

    for t_index, t_start in enumerate(times[:-1]):
        t_end = times[t_index + 1]

        eta = integ.quadrature(pulse.get_electric_field_amplitude, t_start, t_end)[0]
        t = (t_start + t_end) / 2

        kicks.append(ide.delta_kick(time=t, amplitude=eta))

    return kicks


def solve_ide_implicit_from_pulse(pulse, tb, dt=1 * asec):
    kicks = new_decomposition(pulse, tb, dt=dt)

    return solve_ide_implicit_from_kicks(kicks)


def solve_ide_implicit_from_pulse_v2(pulse, tb, dt=1 * asec):
    kicks = new_decomposition(pulse, tb, dt=dt)

    return solve_ide_implicit_from_kicks_v2(kicks)


@si.utils.timed
def construct_A(kicks):
    # A = np.zeros((len(kicks), len(kicks)), dtype = np.complex128)
    A = np.tri(len(kicks), dtype=np.complex128)
    # A = sparse.csr_matrix((len(kicks), len(kicks)), dtype = np.complex128)
    print("size of A", si.utils.bytes_to_str(A.nbytes))

    omega = ion.HydrogenBoundState(1, 0).energy / hbar
    prefactor = ide.hydrogen_prefactor_LEN(electron_charge)

    def kernel(td):
        return ide.hydrogen_kernel_LEN(td) * np.exp(1j * omega * td)

    # tds = np.linspace(0, kicks[-1].time + (100 * asec), num = 1000)
    # kinterp = interp.interp1d(tds, kernel(tds), kind = 'cubic', fill_value = 0, bounds_error = False, assume_sorted = True)

    # kernel = lambda td: ide.hydrogen_kernel_LEN(td) * np.exp(1j * omega * td)

    # time_delays = {(n, m): kicks[n].time - kicks[m].time for n in range(len(kicks)) for m in range(n + 1)}
    # kernel_dict = {(n, m): kernel(time_delays[n, m]) for n in range(len(kicks)) for m in range(n + 1)}
    times = np.array([kick.time for kick in kicks])
    amplitudes = np.array([kick.amplitude for kick in kicks])

    # def element(n, m):
    #     rv = prefactor * amplitudes[n] * amplitudes[m] * kernel(time_delays[n, m])
    #     if m == n:
    #         rv -= 1
    #     elif m == n - 1:  # first subdiagonal
    #         rv += 1
    #
    #     return rv
    #
    # with si.utils.BlockTimer() as timer:
    #     for n in range(len(kicks)):
    #         for m in range(n + 1):
    #             A[n, m] = element(n, m)

    for idx in range(len(kicks)):
        A[idx, :] *= amplitudes[idx]  # row idx
        A[:, idx] *= amplitudes[idx]  # column idx
        A[idx, : idx + 1] *= kernel(times[idx] - times[: idx + 1])
        # A[idx, :] *= kinterp(times[idx] - times)

    A *= prefactor
    A[np.arange(len(kicks)), np.arange(len(kicks))] -= 1  # diagonal
    A[
        np.arange(len(kicks) - 1) + 1, np.arange(len(kicks) - 1)
    ] += 1  # first subdiagonal

    A = sparse.csr_matrix(A, dtype=np.complex128)
    print("size of A", si.utils.bytes_to_str(A.data.nbytes))

    return A


@si.utils.timed
def solve_ide_implicit_from_kicks(kicks):
    A = construct_A(kicks)

    b = np.zeros(len(kicks), dtype=np.complex128)
    b[0] = 1

    a = splinalg.spsolve_triangular(
        # a = linalg.solve_triangular(
        A,
        b,
        lower=True,
        overwrite_b=True,
        # check_finite = False,
        overwrite_A=True,
    )

    # a, info = splinalg.bicgstab(A, b, x0 = np.ones_like(b))
    # print(info)

    return a, kicks


@si.utils.timed
def solve_ide_implicit_from_kicks_v2(kicks):
    b = np.empty(len(kicks), dtype=np.complex128)
    b[0] = 1

    omega = ion.HydrogenBoundState(1, 0).energy / hbar
    prefactor = ide.hydrogen_prefactor_LEN(electron_charge)

    def kernel(td):
        return ide.hydrogen_kernel_LEN(td) * np.exp(1j * omega * td)

    times = np.array([kick.time for kick in kicks])
    amplitudes = np.array([kick.amplitude for kick in kicks])
    k0 = kernel(0)

    for n in range(1, len(b)):
        a_n = amplitudes[n]
        sum_pre = prefactor * a_n
        t_n = times[n]
        s = sum_pre * np.sum(amplitudes[:n] * kernel(times[n] - times[:n]) * b[:n])
        b[n] = (b[n - 1] + s) / (1 - (prefactor * k0 * a_n * a_n))

    return b, kicks


def compare_ide_to_matrix(pulse, tb, dts=(1 * asec,)):
    times = np.linspace(-pulse.pulse_width * tb, pulse.pulse_width * tb, 1e5)

    ak_vs_dt = [solve_ide_implicit_from_pulse(pulse, tb, dt=dt) for dt in dts]
    ak_vs_dt_v2 = [solve_ide_implicit_from_pulse_v2(pulse, tb, dt=dt) for dt in dts]
    a_vs_dt = [ak[0] for ak in ak_vs_dt]
    a_vs_dt_v2 = [ak[0] for ak in ak_vs_dt_v2]
    kicks_vs_dt = [ak[1] for ak in ak_vs_dt]
    sims_vs_dt = [run_hyd_ide_sim(pulse, tb, dt=dt) for dt in dts]

    fm = si.vis.xy_plot(
        f"solution_comparison__{int(time.time())}",
        times,
        pulse.get_electric_field_amplitude(times),
        line_kwargs=[{"color": ion.COLOR_ELECTRIC_FIELD}],
        x_unit="asec",
        x_label=r"$ t $",
        y_unit="atomic_electric_field",
        y_label=r"$ \mathcal{E}(t) $",
        title="IDE/IME Solution Comparison",
        save_on_exit=False,
        close_after_exit=False,
        **PLOT_KWARGS,
    )

    fig = fm.fig

    ax_field = fig.axes[0]
    ax_a2 = ax_field.twinx()

    colors = [f"C{n}" for n in range(len(dts))]

    for a, b, kicks, dt, color in zip(a_vs_dt, a_vs_dt_v2, kicks_vs_dt, dts, colors):
        # kt = np.repeat([wavenumber.time for wavenumber in kicks], 2)[1:]
        # kt[0] = times[0]
        # kt[-1] = times[-1]

        ax_a2.plot(
            np.array([k.time for k in kicks]) / asec,
            np.abs(a) ** 2,
            color=color,
            linestyle="--",
            label=rf"IME, $\Delta t = {dt / asec:3f} \, \mathrm{{as}}$",
        )

        ax_a2.plot(
            np.array([k.time for k in kicks]) / asec,
            np.abs(b) ** 2,
            color=color,
            linestyle=":",
            label=rf"IME, $\Delta t = {dt / asec:3f} \, \mathrm{{as}}$",
        )

    for sim, color in zip(sims_vs_dt, colors):
        ax_a2.plot(
            sim.times / asec,
            sim.b2,
            color=color,
            linestyle="-",
            label=rf"IDE, $\Delta t = {sim.time_step/asec:.3f} \, \mathrm{{as}}$",
        )

    ax_a2.set_ylabel(r"$ \left| a(t) \right|^2 $", fontsize=16)

    ax_field.set_xlim(times[0] / asec, times[-1] / asec)
    ax_a2.legend(loc="lower left", fontsize=10)

    fm.save()

    fm.name += "zoom"
    ax_a2.set_ylim(0.9768, 0.9772)

    fm.save()

    fm.cleanup()


def pw_scan(fluence):
    pulse_widths = np.linspace(1, 800, 100) * asec
    ceps = [0, pi / 4, pi / 2]
    tb = 32

    cep_to_pulses = {
        cep: [
            ion.potentials.SincPulse(
                pulse_width=pw,
                fluence=fluence,
                phase=cep,
                window=ion.potentials.LogisticWindow(
                    window_time=pw * (tb - 2), window_width=0.2 * pw
                ),
            )
            for pw in pulse_widths
        ]
        for cep in ceps
    }
    cep_to_solns = {
        cep: [solve_ide_implicit_from_pulse(pulse, tb) for pulse in pulses]
        for cep, pulses in cep_to_pulses.items()
    }

    si.vis.xy_plot(
        f"pulse_width_scan__flu={fluence / Jcm2:3f}jcm2",
        pulse_widths,
        *[
            [np.abs(soln[0][-1]) ** 2 for soln in solns]
            for cep, solns in cep_to_solns.items()
        ],
        line_labels=[rf"$\varphi = {cep / pi:3f}\pi$" for cep in ceps],
        x_label=r"$ \tau $",
        x_unit="asec",
        y_label=r"$ \left| a(t_f) \right|^2 $",
        title=rf"Delta-Kick Model Pulse Width Scan, $H = {fluence / Jcm2:3f} \, \mathrm{{J/cm^2}}$",
        x_lower_limit=0,
        **PLOT_KWARGS,
    )


def kick_delay_scan(eta=0.1 * atomic_time * atomic_electric_field):
    delays = np.linspace(1, 1000, 500) * asec

    kickss = [
        (ide.delta_kick(-delay / 2, eta), ide.delta_kick(delay / 2, -eta))
        for delay in delays
    ]
    solns = [solve_ide_implicit_from_kicks(kicks) for kicks in kickss]

    si.vis.xy_plot(
        f"kick_delay_scan__eta={eta / atomic_time * atomic_electric_field:.3f}au",
        delays,
        [np.abs(soln[0][-1]) ** 2 for soln in solns],
        x_label=r"Kick Delay $ \Delta $",
        x_unit="asec",
        y_label=r"$ \left| a(t_f) \right|^2 $",
        title=f"Delta-Kick Model Kick Delay Scan, $\eta = {eta / atomic_time * atomic_electric_field:.3f} \, \mathrm{{a.u.}}$",
        vlines=[93 * asec, 150 * asec],
        hlines=[np.abs(solns[-1][0][-1]) ** 2],
        x_lower_limit=0,
        **PLOT_KWARGS,
    )


if __name__ == "__main__":
    with LOGMAN as logger:
        ide._hydrogen_kernel_LEN_factory()  # call early to remove from timing
        pulse = ion.potentials.GaussianPulse.from_number_of_cycles(
            pulse_width=50 * asec, fluence=0.1 * Jcm2, phase=pi / 2, number_of_cycles=2
        )
        # pulse = ion.potentials.SincPulse(pulse_width = 50 * asec, fluence = .1 * Jcm2, phase = pi / 2)

        dts = np.array([1]) * asec
        # dts = np.array([80]) * asec
        compare_ide_to_matrix(pulse, tb=4, dts=dts)

        # etas = np.array([.01, .05, .1, .2, .3, .4, .5,]) * atomic_time * atomic_electric_field
        # etas = np.geomspace(.01, 2, 10) * atomic_time * atomic_electric_field
        # si.utils.multi_map(kick_delay_scan, etas, processes = 3)

        # fluences = np.array([.01, .1, 1, 5, 10]) * Jcm2
        # fluences = np.geomspace(.01, 20, 10) * Jcm2
        # si.utils.multi_map(pw_scan, fluences, processes = 3)
