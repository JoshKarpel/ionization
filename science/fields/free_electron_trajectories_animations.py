import logging
import os
import itertools

import numpy as np
import scipy.integrate as integ

import simulacra as si
from simulacra.units import *

import ionization as ion

import matplotlib.pyplot as plt

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

SIM_LIB = os.path.join(OUT_DIR, "SIMLIB")

logman = si.utils.LogManager("simulacra", "ionization", stdout_level=logging.INFO)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)

T_BOUND_MAP = {ion.potentials.SincPulse: 12, ion.potentials.GaussianPulse: 6}
P_BOUND_MAP = {ion.potentials.SincPulse: 10, ion.potentials.GaussianPulse: 5}


def make_anim(args):
    pulse_type, pw, flu, cep = args

    t_bound = T_BOUND_MAP[pulse_type]
    p_bound = P_BOUND_MAP[pulse_type]

    times = np.linspace(-t_bound * pw, t_bound * pw, 1e4)

    window = ion.potentials.LogisticWindow(
        window_time=p_bound * pw, window_width=0.2 * pw
    )
    pulse = pulse_type(pulse_width=pw, fluence=flu, phase=cep, window=window)
    corrected_pulse = ion.DC_correct_electric_potential(pulse, times)

    efield = corrected_pulse.get_electric_field_amplitude(times)
    afield = corrected_pulse.get_vector_potential_amplitude_numeric_cumulative(times)

    starts = range(0, len(times), 10)

    sliced_times = list(times[start:] for start in starts)
    sliced_vector_potentials = list(
        -proton_charge * integ.cumtrapz(y=efield[start:], x=times[start:], initial=0)
        for start in starts
    )
    sliced_alphas = list(
        (proton_charge / electron_mass)
        * integ.cumtrapz(
            y=-integ.cumtrapz(y=efield[start:], x=times[start:], initial=0),
            x=times[start:],
            initial=0,
        )
        for start in starts
    )

    identifier = f"{pulse_type.__name__}__pw={pw / asec:0f}as_flu={flu / Jcm2:2f}jcm2_cep={cep / pi:2f}pi"

    efield_color = ion.COLOR_ELECTRIC_FIELD
    afield_color = ion.COLOR_VECTOR_POTENTIAL
    trajectory_color = si.vis.BLACK

    ### UNITS
    times = times / asec
    efield = efield / atomic_electric_field
    sliced_times = list(time / asec for time in sliced_times)
    sliced_vector_potentials = list(
        vp / atomic_momentum for vp in sliced_vector_potentials
    )
    sliced_alphas = list(alpha / bohr_radius for alpha in sliced_alphas)

    max_vp = max(np.max(np.abs(vp)) for vp in sliced_vector_potentials)

    with si.vis.FigureManager(
        identifier, fig_width=5, fig_dpi_scale=6, target_dir=OUT_DIR, save_on_exit=False
    ) as figman:
        fig = figman.fig

        # EFIELD
        ax_efield = plt.subplot(111)

        ax_efield.plot(
            times, efield, color=efield_color, label=fr"$ {ion.vis.LATEX_EFIELD}(t) $"
        )
        vect_line, = ax_efield.plot(
            [],
            [],
            color=afield_color,
            animated=True,
            label=fr"$ e \, {ion.LATEX_AFIELD}(t) $",
        )

        ax_efield.set_xlabel(r"Time $ t $ (as)")
        ax_efield.set_ylabel(
            fr"$ {ion.vis.LATEX_EFIELD}(t), \; e \, {ion.LATEX_AFIELD}(t) $ (a.u.)",
            color=efield_color,
        )
        ax_efield.set_title(r"Free Electron Trajectories")

        ax_efield.tick_params("y", colors=efield_color)
        ax_efield.grid(True, color=efield_color, alpha=0.5)

        ax_efield.set_ylim(-max_vp * 1.05, max_vp * 1.05)
        ax_efield.set_xlim(times[0], times[-1])

        plt.legend(loc="upper left")

        # TRAJECTORIES
        ax_trajectory = ax_efield.twinx()

        traj_line, = ax_trajectory.plot([], [], color=trajectory_color, animated=True)
        vert_line = ax_trajectory.axvline(
            np.NaN, color=trajectory_color, alpha=0.5, animated=True
        )
        ax_trajectory.set_ylabel(fr"$ \alpha(t, t') $", color=trajectory_color)

        ax_trajectory.tick_params("y", colors=trajectory_color)
        ax_trajectory.grid(True, color=trajectory_color, alpha=0.5)

        max_alpha = max(np.max(np.abs(alpha)) for alpha in sliced_alphas)
        ax_trajectory.set_ylim(-max_alpha * 1.05, max_alpha * 1.05)
        ax_trajectory.set_xlim(times[0], times[-1])

        fig.tight_layout()

        # ANIMATION

        def update_line(arg):
            t, vp, alpha = arg
            vect_line.set_data(t, vp)
            traj_line.set_data(t, alpha)
            vert_line.set_xdata(t[0])

        si.vis.animate(
            figman,
            update_line,
            update_function_arguments=list(
                zip(sliced_times, sliced_vector_potentials, sliced_alphas)
            ),
            artists=[vect_line, traj_line, vert_line],
            length=30,
        )


if __name__ == "__main__":
    with logman as logger:
        pulse_widths = np.array([200]) * asec
        fluences = np.array([1]) * Jcm2
        phases = [0, pi / 4, pi / 2]
        pulse_types = [ion.potentials.SincPulse, ion.potentials.GaussianPulse]

        si.utils.multi_map(
            make_anim,
            list(itertools.product(pulse_types, pulse_widths, fluences, phases)),
            processes=2,
        )
