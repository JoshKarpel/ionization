import logging
import os

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


def make_anim(args):
    amp, wavelength = args

    t_bound = 5
    p_bound = 4

    pulse_dummy = ion.SineWave.from_wavelength(wavelength=wavelength, amplitude=amp)
    window = ion.SymmetricExponentialTimeWindow(
        window_time=p_bound * pulse_dummy.period, window_width=0.2 * pulse_dummy.period
    )
    pulse = ion.SineWave.from_wavelength(
        wavelength=wavelength, amplitude=amp, window=window
    )
    times = np.linspace(-t_bound * pulse.period, t_bound * pulse.period, 1e4)

    efield = pulse.get_electric_field_amplitude(times)
    afield = pulse.get_vector_potential_amplitude_numeric_cumulative(times)

    starts = range(0, len(times), 10)

    sliced_times = list(times[start:] for start in starts)
    sliced_alphas = list(
        (proton_charge / electron_mass)
        * integ.cumtrapz(
            y=-integ.cumtrapz(y=efield[start:], x=times[start:], initial=0),
            x=times[start:],
            initial=0,
        )
        for start in starts
    )

    identifier = f"HHG__lambda={uround(wavelength, nm, 0)}nm_amp={uround(amp, atomic_electric_field, 5)}au"

    efield_color = ion.COLOR_ELECTRIC_FIELD
    trajectory_color = si.vis.BLACK

    ### UNITS
    times = times / asec
    efield = efield / atomic_electric_field
    sliced_times = list(time / asec for time in sliced_times)
    sliced_alphas = list(alpha / bohr_radius for alpha in sliced_alphas)

    with si.vis.FigureManager(
        identifier, fig_width=5, fig_dpi_scale=6, target_dir=OUT_DIR, save_on_exit=False
    ) as figman:
        fig = figman.fig

        # EFIELD
        ax_efield = plt.subplot(111)

        ax_efield.plot(times, efield, color=efield_color)
        ax_efield.set_xlabel(r"Time $ t $ (as)")
        ax_efield.set_ylabel(fr"$ {ion.LATEX_EFIELD}(t) $", color=efield_color)
        ax_efield.set_title(r"Free Electron Trajectories")

        ax_efield.tick_params("y", colors=efield_color)
        ax_efield.grid(True, color=efield_color, alpha=0.5)

        max_efield = np.max(np.abs(efield))
        ax_efield.set_ylim(-max_efield * 1.05, max_efield * 1.05)
        ax_efield.set_xlim(times[0], times[-1])

        # TRAJECTORIES
        ax_trajectory = ax_efield.twinx()

        traj_line, = ax_trajectory.plot([], [], color=trajectory_color)
        vert_line = ax_trajectory.axvline(
            np.NaN, color=trajectory_color, alpha=0.75, linestyle=":"
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
            traj_line.set_data(*arg)

            t = arg[0][0]
            vert_line.set_xdata(t)

        si.vis.animate(
            figman,
            update_line,
            update_function_arguments=list(zip(sliced_times, sliced_alphas)),
            artists=[traj_line, vert_line],
            length=30,
        )


if __name__ == "__main__":
    with logman as logger:
        # amplitude = np.array([.1, .5, 1]) * atomic_electric_field
        # wavelength = np.array([800]) * nm
        #
        # si.utils.multi_map(make_anim, list(itertools.product(amplitude, wavelength)), processes = 4)
        make_anim((0.001 * atomic_electric_field, 800 * nm))
        make_anim((0.01 * atomic_electric_field, 800 * nm))
        make_anim((0.1 * atomic_electric_field, 800 * nm))
