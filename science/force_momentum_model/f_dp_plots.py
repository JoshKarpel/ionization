import logging
import os
import warnings

import numpy as np
import scipy.integrate as integ
import scipy.interpolate as interp

import simulacra as si
from simulacra.units import *

import ionization as ion
import ionization.integrodiff as ide

import matplotlib.pyplot as plt

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

LOGMAN = si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.DEBUG)

PLOT_KWARGS = dict(
    target_dir = OUT_DIR,
    img_format = 'png',
    fig_dpi_scale = 6,
)

PLOT_POINTS = 300

PULSE_TO_T_BOUND = {
    ion.GaussianPulse: 10,
    ion.SincPulse: 35,
}

PULSE_TO_P_BOUND = {
    ion.GaussianPulse: 3.5,
    ion.SincPulse: 30,
}


def f_dp_plot(pulse):
    t_bound = PULSE_TO_T_BOUND[pulse.__class__]
    p_bound = PULSE_TO_P_BOUND[pulse.__class__]

    times = np.linspace(-t_bound * pulse.pulse_width, t_bound * pulse.pulse_width, PLOT_POINTS * t_bound / p_bound)

    force = lambda t: electron_charge * pulse.get_electric_field_amplitude(t)

    momentum_array = electron_charge * pulse.get_electric_field_integral_numeric_cumulative(times)
    momentum = interp.interp1d(times, momentum_array, fill_value = np.NaN, bounds_error = False, kind = 'cubic')

    deltas = np.linspace(0, 1000, PLOT_POINTS) * asec

    time_mesh, delta_mesh = np.meshgrid(times, deltas, indexing = 'ij')

    f_dp = force(time_mesh) * (momentum(time_mesh) - momentum(time_mesh - delta_mesh))

    si.vis.xyz_plot(
        'f_dp',
        time_mesh,
        delta_mesh,
        f_dp,
        x_unit = 'asec',
        x_label = r'$ t $',
        y_unit = 'asec',
        y_label = r'$ \delta $',
        z_unit = atomic_force * atomic_momentum,
        z_label = r'$ F(t) \times \left[ p(t) - p(t - \delta) \right] $',
        colormap = plt.get_cmap('RdBu_r'),
        x_lower_limit = -p_bound * pulse.pulse_width,
        x_upper_limit = p_bound * pulse.pulse_width,
        **PLOT_KWARGS
    )

    si.vis.xyz_plot(
        'f_dp__ABS',
        time_mesh,
        delta_mesh,
        np.abs(f_dp),
        x_unit = 'asec',
        x_label = r'$ t $',
        y_unit = 'asec',
        y_label = r'$ \delta $',
        z_unit = atomic_force * atomic_momentum,
        z_label = r'$ F(t) \times \left[ p(t) - p(t - \delta) \right] $',
        x_lower_limit = -p_bound * pulse.pulse_width,
        x_upper_limit = p_bound * pulse.pulse_width,
        **PLOT_KWARGS
    )

    with si.utils.BlockTimer() as timer:
        f_dp_K = f_dp * determine_K(pulse, time_mesh, delta_mesh)
    print('time to do the K integrals', timer)
    normed = f_dp_K / np.nanmean(np.abs(f_dp_K))

    si.vis.xyz_plot(
        'f_dp_k',
        time_mesh,
        delta_mesh,
        normed,
        x_unit = 'asec',
        x_label = r'$ t $',
        y_unit = 'asec',
        y_label = r'$ \delta $',
        z_label = r'$ K(t, \delta) \times F(t) \times \left[ p(t) - p(t - \delta) \right] $ (norm.)',
        colormap = plt.get_cmap('richardson'),
        richardson_equator_magnitude = 5,
        x_lower_limit = -p_bound * pulse.pulse_width,
        x_upper_limit = p_bound * pulse.pulse_width,
        **PLOT_KWARGS
    )

    si.vis.xyz_plot(
        'f_dp_k__ABS',
        time_mesh,
        delta_mesh,
        np.abs(f_dp_K),
        x_unit = 'asec',
        x_label = r'$ t $',
        y_unit = 'asec',
        y_label = r'$ \delta $',
        z_unit = atomic_force * atomic_momentum,
        z_label = r'$ K(t, \delta) \times F(t) \times \left[ p(t) - p(t - \delta) \right] $',
        x_lower_limit = -p_bound * pulse.pulse_width,
        x_upper_limit = p_bound * pulse.pulse_width,
        **PLOT_KWARGS
    )


def determine_K(pulse, time, delta):
    num = si.math.complex_quadrature(lambda tp: pulse.get_electric_field_amplitude(tp) * ide.hydrogen_kernel_LEN(time - tp), time - delta, time)[0]
    den = integ.quadrature(pulse.get_electric_field_amplitude, time - delta, time)[0]

    if den == 0:
        return np.NaN

    return num / den


determine_K = np.vectorize(determine_K, otypes = [np.complex128])

if __name__ == '__main__':
    with LOGMAN as logger:
        pulse = ion.GaussianPulse.from_number_of_cycles(pulse_width = 200 * asec, fluence = 1 * Jcm2, number_of_cycles = 3)

        f_dp_plot(pulse)
