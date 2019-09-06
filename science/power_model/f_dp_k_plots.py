import logging
import os
import functools

import numpy as np
import scipy.integrate as integ
import scipy.interpolate as interp

import simulacra as si
import simulacra.units as u

import ionization as ion
import ide as ide

import matplotlib.pyplot as plt

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

LOGMAN = si.utils.LogManager("simulacra", "ionization", stdout_level=logging.DEBUG)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)

PLOT_POINTS = 300

PULSE_TO_T_BOUND = {ion.potentials.GaussianPulse: 10, ion.potentials.SincPulse: 35}

PULSE_TO_P_BOUND = {ion.potentials.GaussianPulse: 3, ion.potentials.SincPulse: 30}


def pulse_to_filename(pulse):
    attrs = [
        ("pulse_width", lambda x: f"{x / u.asec:.3f}"),
        ("fluence", lambda x: f"{x / u.Jcm2:.3f}"),
        ("phase", lambda x: f"{x / u.pi:.3f}"),
    ]

    if isinstance(pulse, ion.potentials.GaussianPulse):
        attrs.append("number_of_cycles")

    if isinstance(pulse, ion.potentials.SincPulse):
        attrs.append(("omega_min", lambda x: f"{x / u.twopi * u.THz:.3f}"))

    return si.utils.obj_to_filename(pulse, attrs)


def make_f_dp_k_plots(pulse):
    t_bound = PULSE_TO_T_BOUND[pulse.__class__]
    p_bound = PULSE_TO_P_BOUND[pulse.__class__]

    prefix = pulse_to_filename(pulse) + "___"

    times = np.linspace(
        -t_bound * pulse.pulse_width,
        t_bound * pulse.pulse_width,
        PLOT_POINTS * t_bound / p_bound,
    )
    # pulse = ion.DC_correct_electric_potential(pulse, times)

    si.vis.xy_plot(
        prefix + "fields",
        times,
        pulse.get_electric_field_amplitude(times) / u.atomic_electric_field,
        pulse.get_vector_potential_amplitude_numeric_cumulative(times)
        * u.proton_charge
        / u.atomic_momentum,
        line_labels=[r"$\mathcal{E}(t)$", r"$q \, \mathcal{A}(t)$"],
        x_unit="asec",
        x_label=r"$ t $",
        y_label="Field Amplitudes (a.u.)",
        x_lower_limit=-p_bound * pulse.pulse_width,
        x_upper_limit=p_bound * pulse.pulse_width,
        **PLOT_KWARGS,
    )

    force = lambda t: u.electron_charge * pulse.get_electric_field_amplitude(t)

    m_times = np.linspace(
        -t_bound * pulse.pulse_width, t_bound * pulse.pulse_width, 1000
    )
    momentum_array = (
        u.electron_charge
        * pulse.get_electric_field_integral_numeric_cumulative(m_times)
    )
    momentum = interp.interp1d(
        m_times, momentum_array, fill_value=np.NaN, bounds_error=False, kind="cubic"
    )

    deltas = np.linspace(0, 3 * pulse.pulse_width, PLOT_POINTS)

    curr_time_mesh, delta_mesh = np.meshgrid(times, deltas, indexing="ij")

    f_dp = force(curr_time_mesh) * (
        momentum(curr_time_mesh) - momentum(curr_time_mesh - deltas)
    )

    si.vis.xyz_plot(
        prefix + "f_dp",
        curr_time_mesh,
        delta_mesh,
        f_dp,
        x_unit="asec",
        x_label=r"$ t $",
        y_unit="asec",
        y_label=r"$ \delta $",
        z_unit=u.atomic_force * u.atomic_momentum,
        z_label=r"$ \mathcal{F}(t) \times \Delta p(t, \delta) $",
        colormap=plt.get_cmap("RdBu_r"),
        x_lower_limit=-p_bound * pulse.pulse_width,
        x_upper_limit=p_bound * pulse.pulse_width,
        title=r"Force $\times$ Momentum",
        **PLOT_KWARGS,
    )

    si.vis.xyz_plot(
        prefix + "f_dp__ABS",
        curr_time_mesh,
        delta_mesh,
        np.abs(f_dp),
        x_unit="asec",
        x_label=r"$ t $",
        y_unit="asec",
        y_label=r"$ \delta $",
        z_unit=u.atomic_force * u.atomic_momentum,
        z_label=r"$ \left| \mathcal{F}(t) \times \Delta p(t, \delta) \right| $",
        x_lower_limit=-p_bound * pulse.pulse_width,
        x_upper_limit=p_bound * pulse.pulse_width,
        title=r"Force $\times$ Momentum (Abs.)",
        **PLOT_KWARGS,
    )

    with si.utils.BlockTimer() as timer:
        f_dp_K = f_dp * determine_K_from_delta(pulse, curr_time_mesh, delta_mesh)
    print("time to do the K integrals", timer)
    normed = f_dp_K / np.nanmean(np.abs(f_dp_K))

    si.vis.xyz_plot(
        prefix + "f_dp_k",
        curr_time_mesh,
        delta_mesh,
        normed,
        x_unit="asec",
        x_label=r"$ t $",
        y_unit="asec",
        y_label=r"$ \delta $",
        colormap=plt.get_cmap("richardson"),
        richardson_equator_magnitude=5,
        x_lower_limit=-p_bound * pulse.pulse_width,
        x_upper_limit=p_bound * pulse.pulse_width,
        title=r"$ \mathcal{K}(t, \, \delta) \times \mathcal{F}(t) \times \Delta p(t, \delta) $",
        **PLOT_KWARGS,
    )

    si.vis.xyz_plot(
        prefix + "f_dp_k__ABS",
        curr_time_mesh,
        delta_mesh,
        np.abs(f_dp_K) / np.nanmean(np.abs(f_dp_K)),
        x_unit="asec",
        x_label=r"$ t $",
        y_unit="asec",
        y_label=r"$ \delta $",
        x_lower_limit=-p_bound * pulse.pulse_width,
        x_upper_limit=p_bound * pulse.pulse_width,
        z_upper_limit=10,
        title=r"$ \left| \mathcal{K}(t, \, \delta) \times \mathcal{F}(t) \times \Delta p(t, \delta) \right| $ (norm. by avg.)",
        **PLOT_KWARGS,
    )


def determine_K_from_delta(pulse, time, delta):
    num = si.math.complex_quadrature(
        lambda tp: pulse.get_electric_field_amplitude(tp)
        * ide.hydrogen_kernel_LEN(time - tp),
        time - delta,
        time,
    )[0]
    den = integ.quadrature(pulse.get_electric_field_amplitude, time - delta, time)[0]

    if den == 0:
        return np.NaN

    return num / den


determine_K_from_delta = np.vectorize(determine_K_from_delta, otypes=[np.complex128])


def make_f_dp_k_plots_by_earlier_time(pulse):
    t_bound = PULSE_TO_T_BOUND[pulse.__class__]
    p_bound = PULSE_TO_P_BOUND[pulse.__class__]

    prefix = pulse_to_filename(pulse) + "___"

    current_times = np.linspace(
        -p_bound * pulse.pulse_width, p_bound * pulse.pulse_width, PLOT_POINTS
    )
    # pulse = ion.DC_correct_electric_potential(pulse, times)

    si.vis.xy_plot(
        prefix + "fields",
        current_times,
        pulse.get_electric_field_amplitude(current_times) / u.atomic_electric_field,
        pulse.get_vector_potential_amplitude_numeric_cumulative(current_times)
        * u.proton_charge
        / u.atomic_momentum,
        line_labels=[r"$\mathcal{E}(t)$", r"$q \, \mathcal{A}(t)$"],
        x_unit="asec",
        x_label=r"$ t $",
        y_label="Field Amplitudes (a.u.)",
        x_lower_limit=-p_bound * pulse.pulse_width,
        x_upper_limit=p_bound * pulse.pulse_width,
        **PLOT_KWARGS,
    )

    force = lambda t: u.electron_charge * pulse.get_electric_field_amplitude(t)

    m_times = np.linspace(
        -t_bound * pulse.pulse_width, t_bound * pulse.pulse_width, 1000
    )
    momentum_array = (
        u.electron_charge
        * pulse.get_electric_field_integral_numeric_cumulative(m_times)
    )
    momentum = interp.interp1d(
        m_times, momentum_array, fill_value=np.NaN, bounds_error=False, kind="cubic"
    )

    current_time_mesh, earlier_time_mesh = np.meshgrid(
        current_times, current_times, indexing="ij"
    )

    f_dp = force(current_time_mesh) * (
        momentum(current_time_mesh) - momentum(earlier_time_mesh)
    )

    si.vis.xyz_plot(
        prefix + "f_dp",
        current_time_mesh,
        earlier_time_mesh,
        f_dp,
        x_unit="asec",
        x_label=r"$ t $",
        y_unit="asec",
        y_label=r"$ t' $",
        z_unit=u.atomic_force * u.atomic_momentum,
        z_label=r"$ \mathcal{F}(t) \times \Delta p(t, t') $",
        colormap=plt.get_cmap("RdBu_r"),
        x_lower_limit=-p_bound * pulse.pulse_width,
        y_lower_limit=-p_bound * pulse.pulse_width,
        x_upper_limit=p_bound * pulse.pulse_width,
        y_upper_limit=p_bound * pulse.pulse_width,
        title=r"Force $\times$ Momentum",
        **PLOT_KWARGS,
    )

    si.vis.xyz_plot(
        prefix + "f_dp__ABS",
        current_time_mesh,
        earlier_time_mesh,
        np.abs(f_dp),
        x_unit="asec",
        x_label=r"$ t $",
        y_unit="asec",
        y_label=r"$ t' $",
        z_unit=u.atomic_force * u.atomic_momentum,
        z_label=r"$ \left| \mathcal{F}(t) \times \Delta p(t, t') \right| $",
        x_lower_limit=-p_bound * pulse.pulse_width,
        y_lower_limit=-p_bound * pulse.pulse_width,
        x_upper_limit=p_bound * pulse.pulse_width,
        y_upper_limit=p_bound * pulse.pulse_width,
        title=r"Force $\times$ Momentum (Abs.)",
        **PLOT_KWARGS,
    )

    with si.utils.BlockTimer() as timer:
        f_dp_K = f_dp * determine_K_from_earlier_time(
            pulse, current_time_mesh, earlier_time_mesh
        )
    print("time to do the K integrals", timer)
    normed = f_dp_K / np.nanmean(np.abs(f_dp_K))

    si.vis.xyz_plot(
        prefix + "f_dp_k",
        current_time_mesh,
        earlier_time_mesh,
        normed,
        x_unit="asec",
        x_label=r"$ t $",
        y_unit="asec",
        y_label=r"$ t' $",
        colormap=plt.get_cmap("richardson"),
        richardson_equator_magnitude=5,
        x_lower_limit=-p_bound * pulse.pulse_width,
        y_lower_limit=-p_bound * pulse.pulse_width,
        x_upper_limit=p_bound * pulse.pulse_width,
        y_upper_limit=p_bound * pulse.pulse_width,
        title=r"$ \mathcal{K}(t, \, t') \times \mathcal{F}(t) \times \Delta p(t, t') $",
        **PLOT_KWARGS,
    )

    si.vis.xyz_plot(
        prefix + "f_dp_k__ABS",
        current_time_mesh,
        earlier_time_mesh,
        np.abs(f_dp_K) / np.nanmean(np.abs(f_dp_K)),
        x_unit="asec",
        x_label=r"$ t $",
        y_unit="asec",
        y_label=r"$ t' $",
        x_lower_limit=-p_bound * pulse.pulse_width,
        y_lower_limit=-p_bound * pulse.pulse_width,
        x_upper_limit=p_bound * pulse.pulse_width,
        y_upper_limit=p_bound * pulse.pulse_width,
        z_upper_limit=10,
        title=r"$ \left| \mathcal{K}(t, \, t') \times \mathcal{F}(t) \times \Delta p(t, t') \right| $ (norm. by avg.)",
        **PLOT_KWARGS,
    )


def determine_K_from_earlier_time(pulse, current_time, earlier_time):
    if earlier_time > current_time:
        return np.NaN

    num = si.math.complex_quadrature(
        lambda tp: pulse.get_electric_field_amplitude(tp)
        * ide.hydrogen_kernel_LEN(current_time - tp),
        earlier_time,
        current_time,
    )[0]
    den = integ.quadrature(
        pulse.get_electric_field_amplitude, earlier_time, current_time
    )[0]

    if den == 0:
        return np.NaN

    return num / den


determine_K_from_earlier_time = np.vectorize(
    determine_K_from_earlier_time, otypes=[np.complex128]
)

if __name__ == "__main__":
    with LOGMAN as logger:
        pulse_widths = np.array([50, 100, 150, 200, 250, 300, 500, 800]) * u.asec
        fluences = np.array([1]) * u.Jcm2
        phases = np.array([0, 0.5]) * u.pi

        pulses = [
            ion.potentials.GaussianPulse.from_number_of_cycles(
                pulse_width=pw, fluence=flu, phase=cep, number_of_cycles=3
            )
            for pw in pulse_widths
            for flu in fluences
            for cep in phases
        ]

        si.utils.multi_map(make_f_dp_k_plots_by_earlier_time, pulses, processes=3)
        # si.utils.multi_map(make_f_dp_k_plots, pulses, processes = 3)
