import logging
import os

import numpy as np

import simulacra as si
from simulacra.units import *
import ionization as ion

# import matplotlib.pyplot as plt

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)
SIM_LIB = os.path.join(OUT_DIR, "simlib")

logman = si.utils.LogManager("simulacra", "ionization", stdout_level=logging.INFO)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=5)


def instantaneous_tunneling_rate(
    electric_field_amplitude, ionization_potential=-rydberg
):
    # f = np.abs(electric_field_amplitude / atomic_electric_field)
    #
    # return (4 / f) * (electron_mass_reduced * (proton_charge ** 4) / (hbar ** 3)) * np.exp(-(2 / 3) / f)

    amplitude_scaled = np.abs(electric_field_amplitude / atomic_electric_field)
    potential_scaled = np.abs(ionization_potential / hartree)

    f = amplitude_scaled / ((2 * potential_scaled) ** 1.5)

    return np.where(
        np.not_equal(electric_field_amplitude, 0),
        (4 / f) * np.exp(-2 / (3 * f)) / atomic_time,
        0,
    )
    # return 4 * np.sqrt(4 / (3 * f)) * np.exp(-2 / (3 * f)) / atomic_time

    # e_a = (electron_mass_reduced ** 2) * (proton_charge ** 5) / (((4 * pi * epsilon_0) ** 3) * (hbar ** 4))
    # w_a = (electron_mass_reduced * (proton_charge ** 4)) / (((4 * pi * epsilon_0) ** 2) * (hbar ** 3))
    # f = e_a / np.abs(electric_field_amplitude)

    # return 4 * w_a * f * np.exp(-2 * f / 3)


def averaged_tunneling_rate(electric_field_amplitude, ionization_potential=-rydberg):
    # f = np.abs(electric_field_amplitude / atomic_electric_field)
    #
    # return (4 / f) * (electron_mass_reduced * (proton_charge ** 4) / (hbar ** 3)) * np.exp(-(2 / 3) / f)

    amplitude_scaled = np.abs(electric_field_amplitude / atomic_electric_field)
    potential_scaled = np.abs(ionization_potential / hartree)

    f = amplitude_scaled / ((2 * potential_scaled) ** 1.5)

    # return (4 / f) * np.exp(-2 / (3 * f)) / atomic_time
    return np.where(
        np.not_equal(electric_field_amplitude, 0),
        4 * np.sqrt(3 / (pi * f)) * np.exp(-2 / (3 * f)) / atomic_time,
        0,
    )
    # return np.where(np.not_equal(electric_field_amplitude, 0), 4 * np.sqrt(4 / (3 * f)) * np.exp(-2 / (3 * f)) / atomic_time, 0)


def compare_quasistatic_to_tdse(intensity, photon_energy):
    dummy = ion.SineWave.from_photon_energy(0.5 * eV)

    time_initial = 0
    time_final = 8 * dummy.period_carrier

    time_front = 1 * dummy.period_carrier
    time_plateau = 5 * dummy.period_carrier

    efield = ion.SineWave.from_photon_energy_and_intensity(photon_energy, intensity)
    efield.window = ion.SmoothedTrapezoidalWindow(
        time_front=time_front, time_plateau=time_plateau
    )

    r_bound = 100
    r_points = 4 * r_bound
    l_bound = 400
    dt = 4
    store = 5

    h = hash(
        (
            intensity,
            photon_energy,
            time_initial,
            time_final,
            time_front,
            time_plateau,
            r_bound,
            r_points,
            l_bound,
            dt,
            store,
        )
    )

    title = f"P={uround(intensity, atomic_intensity, 5)}_E={uround(photon_energy, eV, 1)}_R={r_bound}_Rp={r_points}_L={l_bound}_dt={dt}_sde={store}"

    spec = ion.SphericalHarmonicSpecification(
        f"tdse_{h}",
        r_bound=r_bound * bohr_radius,
        r_points=r_points,
        l_bound=l_bound,
        internal_potential=ion.SoftCoulomb(softening_distance=0.05 * bohr_radius),
        time_initial=time_initial,
        time_final=time_final,
        time_step=dt * asec,
        electric_potential=efield,
        use_numeric_eigenstates=True,
        numeric_eigenstate_max_energy=20 * eV,
        numeric_eigenstate_max_angular_momentum=5,
        store_data_every=store,
        checkpoints=True,
        checkpoint_dir=SIM_LIB,
        checkpoint_every=50,
        mask=ion.RadialCosineMask(0.9 * r_bound * bohr_radius, r_bound * bohr_radius),
    )

    sim = si.utils.find_or_init_sim(spec, search_dir=SIM_LIB)

    sim.info().log()
    if not sim.status == si.Status.FINISHED:
        sim.run_simulation(progress_bar=True)
        sim.info().log()

    sim.save(target_dir=SIM_LIB)

    sim.plot_wavefunction_vs_time(**PLOT_KWARGS)

    times = np.linspace(time_initial, time_final, 1e3)

    si.vis.xy_plot(
        title + "__efield_vs_time",
        times,
        efield.get_electric_field_amplitude(times),
        x_label=r"Time $t$",
        x_unit="asec",
        y_label=fr"$ {ion.LATEX_EFIELD}(t) $)",
        y_unit="atomic_electric_field",
        **PLOT_KWARGS,
    )

    tunneling_rate_vs_time = instantaneous_tunneling_rate(
        efield.get_electric_field_amplitude(times), sim.spec.initial_state.energy
    )

    si.vis.xy_plot(
        title + "__tunneling_rate_vs_time",
        times,
        tunneling_rate_vs_time * asec,
        x_label=r"Time $t$",
        x_unit="asec",
        y_label=r"Tunneling Rate ($\mathrm{as}^{-1}$)",
        **PLOT_KWARGS,
    )

    wavefunction_remaining = np.empty_like(times)
    wavefunction_remaining[0] = 1
    for ii, tunneling_rate in enumerate(tunneling_rate_vs_time[:-1]):
        wavefunction_remaining[ii + 1] = wavefunction_remaining[ii] * (
            1 - (tunneling_rate * np.abs(times[ii + 1] - times[ii]))
        )

    cycle_avg_rate = averaged_tunneling_rate(efield.amplitude)
    cycle_avg_norm_remaining_plat = np.exp(-cycle_avg_rate * time_plateau)
    cycle_avg_norm_remaining_all = np.exp(
        -cycle_avg_rate * (time_plateau + (2 * time_front))
    )

    for log in (True, False):
        si.vis.xxyy_plot(
            title + f"__comparison__log={log}",
            (sim.data_times, sim.data_times, sim.data_times, times),
            (
                sim.norm_vs_time,
                sim.total_bound_state_overlap_vs_time,
                sim.state_overlaps_vs_time[sim.spec.initial_state],
                wavefunction_remaining,
            ),
            line_labels=(
                "TDSE Norm",
                "TDSE Bound States",
                "TDSE Initial State",
                "Tunneling",
            ),
            x_label=r"Time $t$",
            x_unit="fsec",
            y_label="Remaining Wavefunction",
            y_log_axis=log,
            hlines=[cycle_avg_norm_remaining_plat, cycle_avg_norm_remaining_all],
            hline_kwargs=[{"linestyle": "--"}, {"linestyle": ":"}],
            **PLOT_KWARGS,
        )


if __name__ == "__main__":
    with logman as logger:
        intensities = np.linspace(
            (1 / 35 ** 2) * atomic_intensity, (1 / 10 ** 2) * atomic_intensity, 1e5
        )
        efields = np.sqrt(2 * intensities / (c * epsilon_0))

        quasi_static_tunneling_rates = averaged_tunneling_rate(efields)

        si.vis.xy_plot(
            "rate_vs_field",
            efields,
            quasi_static_tunneling_rates,
            x_label=fr"Electric Field $ {ion.LATEX_EFIELD}_0 $",
            x_unit="atomic_electric_field",
            y_label=fr"Tunneling Rate ($\mathrm{{s^{{-1}}}}$)",
            y_log_axis=True,
            **PLOT_KWARGS,
        )

        si.vis.xy_plot(
            "rate_vs_intensity",
            intensities,
            quasi_static_tunneling_rates,
            x_label="Intensity $P$",
            x_unit="atomic_intensity",
            y_label=fr"Tunneling Rate ($\mathrm{{s^{{-1}}}}$)",
            y_log_axis=True,
            **PLOT_KWARGS,
        )

        si.vis.xy_plot(
            "rate_vs_inv_sqrt_intensity",
            1 / np.sqrt(intensities / atomic_intensity),
            quasi_static_tunneling_rates,
            x_label="$ 1 / \sqrt{P / P_{\mathrm{atomic}}} $",
            y_label=fr"Tunneling Rate ($\mathrm{{s^{{-1}}}}$)",
            y_log_axis=True,
            **PLOT_KWARGS,
        )

        # compare_quasistatic_to_tdse(atomic_intensity / (10 ** 2), .5 * eV)
        # for intensity in [.0025]:
        for intensity in [0.0020, 0.0025, 0.0030]:
            compare_quasistatic_to_tdse(intensity * atomic_intensity, 0.5 * eV)

        ##############

        # intensity = .0025 * atomic_intensity
        # efield = np.sqrt(2 * intensity / (c * epsilon_0))
        # rate = averaged_tunneling_rate(efield)
        #
        # print(rate)
        # print(rate / THz)
        #
        # dummy = ion.SineWave.from_photon_energy(.5 * eV)
        # plat_time = 5 * dummy.period
        #
        # remaining = np.exp(-rate * plat_time)
        # print(remaining)
