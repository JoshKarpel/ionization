import logging
import os

import simulacra as si
from simulacra.units import *

import ionization as ion


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

if __name__ == "__main__":
    with si.utils.LogManager(
        "simulacra",
        "ionization",
        stdout_logs=True,
        stdout_level=logging.DEBUG,
        file_logs=False,
        file_mode="w",
        file_dir=OUT_DIR,
        file_name="log",
    ) as logger:
        bound = 50
        points_per_bohr_radius = 4

        t_bound = 500
        t_extra = 0

        amp = 0.05
        phase = 0

        window = ion.potentials.LogisticWindow(
            window_time=0.9 * t_bound * asec, window_width=10 * asec
        )

        # efield = ion.SineWave.from_photon_energy(rydberg + 20 * eV, amplitude = .05 * atomic_electric_field,
        #                                                          window = ion.potentials.LogisticWindow(window_time = .9 * t_bound * asec, window_width = 10 * asec))

        efield = ion.SineWave.from_photon_energy(
            rydberg + 20 * eV,
            amplitude=amp * atomic_electric_field,
            phase=phase,
            window=ion.potentials.LogisticWindow(
                window_time=0.9 * t_bound * asec, window_width=10 * asec
            ),
        )
        #
        # efield += ion.SineWave.from_photon_energy(rydberg + 30 * eV, amplitude = .05 * atomic_electric_field,
        #                                           window = ion.potentials.LogisticWindow(window_time = .9 * t_bound * asec, window_width = 10 * asec))

        # efield = ion.SineWave(twopi * (c / (800 * nm)), amplitude = .01 * atomic_electric_field,
        #                       window = window)

        spec_kwargs = dict(
            r_bound=bound * bohr_radius,
            r_points=bound * points_per_bohr_radius,
            l_bound=20,
            initial_state=ion.HydrogenBoundState(1, 0),
            time_initial=-t_bound * asec,
            time_final=(t_bound + t_extra) * asec,
            time_step=1 * asec,
            use_numeric_eigenstates=True,
            numeric_eigenstate_max_energy=50 * eV,
            numeric_eigenstate_max_angular_momentum=5,
            electric_potential=efield,
            electric_potential_dc_correction=True,
            mask=ion.RadialCosineMask(
                inner_radius=0.8 * bound * bohr_radius, outer_radius=bound * bohr_radius
            ),
            store_data_every=-1,
        )

        sim = ion.SphericalHarmonicSpecification(
            f"PWTest_amp={amp}aef_phase={phase / pi:3f}pi__tB={t_bound}pw__tE={t_extra}asec",
            **spec_kwargs,
        ).to_sim()

        sim.run()
        print(sim.info())

        # sim.mesh.plot_g(target_dir = OUT_DIR)
        # sim.mesh.plot_g(target_dir = OUT_DIR, name_postfix = '_25', plot_limit = 25 * bohr_radius)
        #
        # plot_kwargs = dict(
        #     target_dir = OUT_DIR,
        #     bound_state_max_n = 4,
        # )
        #
        # sim.plot_wavefunction_vs_time(**plot_kwargs, name_postfix = '__no_grouping')
        # sim.plot_wavefunction_vs_time(**plot_kwargs, name_postfix = '__no_grouping__collapsed_l',
        #                               collapse_bound_state_angular_momentums = True)
        #
        # grouped_states, group_labels = sim.group_free_states_by_continuous_attr('energy', divisions = 12, cutoff_value = 150 * eV, label_unit = 'eV')
        # sim.plot_wavefunction_vs_time(**plot_kwargs, name_postfix = '__energy',
        #                               grouped_free_states = grouped_states, group_labels = group_labels)
        # sim.plot_wavefunction_vs_time(**plot_kwargs, name_postfix = '__energy__collapsed_l',
        #                               collapse_bound_state_angular_momentums = True,
        #                               grouped_free_states = grouped_states, group_labels = group_labels)
        #
        # grouped_states, group_labels = sim.group_free_states_by_discrete_attr('l', cutoff_value = 20)
        # sim.plot_wavefunction_vs_time(**plot_kwargs, name_postfix = '__l',
        #                               grouped_free_states = grouped_states, group_labels = group_labels)
        # sim.plot_wavefunction_vs_time(**plot_kwargs, name_postfix = '__l__collapsed_l',
        #                               collapse_bound_state_angular_momentums = True,
        #                               grouped_free_states = grouped_states, group_labels = group_labels)
        #
        # sim.plot_energy_spectrum(**plot_kwargs,
        #                          energy_upper_limit = 50 * eV, states = 'all',
        #                          group_angular_momentum = False)
        # sim.plot_energy_spectrum(**plot_kwargs,
        #                          states = 'bound',
        #                          group_angular_momentum = False)
        # sim.plot_energy_spectrum(**plot_kwargs,
        #                          energy_upper_limit = 50 * eV, states = 'free',
        #                          bins = 25,
        #                          group_angular_momentum = False)
        #
        # sim.plot_energy_spectrum(**plot_kwargs,
        #                          energy_upper_limit = 50 * eV, states = 'all',
        #                          angular_momentum_cutoff = 10)
        # sim.plot_energy_spectrum(**plot_kwargs,
        #                          states = 'bound',
        #                          angular_momentum_cutoff = 10)
        # sim.plot_energy_spectrum(**plot_kwargs,
        #                          bins = 25,
        #                          energy_upper_limit = 50 * eV, states = 'free',
        #                          angular_momentum_cutoff = 10)

        spectrum_kwargs = dict(target_dir=OUT_DIR, r_points=500)

        for log in (True, False):
            # thetas, wavenumbers, along_z = sim.mesh.inner_product_with_plane_waves(thetas = [0], wavenumbers = np.linspace(.1, 50, 500) * per_nm,
            #                                                                        g_mesh = sim.mesh.get_g_with_states_removed(sim.bound_states))
            #
            # si.utils.xy_plot('along_theta=0__log={}__wavenumber'.format(log),
            #                  wavenumbers[0],
            #                  np.abs(along_z[0]) ** 2,
            #                  x_unit = 'per_nm', x_label = 'Wavenumber $wavenumber$',
            #                  y_log_axis = log,
            #                  target_dir = OUT_DIR)
            #
            # si.utils.xy_plot('along_theta=0__log={}__momentum'.format(log),
            #                  wavenumbers[0] * hbar,
            #                  np.abs(along_z[0]) ** 2,
            #                  x_unit = 'atomic_momentum', x_label = 'Momentum $p$',
            #                  y_log_axis = log,
            #                  target_dir = OUT_DIR)

            # thetas, wavenumbers, along_z = sim.mesh.inner_product_with_plane_waves_at_infinity(thetas = [0], wavenumbers = np.linspace(.1, 50, 500) * per_nm,
            #                                                                                    g_mesh = sim.mesh.get_g_with_states_removed(sim.bound_states))
            #
            # si.utils.xy_plot('along_theta=0__log={}__wavenumber__at_inf'.format(log),
            #                  wavenumbers[0],
            #                  np.abs(along_z[0]) ** 2,
            #                  x_unit = 'per_nm', x_label = 'Wavenumber $wavenumber$',
            #                  y_log_axis = log,
            #                  target_dir = OUT_DIR)
            #
            # si.utils.xy_plot('along_theta=0__log={}__momentum__at_inf'.format(log),
            #                  wavenumbers[0] * hbar,
            #                  np.abs(along_z[0]) ** 2,
            #                  x_unit = 'atomic_momentum', x_label = 'Momentum $p$',
            #                  y_log_axis = log,
            #                  target_dir = OUT_DIR)

            with si.utils.BlockTimer() as t:
                sim.mesh.plot_electron_momentum_spectrum(
                    r_type="energy",
                    r_unit="eV",
                    r_lower_lim=0.1 * eV,
                    r_upper_lim=50 * eV,
                    log=log,
                    **spectrum_kwargs,
                )
                sim.mesh.plot_electron_momentum_spectrum(
                    r_type="wavenumber",
                    r_upper_lim=40 * per_nm,
                    log=log,
                    **spectrum_kwargs,
                )
                sim.mesh.plot_electron_momentum_spectrum(
                    r_type="momentum",
                    r_unit="atomic_momentum",
                    r_lower_lim=0.01 * atomic_momentum,
                    r_upper_lim=2.5 * atomic_momentum,
                    log=log,
                    **spectrum_kwargs,
                )
            print("RAW PLOTS", t)

            with si.utils.BlockTimer() as t:
                sim.mesh.plot_electron_momentum_spectrum(
                    r_type="energy",
                    r_unit="eV",
                    r_lower_lim=0.1 * eV,
                    r_upper_lim=50 * eV,
                    log=log,
                    g=sim.mesh.get_g_with_states_removed(sim.bound_states),
                    name_postfix="__bound_removed",
                    **spectrum_kwargs,
                )
                sim.mesh.plot_electron_momentum_spectrum(
                    r_type="wavenumber",
                    r_upper_lim=40 * per_nm,
                    log=log,
                    g=sim.mesh.get_g_with_states_removed(sim.bound_states),
                    name_postfix="__bound_removed",
                    **spectrum_kwargs,
                )
                sim.mesh.plot_electron_momentum_spectrum(
                    r_type="momentum",
                    r_unit="atomic_momentum",
                    r_lower_lim=0.01 * atomic_momentum,
                    r_upper_lim=2.5 * atomic_momentum,
                    log=log,
                    g=sim.mesh.get_g_with_states_removed(sim.bound_states),
                    name_postfix="__bound_removed",
                    **spectrum_kwargs,
                )
            print("BOUND REMOVED PLOTS", t)
