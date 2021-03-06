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
        bound = 200
        points_per_bohr_radius = 4
        l_bound = 200

        pw = 100
        t_bound = 30 * pw

        flu = 0.1

        store_every = 1

        ani = [
            src.ionization.animators.SphericalHarmonicAnimator(
                target_dir=OUT_DIR,
                top_right_axis_manager_type=src.ionization.animators.AngularMomentumDecompositionAxis,
                length=10,
            )
        ]
        # ani = []

        window = ion.potentials.LogisticWindow(
            window_time=0.9 * t_bound * asec, window_width=10 * asec
        )
        efield = ion.potentials.SumOfSinesPulse(
            pulse_width=pw * asec,
            pulse_frequency_ratio=25,
            number_of_modes=71,
            fluence=flu * Jcm2,
            window=window,
        )

        spec_kwargs = dict(
            r_bound=bound * bohr_radius,
            r_points=bound * points_per_bohr_radius,
            l_bound=l_bound,
            initial_state=ion.HydrogenBoundState(1, 0),
            time_initial=-t_bound * asec,
            time_final=t_bound * asec,
            time_step=1 * asec,
            use_numeric_eigenstates=True,
            numeric_eigenstate_max_energy=50 * eV,
            numeric_eigenstate_max_angular_momentum=20,
            electric_potential=efield,
            electric_potential_dc_correction=True,
            mask=ion.RadialCosineMask(
                inner_radius=0.8 * bound * bohr_radius, outer_radius=bound * bohr_radius
            ),
            store_norm_by_l=True,
            store_data_every=store_every,
            animators=ani,
            snapshot_times=[10 * pw * asec],
            snapshot_indices=[-1],
            snapshot_type=ion.SphericalHarmonicSnapshot,
        )

        sim = ion.SphericalHarmonicSpecification(
            "SumOfSines__flu={}jcm2__{}at{}x{}__data_rate={}".format(
                flu, bound, points_per_bohr_radius, l_bound, store_every
            ),
            **spec_kwargs
        ).to_sim()

        print(sim.info())
        sim.run()
        print(sim.info())

        # path = sim.save(target_dir = OUT_DIR, save_mesh = True)
        # sim = si.Simulation.load(path)

        sim.mesh.plot_g2(target_dir=OUT_DIR)
        sim.mesh.plot_g2(
            target_dir=OUT_DIR, name_postfix="_25", plot_limit=50 * bohr_radius
        )

        plot_kwargs = dict(target_dir=OUT_DIR, bound_state_max_n=4)

        sim.plot_wavefunction_vs_time(**plot_kwargs, name_postfix="__no_grouping")
        sim.plot_wavefunction_vs_time(
            **plot_kwargs,
            name_postfix="__no_grouping__collapsed_l",
            collapse_bound_state_angular_momenta=True
        )

        grouped_states, group_labels = sim.group_free_states_by_continuous_attr(
            "energy", divisions=12, cutoff_value=150 * eV, label_unit_value="eV"
        )
        sim.plot_wavefunction_vs_time(
            **plot_kwargs,
            name_postfix="__energy",
            grouped_free_states=grouped_states,
            group_free_states_labels=group_labels
        )
        sim.plot_wavefunction_vs_time(
            **plot_kwargs,
            name_postfix="__energy__collapsed_l",
            collapse_bound_state_angular_momenta=True,
            grouped_free_states=grouped_states,
            group_free_states_labels=group_labels
        )

        grouped_states, group_labels = sim.group_free_states_by_discrete_attr(
            "l", cutoff_value=20
        )
        sim.plot_wavefunction_vs_time(
            **plot_kwargs,
            name_postfix="__l",
            grouped_free_states=grouped_states,
            group_free_states_labels=group_labels
        )
        sim.plot_wavefunction_vs_time(
            **plot_kwargs,
            name_postfix="__l__collapsed_l",
            collapse_bound_state_angular_momenta=True,
            grouped_free_states=grouped_states,
            group_free_states_labels=group_labels
        )

        sim.plot_energy_spectrum(
            **plot_kwargs,
            energy_upper_limit=50 * eV,
            states="all",
            group_angular_momentum=False
        )
        sim.plot_energy_spectrum(
            **plot_kwargs, states="bound", group_angular_momentum=False
        )
        sim.plot_energy_spectrum(
            **plot_kwargs,
            energy_upper_limit=50 * eV,
            states="free",
            bins=25,
            group_angular_momentum=False
        )

        sim.plot_energy_spectrum(
            **plot_kwargs,
            energy_upper_limit=50 * eV,
            states="all",
            angular_momentum_cutoff=10
        )
        sim.plot_energy_spectrum(
            **plot_kwargs, states="bound", angular_momentum_cutoff=10
        )
        sim.plot_energy_spectrum(
            **plot_kwargs,
            bins=25,
            energy_upper_limit=50 * eV,
            states="free",
            angular_momentum_cutoff=10
        )

        spectrum_kwargs = dict(target_dir=OUT_DIR, r_points=500)

        for log in (True, False):
            sim.mesh.plot_electron_momentum_spectrum(
                r_type="energy",
                r_unit="eV",
                r_lower_lim=0.1 * eV,
                r_upper_lim=100 * eV,
                log=log,
                **spectrum_kwargs
            )
            sim.mesh.plot_electron_momentum_spectrum(
                r_type="wavenumber",
                r_upper_lim=100 * per_nm,
                log=log,
                **spectrum_kwargs
            )
            sim.mesh.plot_electron_momentum_spectrum(
                r_type="momentum",
                r_unit="atomic_momentum",
                r_lower_lim=0.01 * atomic_momentum,
                r_upper_lim=5 * atomic_momentum,
                log=log,
                **spectrum_kwargs
            )

            sim.mesh.plot_electron_momentum_spectrum(
                r_type="energy",
                r_unit="eV",
                r_lower_lim=0.1 * eV,
                r_upper_lim=100 * eV,
                log=log,
                g=sim.mesh.get_g_with_states_removed(sim.bound_states),
                name_postfix="__bound_removed",
                **spectrum_kwargs
            )
            sim.mesh.plot_electron_momentum_spectrum(
                r_type="wavenumber",
                r_upper_lim=40 * per_nm,
                log=log,
                g=sim.mesh.get_g_with_states_removed(sim.bound_states),
                name_postfix="__bound_removed",
                **spectrum_kwargs
            )
            sim.mesh.plot_electron_momentum_spectrum(
                r_type="momentum",
                r_unit="atomic_momentum",
                r_lower_lim=0.01 * atomic_momentum,
                r_upper_lim=5 * atomic_momentum,
                log=log,
                g=sim.mesh.get_g_with_states_removed(sim.bound_states),
                name_postfix="__bound_removed",
                **spectrum_kwargs
            )
