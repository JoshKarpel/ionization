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
        bound = 70
        points_per_bohr_radius = 4

        dt = 0.1
        store_every = int(1 / dt)

        spec_kwargs = dict(
            r_bound=bound * bohr_radius,
            r_points=bound * points_per_bohr_radius,
            l_bound=70,
            initial_state=ion.HydrogenBoundState(1, 0),
            time_initial=0 * asec,
            time_final=500 * asec,
            time_step=dt * asec,
            use_numeric_eigenstates=True,
            numeric_eigenstate_max_energy=200 * eV,
            numeric_eigenstate_max_angular_momentum=50,
            electric_potential=ion.SineWave.from_photon_energy(
                rydberg + 5 * eV, amplitude=0.5 * atomic_electric_field
            ),
            mask=ion.RadialCosineMask(
                inner_radius=0.8 * bound * bohr_radius, outer_radius=bound * bohr_radius
            ),
        )

        analytic_test_states = [
            ion.HydrogenBoundState(n, l) for n in range(6) for l in range(n)
        ]

        sims = [
            ion.SphericalHarmonicSpecification(
                "dt={}as_every{}".format(dt, store_every),
                store_data_every=store_every,
                **spec_kwargs
            ).to_sim()
        ]

        for sim in sims:
            sim.run()

            sim.save(target_dir=OUT_DIR, save_mesh=False)

            plot_kwargs = dict(
                target_dir=OUT_DIR, img_format="pdf", bound_state_max_n=4
            )

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
