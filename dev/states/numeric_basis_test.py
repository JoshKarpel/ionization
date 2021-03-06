import logging
import os

import simulacra as si
from simulacra.units import *

import ionization as ion


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)


def run(spec):
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
        sim = spec.to_sim()

        sim.run()
        print(sim.info())

        sim.save(target_dir=OUT_DIR, save_mesh=False)

        plot_kwargs = dict(target_dir=OUT_DIR, bound_state_max_n=4)

        for log in (True, False):
            plot_kwargs["log"] = log

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

        spec_kwargs = dict(
            r_bound=bound * bohr_radius,
            r_points=bound * points_per_bohr_radius,
            l_bound=70,
            initial_state=ion.HydrogenBoundState(1, 0),
            time_initial=0 * asec,
            time_final=500 * asec,
            time_step=1 * asec,
            use_numeric_eigenstates=True,
            numeric_eigenstate_max_energy=200 * eV,
            numeric_eigenstate_max_angular_momentum=50,
            electric_potential=ion.SineWave.from_photon_energy(
                rydberg + 5 * eV, amplitude=0.5 * atomic_electric_field
            ),
            mask=ion.RadialCosineMask(
                inner_radius=0.8 * bound * bohr_radius, outer_radius=bound * bohr_radius
            ),
            out_dir=OUT_DIR,
        )

        every = [1, 5, 10, 100, 600]
        specs = [
            ion.SphericalHarmonicSpecification(
                "eig_{}".format(e), store_data_every=e, **spec_kwargs
            )
            for e in every
        ]

        si.utils.multi_map(run, specs, processes=2)
