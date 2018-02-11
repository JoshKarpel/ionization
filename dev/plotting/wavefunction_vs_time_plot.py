import os
import logging

import simulacra as si
from simulacra.units import *
import ionization as ion


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization', stdout_level = logging.INFO) as logger:
        name = 'test_1_higher_energy_higher_amp_shorter'
        try:
            sim = ion.ElectricFieldSimulation.load(os.path.join(OUT_DIR, f'{name}.sim'))
        except FileNotFoundError:
            sim = ion.SphericalHarmonicSpecification(
                    name,
                    r_bound = 100 * bohr_radius,
                    r_points = 400,
                    l_bound = 100,
                    use_numeric_eigenstates = True,
                    numeric_eigenstate_max_angular_momentum = 10,
                    numeric_eigenstate_max_energy = 100 * eV,
                    time_initial = 0 * asec, time_final = 1000 * asec,
                    electric_potential = ion.SineWave.from_photon_energy((13.6 + 10) * eV, amplitude = .2 * atomic_electric_field),
                    # electric_potential = ion.Rectangle(start_time = -200 * asec, end_time = 200 * asec, amplitude = .2 * atomic_electric_field)
                    checkpoints = True,
                    checkpoint_dir = OUT_DIR,
                    store_data_every = 1,
                    checkpoint_every = 50,
            ).to_sim()

        sim.info().log()
        if sim.status != si.Status.FINISHED:
            sim.run(progress_bar = True)
            sim.info().log()

            sim.save(target_dir = OUT_DIR, save_mesh = True)

        PLOT_KWARGS = dict(
                target_dir = OUT_DIR,
                img_format = 'png',
                fig_dpi_scale = 3,
        )

        e_states, e_labels = sim.group_free_states_by_continuous_attr('energy', attr_unit = 'eV', cutoff_value = 100 * eV)

        sim.plot_wavefunction_vs_time(
                name_postfix = '__collapsed__grouped_by_energy',
                collapse_bound_state_angular_momenta = True,
                grouped_free_states = e_states, group_free_states_labels = e_labels,
                **PLOT_KWARGS,
        )

        sim.plot_wavefunction_vs_time(
                name_postfix = '__grouped_by_energy',
                collapse_bound_state_angular_momenta = False,
                grouped_free_states = e_states, group_free_states_labels = e_labels,
                **PLOT_KWARGS,
        )

        l_states, l_labels = sim.group_free_states_by_discrete_attr('l', cutoff_value = 20)

        sim.plot_wavefunction_vs_time(
                name_postfix = '__collapsed_grouped_by_L',
                collapse_bound_state_angular_momenta = True,
                grouped_free_states = l_states, group_free_states_labels = l_labels,
                **PLOT_KWARGS,
        )

        sim.plot_wavefunction_vs_time(
                name_postfix = '__grouped_by_L',
                collapse_bound_state_angular_momenta = False,
                grouped_free_states = l_states, group_free_states_labels = l_labels,
                **PLOT_KWARGS,
        )

        gs = (None, sim.mesh.get_g_with_states_removed(sim.bound_states))

        for log in (True, False):
            for g in gs:
                if g is not None:
                    ps = '__rm_bound'
                else:
                    ps = ''

                sim.mesh.plot_electron_momentum_spectrum(
                        name_postfix = '__wavenumber_spectrum' + ps,
                        r_type = 'wavenumber', r_unit = 'per_nm',
                        log = log,
                        g = g,
                        **PLOT_KWARGS,
                )

                sim.mesh.plot_electron_momentum_spectrum(
                        name_postfix = '__energy_spectrum' + ps,
                        r_type = 'energy', r_unit = 'eV',
                        r_lower_lim = .1 * eV, r_upper_lim = 50 * eV,
                        log = log,
                        g = g,
                        **PLOT_KWARGS,
                )

                sim.mesh.plot_electron_momentum_spectrum(
                        name_postfix = '__momentum_spectrum' + ps,
                        r_type = 'momentum', r_unit = 'atomic_momentum',
                        r_lower_lim = .01 * atomic_momentum, r_upper_lim = 10 * atomic_momentum,
                        log = log,
                        g = g,
                        **PLOT_KWARGS,
                )
