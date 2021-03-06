import datetime as dt
import logging
import os

import numpy as np
import simulacra as si
from simulacra.units import *
from tqdm import tqdm

import ionization as ion


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(
    os.getcwd(), "out", FILE_NAME, dt.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
)


def dict_to_arrays(d):
    k_array = np.zeros(len(d), dtype=np.float64) * np.NaN
    v_array = np.zeros(len(d), dtype=np.float64) * np.NaN
    for ii, (k, v) in enumerate(sorted(d.items())):
        k_array[ii] = k
        v_array[ii] = v

    return k_array, v_array


if __name__ == "__main__":
    with si.utils.LogManager(
        "simulacra",
        "ionization",
        stdout_level=logging.DEBUG,
        file_logs=True,
        file_dir=OUT_DIR,
        file_name="log",
    ) as logger:
        pw = 50  # asec
        flu = 1  # Jcm2
        phase = "cos"

        time_bound = 10  # multiples of pw
        dt = 1  # asec

        space_bound = 200  # BR
        points_per_bohr_radius = 4
        l_bound = 20

        initial_state = ion.HydrogenBoundState(1, 0)

        test_states_bound_max_n = 10
        test_states_free_per_l = 1000
        test_states_max_energy = 50  # eV
        test_states_l_points = 15  # < l_bound

        identifier = "{}_{}__{}_at_{}x{}__pw={}as__phase={}".format(
            initial_state.n,
            initial_state.l,
            space_bound,
            points_per_bohr_radius,
            l_bound,
            pw,
            phase,
        )

        # CONVERT TO REAL UNITS
        pw *= asec
        flu *= Jcm2

        time_bound *= pw
        dt *= asec

        r_points = space_bound * points_per_bohr_radius
        space_bound *= bohr_radius

        test_states_max_energy *= eV

        # PREP
        mask = ion.RadialCosineMask(
            inner_radius=0.8 * space_bound, outer_radius=space_bound
        )

        # electric_field = ion.potentials.SincPulse(pulse_width = pw, fluence = flu, dc_correction_time = time_bound, phase = phase,
        #                                window = ion.potentials.LogisticWindow(window_time = time_bound - pw, window_width = pw / 2))
        # electric_field = ion.potentials.Rectangle(start_time = -time_bound + 25 * asec, end_time = -time_bound + 25 * asec + pw, amplitude = 1 * atomic_electric_field)
        # electric_field = ion.SineWave.from_photon_energy(rydberg + 1 * eV, amplitude = .01 * atomic_electric_field)
        electric_field = ion.SineWave.from_photon_energy(
            rydberg + 5 * eV, amplitude=0.01 * atomic_electric_field
        )

        test_states = [
            ion.HydrogenBoundState(n, l)
            for n in range(test_states_bound_max_n + 1)
            for l in range(n)
        ]

        coulomb_state_energies = np.linspace(
            0 * eV, test_states_max_energy, test_states_free_per_l + 1
        )
        d_energy = np.abs(coulomb_state_energies[1] - coulomb_state_energies[0])
        logger.info("d_energy: {} eV | {} J".format(d_energy / eV, d_energy))
        coulomb_states = [
            ion.HydrogenCoulombState(energy=e, l=l)
            for e in coulomb_state_energies
            for l in range(test_states_l_points)
        ]

        # test_states += coulomb_states

        logger.info("Warning: using {} test states".format(len(test_states)))

        # SET UP SIM
        sim = ion.SphericalHarmonicSpecification(
            identifier,
            r_bound=space_bound,
            r_points=r_points,
            l_bound=l_bound,
            # time_initial = - time_bound, time_final = time_bound, time_step = dt,
            time_initial=0,
            time_final=electric_field.period_carrier * 20,
            time_step=dt,
            internal_potential=ion.Coulomb(),
            electric_potential=electric_field,
            initial_state=initial_state,
            test_states=test_states,
            mask=mask,
        ).to_sim()

        sim.info().log()
        sim.run()
        sim.info().log()

        sim.plot_wavefunction_vs_time(target_dir=OUT_DIR)

        overlap_by_energy = {s.energy: 0 for s in coulomb_states}
        overlap_by_l = {s.l: 0 for s in coulomb_states}
        overlap_by_state = {s: 0 for s in coulomb_states}

        logger.info(len(overlap_by_energy))
        logger.info(len(overlap_by_l))
        logger.info(len(overlap_by_state))

        for s in tqdm(coulomb_states):
            overlap = sim.mesh.state_overlap(s) * d_energy
            # overlap = sim.mesh.state_overlap(s) * d_energy / sim.mesh.state_overlap(s, s)

            overlap_by_energy[s.energy] += overlap
            overlap_by_l[s.l] += overlap
            overlap_by_state[s] = overlap

        energy, overlap_by_energy = dict_to_arrays(overlap_by_energy)
        l, overlap_by_l = dict_to_arrays(overlap_by_l)

        logger.info(energy)
        logger.info(overlap_by_energy)
        logger.info(l)
        logger.info(overlap_by_l)

        logger.info("total overlap by energy: {}".format(np.nansum(overlap_by_energy)))
        logger.info("total overlap by l: {}".format(np.nansum(overlap_by_l)))
        logger.info(
            "total overlap by state: {}".format(
                np.nansum(np.array(list(overlap_by_state.values())))
            )
        )

        cont_norm = np.nansum(overlap_by_energy)
        bound_norm = np.sum([x[-1] for x in sim.state_overlaps_vs_time.values()])

        logger.info("bound: {}".format(bound_norm))
        logger.info("bound + cont: {}".format(cont_norm + bound_norm))
        logger.info("norm - bound: {}".format(sim.norm_vs_time[-1] - bound_norm))
        logger.info("norm: {}".format(sim.norm_vs_time[-1]))

        si.vis.xy_plot(
            identifier + "__overlap_by_energy",
            energy,
            overlap_by_energy,
            x_unit="eV",
            target_dir=OUT_DIR,
        )

        si.vis.xy_plot(
            identifier + "__overlap_by_l", l, overlap_by_l, target_dir=OUT_DIR
        )
