import logging
import os

import simulacra as si
import simulacra.units as u

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

if __name__ == "__main__":
    with si.utils.LogManager(
        "simulacra", "ionization", stdout_logs=True, stdout_level=logging.INFO
    ) as logger:
        sim = mesh.SphericalHarmonicSpecification(
            "sim_size_test",
            r_bound=100 * u.bohr_radius,
            r_points=100 * 10,
            l_bound=500,
            use_numeric_eigenstates=True,
            numeric_eigenstate_max_angular_momentum=5,
            numeric_eigenstate_max_energy=20 * u.eV,
            time_initial=0,
            time_final=100 * u.asec,
            time_step=1 * u.asec,
            electric_potential=potentials.SineWave.from_photon_energy(
                10 * u.eV, amplitude=1 * u.atomic_electric_field
            ),
            store_data_every=1,
        ).to_sim()
        sim.run(progress_bar=True)

        print(sim.info())

        print()

        sim.file_name = "with_mesh"
        path_with_mesh = sim.save(target_dir=OUT_DIR, save_mesh=True, compressed=False)
        loaded_with_mesh = mesh.MeshSimulation.load(path_with_mesh)
        print(
            f"{sim.file_name}'s actual size on disk: {si.utils.get_file_size_as_string(path_with_mesh)}"
        )

        sim.file_name = "with_mesh__compressed"
        path_with_mesh_compressed = sim.save(
            target_dir=OUT_DIR, save_mesh=True, compressed=True
        )
        loaded_with_mesh_compressed = mesh.MeshSimulation.load(path_with_mesh)
        print(
            f"{sim.file_name}'s actual size on disk: {si.utils.get_file_size_as_string(path_with_mesh_compressed)}"
        )

        sim.file_name = "without_mesh"
        path_without_mesh = sim.save(
            target_dir=OUT_DIR, save_mesh=False, compressed=False
        )
        loaded_without_mesh = mesh.MeshSimulation.load(path_without_mesh)
        print(
            f"{sim.file_name}'s actual size on disk: {si.utils.get_file_size_as_string(path_without_mesh)}"
        )

        sim.file_name = "without_mesh__compressed"
        path_without_mesh_compressed = sim.save(
            target_dir=OUT_DIR, save_mesh=False, compressed=True
        )
        loaded_without_mesh = mesh.MeshSimulation.load(path_without_mesh_compressed)
        print(
            f"{sim.file_name}'s actual size on disk: {si.utils.get_file_size_as_string(path_without_mesh_compressed)}"
        )
