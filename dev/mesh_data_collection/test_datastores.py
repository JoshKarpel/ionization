import logging
import os

import simulacra as si
import simulacra.units as u

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

LOGMAN = si.utils.LogManager("simulacra", "ionization", stdout_level=logging.INFO)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)


def plot_fields(sim):
    si.vis.xy_plot(
        "fields",
        sim.data.times,
        sim.data.electric_field_amplitude / u.atomic_electric_field,
        sim.data.vector_potential_amplitude * u.proton_charge / u.atomic_momentum,
        line_labels=[r"$ \mathcal{E}(t) $", r"$ e \, \mathcal{A}(t) $"],
        x_unit="asec",
        **PLOT_KWARGS,
    )


def plot_norm_and_inner_products(sim):
    si.vis.xy_plot(
        "norm_and_inner_products",
        sim.data.times,
        sim.data.norm,
        sim.data.initial_state_overlap,
        *[ip for ip in sim.data.state_overlaps.values()],
        line_labels=[
            r"$ \left\langle \Psi | \Psi \right\rangle $",
            r"$ \left| \left\langle \Psi | \Psi(t = 0) \right\rangle \right|^2 $",
            *[
                rf"$ \left| \left\langle \Psi | {s.tex} \right\rangle \right|^2 $"
                for s in sim.data.inner_products.keys()
            ],
        ],
        line_kwargs=[
            None,
            None,
            *[{"linestyle": "--"} for _ in range(len(sim.data.inner_products))],
        ],
        x_unit="asec",
        **PLOT_KWARGS,
    )


def plot_r_and_z_expectation_values(sim):
    si.vis.xy_plot(
        "r_and_z_expectation_values",
        sim.data.times,
        sim.data.r_expectation_value,
        sim.data.z_expectation_value,
        sim.data.z_dipole_moment_expectation_value / u.proton_charge,
        line_labels=[
            r"$ \left\langle \widehat{r} \right\rangle $",
            r"$ \left\langle \widehat{z} \right\rangle $",
            r"$ \left\langle \widehat{d_z} \right\rangle / e$",
        ],
        x_unit="asec",
        y_unit="bohr_radius",
        **PLOT_KWARGS,
    )


def plot_energy_expectation_values(sim):
    si.vis.xy_plot(
        "energy_expectation_values",
        sim.data.times,
        sim.data.internal_energy_expectation_value,
        sim.data.total_energy_expectation_value,
        line_labels=[
            r"$ \left\langle \mathcal{H_0} \right\rangle $",
            r"$ \left\langle \mathcal{H} \right\rangle $",
        ],
        x_unit="asec",
        y_unit="eV",
        **PLOT_KWARGS,
    )


def plot_norm_within_radius(sim):
    si.vis.xy_plot(
        "norm_within_radius",
        sim.data.times,
        *sim.data.norm_within_radius.values(),
        line_labels=[
            rf"$ \left\langle \Psi | \Psi \right\rangle (r < {r}) $"
            for r in sim.data.norm_within_radius.keys()
        ],
        x_unit="asec",
        **PLOT_KWARGS,
    )


if __name__ == "__main__":
    with LOGMAN as logger:
        potential = ion.potentials.Rectangle(
            start_time=10 * u.asec,
            end_time=40 * u.asec,
            amplitude=1 * u.atomic_electric_field,
        )
        potential += ion.potentials.Rectangle(
            start_time=60 * u.asec,
            end_time=90 * u.asec,
            amplitude=-1 * u.atomic_electric_field,
        )

        sim = mesh.SphericalHarmonicSpecification(
            "test",
            r_bound=50 * u.bohr_radius,
            r_points=500,
            l_bound=100,
            time_initial=0 * u.asec,
            time_final=100 * u.asec,
            time_step=1 * u.asec,
            electric_potential=potential,
            evolution_method=mesh.SplitInteractionOperator(),
            use_numeric_eigenstates=False,
            test_states=[
                states.HydrogenBoundState(n, l) for n in range(1, 3) for l in range(n)
            ],
            datastores=(
                mesh.Fields(),
                mesh.Norm(),
                mesh.InnerProducts(),
                mesh.RExpectationValue(),
                mesh.ZExpectationValue(),
                mesh.InternalEnergyExpectationValue(),
                mesh.TotalEnergyExpectationValue(),
                mesh.NormWithinRadius(
                    radii=(1 * u.bohr_radius, 5 * u.bohr_radius, 20 * u.bohr_radius)
                )
                # ion.mesh.DirectionalRadialProbabilityCurrent,
            ),
            # animators = [
            #     ion.mesh.anim.PolarAnimator(
            #         axman_wavefunction = ion.mesh.anim.SphericalHarmonicPhiSliceMeshAxis(),
            #         length = 10,
            #         target_dir = OUT_DIR,
            #     )
            # ],
        ).to_sim()

        sim.run(progress_bar=True)

        # for n, ds in sim.datastores.items():
        #     print(n, ds)

        # print(sim.data.norm)
        # print(sim.data.initial_state_overlap)
        # print(sim.data.z_expectation_value)
        # print(sim.data.r_expectation_value)

        # print(sim.data.norm)
        # print(sim.data.inner_products)
        # print(sim.data.state_overlaps)
        # print(sim.data.bogus)

        plot_fields(sim)
        plot_norm_and_inner_products(sim)
        plot_r_and_z_expectation_values(sim)
        plot_energy_expectation_values(sim)
        plot_norm_within_radius(sim)
