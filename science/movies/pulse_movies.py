import logging
import os
from copy import deepcopy
import itertools

import numpy as np

import simulacra as si
from simulacra.units import *

import ionization as ion

import matplotlib.pyplot as plt

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

logman = si.utils.LogManager("simulacra", "ionization", stdout_level=logging.INFO)


def run(spec):
    with logman as logger:
        sim = spec.to_sim()

        sim.info().log()
        sim.run()
        sim.info().log()


if __name__ == "__main__":
    with logman as logger:
        with si.utils.BlockTimer() as timer:
            r_bound = 200
            points_per_r = 8
            r_points = r_bound * points_per_r
            l_bound = 500
            dt = 0.5
            t_bound = 15
            n_max = 3

            shading = "flat"

            initial_states = [ion.HydrogenBoundState(1, 0)]

            pulse_types = [ion.potentials.SincPulse]
            pulse_widths = [93, 200]
            fluences = [1, 5, 10]
            phases = np.linspace(0, pi / 2, 3)

            # used by all sims
            mask = ion.RadialCosineMask(
                inner_radius=(r_bound - 25) * bohr_radius,
                outer_radius=r_bound * bohr_radius,
            )
            out_dir_mod = os.path.join(
                OUT_DIR,
                f"R={r_bound}br_PPR={points_per_r}_L={l_bound}_dt={dt}as_T={t_bound}pw",
            )

            # used by all animators
            animator_kwargs = dict(
                target_dir=out_dir_mod, fig_dpi_scale=1, length=30, fps=30
            )

            axman_lower_right = animation.animators.ElectricPotentialPlotAxis(
                show_electric_field=True,
                show_vector_potential=False,
                show_y_label=False,
                show_ticks_right=True,
                legend_kwargs={"fontsize": 30},
            )

            specs = []
            for (
                initial_state,
                pulse_type,
                pulse_width,
                fluence,
                phase,
            ) in itertools.product(
                initial_states, pulse_types, pulse_widths, fluences, phases
            ):
                name = f"{pulse_type.__name__}__{initial_state.n}_{initial_state.l}__pw={pulse_width}as_flu={fluence}jcm2_cep={phase / pi:3f}pi"

                pw = pulse_width * asec
                flu = fluence * Jcm2

                efield = pulse_type.from_omega_min(
                    pulse_width=pw,
                    fluence=flu,
                    phase=phase,
                    window=ion.potentials.LogisticWindow(
                        window_time=(t_bound - 2) * pw, window_width=0.2 * pw
                    ),
                )

                wavefunction_axman = animation.animators.WavefunctionStackplotAxis(
                    states=[initial_state],
                    legend_kwargs={"fontsize": 30, "borderaxespad": 0.15},
                )

                animators = [
                    animation.animators.PolarAnimator(
                        postfix="__g_wavefunction__zoom",
                        axman_wavefunction=animation.animators.SphericalHarmonicPhiSliceMeshAxis(
                            which="g",
                            colormap=plt.get_cmap("richardson"),
                            norm=si.vis.RichardsonNormalization(),
                            plot_limit=30 * bohr_radius,
                            shading=shading,
                        ),
                        axman_lower_right=deepcopy(axman_lower_right),
                        axman_upper_right=animation.animators.WavefunctionStackplotAxis(
                            states=[initial_state]
                        ),
                        axman_colorbar=None,
                        **animator_kwargs,
                    ),
                    animation.animators.PolarAnimator(
                        postfix="__g2_wavefunction__zoom",
                        axman_wavefunction=animation.animators.SphericalHarmonicPhiSliceMeshAxis(
                            which="g2", plot_limit=30 * bohr_radius, shading=shading
                        ),
                        axman_lower_right=deepcopy(axman_lower_right),
                        axman_upper_right=animation.animators.WavefunctionStackplotAxis(
                            states=[initial_state]
                        ),
                        **animator_kwargs,
                    ),
                    animation.animators.PolarAnimator(
                        postfix="__g_wavefunction",
                        axman_wavefunction=animation.animators.SphericalHarmonicPhiSliceMeshAxis(
                            which="g",
                            colormap=plt.get_cmap("richardson"),
                            norm=si.vis.RichardsonNormalization(),
                            shading=shading,
                        ),
                        axman_lower_right=deepcopy(axman_lower_right),
                        axman_upper_right=animation.animators.WavefunctionStackplotAxis(
                            states=[initial_state]
                        ),
                        axman_colorbar=None,
                        **animator_kwargs,
                    ),
                    animation.animators.PolarAnimator(
                        postfix="__g2_wavefunction",
                        axman_wavefunction=animation.animators.SphericalHarmonicPhiSliceMeshAxis(
                            which="g2", shading=shading
                        ),
                        axman_lower_right=deepcopy(axman_lower_right),
                        axman_upper_right=animation.animators.WavefunctionStackplotAxis(
                            states=[initial_state]
                        ),
                        **animator_kwargs,
                    ),
                ]

                specs.append(
                    ion.SphericalHarmonicSpecification(
                        name,
                        electric_potential=efield,
                        time_initial=-t_bound * pw,
                        time_final=t_bound * pw,
                        time_step=dt * asec,
                        r_points=r_points,
                        r_bound=r_bound * bohr_radius,
                        l_bound=l_bound,
                        initial_state=initial_state,
                        mask=mask,
                        use_numeric_eigenstates=True,
                        numeric_eigenstate_max_energy=50 * eV,
                        numeric_eigenstate_max_angular_momentum=20,
                        animators=deepcopy(animators),
                        electric_potential_dc_correction=True,
                        store_data_every=5,
                    )
                )

            si.utils.multi_map(run, specs, processes=4)

        print(timer)
