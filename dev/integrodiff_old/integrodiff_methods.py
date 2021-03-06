import logging
import os
import itertools

import numpy as np

import simulacra as si
from simulacra.units import *
import ionization as ion
import ide as ide

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

logman = si.utils.LogManager(
    "simulacra", "ionization", stdout_logs=True, stdout_level=logging.DEBUG
)

PLOT_KWARGS = (
    dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=3),
    dict(target_dir=OUT_DIR),
)


def run(spec):
    with logman as logger:
        sim = spec.to_sim()

        logger.debug(sim.info())
        sim.run()
        logger.debug(sim.info())

        for kwargs in PLOT_KWARGS:
            sim.plot_b2_vs_time(
                name_postfix=f"{sim.spec.time_step / asec:3f}", **kwargs
            )

        return sim


if __name__ == "__main__":
    with logman as logger:
        # gauge = 'LEN'
        # dt = 1 * asec

        for gauge in ["LEN", "VEL"]:
            for dt in [1 * asec, 0.1 * asec]:
                pw = 100 * asec
                flu = 0.1 * Jcm2

                # t_bound = 10
                # electric_pot = ion.potentials.SincPulse(pulse_width = pw, fluence = flu,
                #                              window = ion.potentials.LogisticWindow((t_bound - 2) * pw, .2 * pw))

                t_bound = 3
                electric_pot = ion.potentials.Rectangle(
                    start_time=-5 * pw,
                    end_time=5 * pw,
                    amplitude=1 * atomic_electric_field,
                    window=ion.potentials.LogisticWindow(
                        pw / 2, 0.1 * pw, window_center=-1 * pw
                    ),
                )
                electric_pot += ion.potentials.Rectangle(
                    start_time=0 * pw,
                    end_time=5 * pw,
                    amplitude=-1 * atomic_electric_field,
                    window=ion.potentials.LogisticWindow(
                        pw / 2, 0.1 * pw, window_center=1 * pw
                    ),
                )

                q = electron_charge
                m = electron_mass_reduced
                L = bohr_radius

                int_methods = ["simpson"]
                # int_methods = ['trapezoid', 'simpson']
                # evol_methods = ['ARK4', ]
                evol_methods = ["FE", "BE", "TRAP", "RK4"]
                # evol_methods = ['BE']
                # evol_methods = ['FE', 'BE']

                shared_kwargs = dict(
                    time_initial=-t_bound * pw,
                    time_final=t_bound * pw,
                    time_step=dt,
                    electric_potential=electric_pot,
                    test_energy=-(hbar ** 2) / (2 * m * (L ** 2)),
                )
                if gauge == "LEN":
                    spec_kwargs = dict(
                        prefactor=-np.sqrt(pi) * (L ** 2) * ((q / hbar) ** 2),
                        evolution_gauge="LEN",
                        kernel=ide.gaussian_kernel_LEN,
                        kernel_kwargs=dict(tau_alpha=4 * m * (L ** 2) / hbar),
                        **shared_kwargs,
                    )
                elif gauge == "VEL":
                    spec_kwargs = dict(
                        prefactor=-((q / m) ** 2) / (4 * (L ** 2)),
                        evolution_gauge="VEL",
                        kernel=ide.gaussian_kernel_VEL,
                        kernel_kwargs=dict(tau_alpha=2 * m * (L ** 2) / hbar, width=L),
                        **shared_kwargs,
                    )

                specs = []

                for evol_method, int_method in itertools.product(
                    evol_methods, int_methods
                ):
                    specs.append(
                        ide.IntegroDifferentialEquationSpecification(
                            f"{gauge}_{evol_method}",
                            # 'int={}_evol={}'.format(int_method, evol_method),
                            evolution_method=evol_method,
                            integration_method=int_method,
                            **spec_kwargs,
                        )
                    )

                # for int_method, eps in itertools.product(int_methods, [1e-3]):
                #     # for int_method, eps in itertools.product(int_methods, [1e-3, 1e-6, 1e-9]):
                #     specs.append(ide.IntegroDifferentialEquationSpecification(
                #         f'{gauge}_ARK4_eps={eps}',
                #         evolution_method = 'ARK4',
                #         integration_method = int_method,
                #         epsilon = eps,
                #         time_step_minimum = .1 * asec,
                #         **spec_kwargs
                #     ))

                # results = si.utils.multi_map(run, specs)
                results = si.utils.multi_map(run, specs, processes=3)

                for kwargs in PLOT_KWARGS:
                    si.vis.xxyy_plot(
                        f"{gauge}__method_comparison__pw={pw / asec:3f}_dt={dt / asec:3f}",
                        (r.times for r in results),
                        (r.b2 for r in results),
                        line_labels=(r.name for r in results),
                        x_label=r"Time $t$",
                        x_unit="asec",
                        x_lower_limit=-t_bound * pw,
                        x_upper_limit=t_bound * pw,
                        y_label=r"$\left| a(t) \right|^2$",
                        y_lower_limit=0,
                        y_upper_limit=1,
                        title=fr"Method Comparison at $\tau = {pw / asec:3f}$ as, $\Delta t = {dt / asec:3f}$ as",
                        **kwargs,
                    )

                    for r in results:
                        print(r.info())

                    print()

                    # print(f'gauge: {gauge}, dt = {dt / asec:3f} as')
                    just = max(len(r.name) for r in results) + 1
                    with open(
                        os.path.join(OUT_DIR, f"gauge={gauge}_dt={dt / asec:3f}as.txt"),
                        mode="w",
                    ) as file:
                        for r in results:
                            print(
                                r.name.rjust(just),
                                str(np.abs(r.a[-1]) ** 2).ljust(15),
                                str(len(r.times)).rjust(5),
                                str(r.running_time.total_seconds()),
                                file=file,
                            )
