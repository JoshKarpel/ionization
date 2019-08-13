import logging
import os

import simulacra as si
import simulacra.units as u

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

LOGMAN = si.utils.LogManager("simulacra", "ionization", stdout_level=logging.INFO)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)

if __name__ == "__main__":
    with LOGMAN as logger:
        amp = 0.05 * u.atomic_electric_field

        sim = tunneling.TunnelingSpecification(
            "test",
            tunneling_model=tunneling.models.LandauRate(),
            time_final=202 * u.fsec,
            time_step=0.01 * u.fsec,
            electric_potential=potentials.Rectangle(
                start_time=1 * u.fsec, end_time=201 * u.fsec, amplitude=amp
            ),
        ).to_sim()

        print(sim.info())

        sim.run(progress_bar=True)

        print(sim.b2)

        si.vis.xy_plot(
            "b_vs_t",
            sim.times,
            sim.b2,
            x_unit="fsec",
            vlines=[186 * u.fsec],
            hlines=[1 / u.e],
            **PLOT_KWARGS
        )

        print(sim.spec.tunneling_model._tunneling_rate(amp, -u.rydberg) / u.per_fsec)
