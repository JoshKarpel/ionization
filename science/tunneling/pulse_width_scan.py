import logging
import os

import numpy as np

import simulacra as si
import simulacra.units as u

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), "out", FILE_NAME)

LOGMAN = si.utils.LogManager("simulacra", "ionization", stdout_level=logging.INFO)

PLOT_KWARGS = dict(target_dir=OUT_DIR, img_format="png", fig_dpi_scale=6)


def identifier(pulse):
    return f"pw={pulse.pulse_width / u.asec:.3f}as_flu={pulse.fluence / u.Jcm2:.3f}jcm2_cep={pulse.phase / u.pi:.3f}pi"


def run(spec):
    with LOGMAN as logger:
        sim = spec.to_simulation()

        sim.run()

        return sim


if __name__ == "__main__":
    with LOGMAN as logger:
        tb = 35

        pws = np.array([700, 800, 900, 1000]) * u.asec
        fluences = np.array([0.1, 0.5, 2]) * u.Jcm2
        ceps = np.linspace(0, u.pi, 100)

        for pw in pws:
            results = {}

            for flu in fluences:
                pulses = [
                    ion.potentials.SincPulse(pulse_width=pw, fluence=flu, phase=cep)
                    for cep in ceps
                ]

                specs = [
                    tunneling.TunnelingSpecification(
                        identifier(pulse),
                        time_initial=-tb * pulse.pulse_width,
                        time_final=tb * pulse.pulse_width,
                        time_step=10 * u.asec,
                        electric_potential=pulse,
                        electric_potential_dc_correction=True,
                        tunneling_model=tunneling.LandauRate(),
                    )
                    for pulse in pulses
                ]

                results[flu] = si.utils.multi_map(run, specs, processes=2)

                si.vis.xy_plot(
                    f"ionization_vs_cep__pw={pw / u.asec:.3f}as_flu={flu / u.Jcm2:.3f}jcm2",
                    ceps,
                    [r.b2[-1] for r in results[flu]],
                    x_label=r"$ \varphi $",
                    x_unit="rad",
                    y_label=r"$ \left| b_{\varphi}(t_f) \right|^2 $",
                    **PLOT_KWARGS,
                )

            si.vis.xy_plot(
                f"ionization_vs_cep__pw={pw / u.asec:.3f}as",
                ceps,
                *[
                    [r.b2[-1] / results[flu][0].b2[-1] for r in results[flu]]
                    for flu in fluences
                ],
                line_labels=[
                    rf"$ {flu / u.Jcm2:.3f} \mathrm{{J/cm^2}} $" for flu in fluences
                ],
                x_label=r"$ \varphi $",
                x_unit="rad",
                y_label=r"$ \left| b_{\varphi}(t_f) \right|^2 \; / \; \left| b_{\varphi = 0}(t_f) \right|^2 $",
                **PLOT_KWARGS,
            )
