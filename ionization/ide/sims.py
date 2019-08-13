import logging
import datetime
import itertools
from typing import Union, Callable

from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integ
import scipy.interpolate as interp

import simulacra as si
import simulacra.units as u

from .. import core, potentials, tunneling, vis, utils

from . import kernels, evolution_methods

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class IntegroDifferentialEquationSimulation(si.Simulation):
    """
    A class that encapsulates a simulation of an IDE with the form
    db/dt = prefactor * f(t) * integral[ f(t') * b(t') * kernel(t, t')
    which happen to arise when working with the IDE model.

    Attributes
    ----------
    time : :class:`float`
        The current simulation time.
    time_steps : :class:`int`
        The current number of time steps that have been solved for.
    b
        The solution of the IDE vs time (i.e., the bound state probability amplitude in the rotating frame of the bound state).
    b2
        The square of the absolute value of the solution vs time (i.e., the probability that the system is in the bound state).
    """

    def __init__(self, spec: si.Specification):
        super().__init__(spec)

        self.latest_checkpoint_time = datetime.datetime.utcnow()

        self.times = [self.spec.time_initial]
        self.time_index = 0
        self.time_step = self.spec.time_step

        self.b = [self.spec.b_initial]

        time_range = self.spec.time_final - self.spec.time_initial
        dummy_times = np.linspace(
            self.spec.time_initial,
            self.spec.time_final,
            int(time_range / self.time_step),
        )
        if self.spec.electric_potential_dc_correction:
            old_pot = self.spec.electric_potential
            self.spec.electric_potential = potentials.DC_correct_electric_potential(
                self.spec.electric_potential, dummy_times
            )

            logger.warning(
                f"DC-corrected electric potential {old_pot} --> {self.spec.electric_potential} for {self}"
            )

        if self.spec.electric_potential_fluence_correction:
            old_pot = self.spec.electric_potential
            self.spec.electric_potential = potentials.FluenceCorrector(
                electric_potential=self.spec.electric_potential,
                times=dummy_times,
                target_fluence=list(self.spec.electric_potential)[
                    0
                ].fluence,  # the analytic fluence of the embedded pulse, whether it's been dc-corrected or not
            )

            logger.warning(
                f"Fluence-corrected electric potential {old_pot} --> {self.spec.electric_potential} for {self}"
            )

        self.interpolated_vector_potential = interp.CubicSpline(
            x=dummy_times,
            y=self.spec.electric_potential.get_vector_potential_amplitude_numeric_cumulative(
                dummy_times
            ),
            bc_type="natural",
        )

        if self.spec.integration_method == "simpson":
            self.integrate = integ.simps
        elif self.spec.integration_method == "trapezoid":
            self.integrate = integ.trapz

        if self.spec.evolution_gauge == core.Gauge.LENGTH:
            self.f = self.spec.electric_potential.get_electric_field_amplitude
        elif self.spec.evolution_gauge == core.Gauge.VELOCITY:
            self.f = (
                self.spec.electric_potential.get_vector_potential_amplitude_numeric_cumulative
            )

    @property
    def time_steps(self):
        return len(self.times)

    @property
    def time(self):
        return self.times[self.time_index]

    @property
    def data_times(self):
        return self.times

    @property
    def b2(self):
        return np.abs(self.b) ** 2

    def evaluate_kernel(self, time_current, time_previous):
        """
        Calculate the values of the IDE kernel as a function of the time difference.

        Parameters
        ----------
        time_current
        time_previous

        Returns
        -------
        kernel :
            The value of the kernel at the time differences.
        """
        return self.spec.kernel(
            time_current,
            time_previous,
            self.spec.electric_potential,
            self.interpolated_vector_potential,
        )

    def run(
        self,
        progress_bar: bool = False,
        callback: Callable[["IntegroDifferentialEquationSimulation"], None] = None,
    ):
        """
        Run the IDE simulation by repeatedly evolving it forward in time.


        Parameters
        ----------
        callback : callable
            A function that accepts the ``Simulation`` as an argument, called at the end of each time step.
        """
        logger.info(
            f"Performing time evolution on {self.name} ({self.file_name}), starting from time index {self.time_index}"
        )
        self.status = si.Status.RUNNING

        if progress_bar:
            num_asecs_remaining = int((self.spec.time_final - self.times[-1]) / u.asec)
            current_asecs_remaining = num_asecs_remaining
            pbar = tqdm(total=num_asecs_remaining, ascii=True, ncols=80)

        while self.time < self.spec.time_final:
            new_b, new_t = self.spec.evolution_method.evolve(
                self, self.b, self.times, self.time_step
            )
            dt = new_t - self.time

            efield = self.spec.electric_potential.get_electric_field_amplitude(
                self.time + (dt / 2)
            )
            tunneling_rate = self.spec.tunneling_model.tunneling_rate(
                efield, self.spec.ionization_potential
            )
            new_b *= np.exp(dt * tunneling_rate)

            self.b.append(new_b)
            self.times.append(new_t)
            self.time_index += 1

            logger.debug(
                f"{self} evolved to time index {self.time_index} ({round(self.completion_percent)}%)"
            )

            if callback is not None:
                callback(self)

            if self.spec.checkpoints:
                now = datetime.datetime.utcnow()
                if (now - self.latest_checkpoint_time) > self.spec.checkpoint_every:
                    self.do_checkpoint(now)

            try:
                new_asecs_remaining = int(
                    (self.spec.time_final - self.times[-1]) / u.asec
                )
                pbar.update(current_asecs_remaining - new_asecs_remaining)
                current_asecs_remaining = new_asecs_remaining
            except NameError:
                pass

        try:
            pbar.close()
        except NameError:
            pass

        self.b = np.array(self.b)
        self.times = np.array(self.times)

        time_indices = np.array(range(0, len(self.times)))
        self.data_mask = np.equal(time_indices, 0) + np.equal(
            time_indices, self.time_steps - 1
        )
        if self.spec.store_data_every >= 1:
            self.data_mask += np.equal(time_indices % self.spec.store_data_every, 0)

        self.times = self.times[self.data_mask]
        self.b = self.b[self.data_mask]

        self.status = si.Status.FINISHED
        logger.info(
            f"Finished performing time evolution on {self.name} ({self.file_name})"
        )

    def do_checkpoint(self, now):
        self.status = si.Status.PAUSED
        self.save(target_dir=self.spec.checkpoint_dir)
        self.latest_checkpoint_time = now
        logger.info(
            f"Checkpointed {self} at time index {self.time_index} ({round(self.completion_percent)}%)"
        )
        self.status = si.Status.RUNNING

    @property
    def completion_percent(self):
        return (
            100
            * (self.time - self.spec.time_initial)
            / (self.spec.time_final - self.spec.time_initial)
        )

    def attach_electric_potential_plot_to_axis(
        self,
        axis,
        time_unit="asec",
        legend_kwargs=None,
        show_y_label: bool = False,
        show_electric_field: bool = True,
        show_vector_potential: bool = True,
    ):
        time_unit_value, time_unit_latex = u.get_unit_value_and_latex_from_unit(
            time_unit
        )

        if legend_kwargs is None:
            legend_kwargs = dict()
        legend_defaults = dict(
            loc="lower left", fontsize=10, fancybox=True, framealpha=0.3
        )
        legend_kwargs = {**legend_defaults, **legend_kwargs}

        y_labels = []
        if show_electric_field:
            e_label = fr"$ {vis.LATEX_EFIELD}(t) $"
            axis.plot(
                self.times / time_unit_value,
                self.spec.electric_potential.get_electric_field_amplitude(self.times)
                / u.atomic_electric_field,
                color=vis.COLOR_EFIELD,
                linewidth=1.5,
                label=e_label,
            )
            y_labels.append(e_label)
        if show_vector_potential:
            a_label = fr"$ e \, {vis.LATEX_AFIELD}(t) $"
            axis.plot(
                self.times / time_unit_value,
                u.proton_charge
                * self.spec.electric_potential.get_vector_potential_amplitude_numeric_cumulative(
                    self.times
                )
                / u.atomic_momentum,
                color=vis.COLOR_AFIELD,
                linewidth=1.5,
                label=a_label,
            )
            y_labels.append(a_label)

        if show_y_label:
            axis.set_ylabel(", ".join(y_labels), fontsize=13)

        axis.set_xlabel("Time $t$ (${}$)".format(time_unit_latex), fontsize=13)

        axis.tick_params(labelright=True)

        axis.set_xlim(self.times[0] / time_unit_value, self.times[-1] / time_unit_value)

        axis.legend(**legend_kwargs)

        axis.grid(True, **si.vis.GRID_KWARGS)

    def plot_wavefunction_vs_time(self, *args, **kwargs):
        """Alias for plot_b2_vs_time."""
        self.plot_b2_vs_time(*args, **kwargs)

    def plot_b2_vs_time(
        self,
        log=False,
        time_unit="asec",
        show_vector_potential=False,
        show_title=False,
        **kwargs,
    ):
        with si.vis.FigureManager(self.file_name + "__b2_vs_time", **kwargs) as figman:
            fig = figman.fig

            t_scale_unit, t_scale_name = u.get_unit_value_and_latex_from_unit(time_unit)

            grid_spec = matplotlib.gridspec.GridSpec(
                2, 1, height_ratios=[4, 1], hspace=0.07
            )
            ax_b2 = plt.subplot(grid_spec[0])
            ax_pot = plt.subplot(grid_spec[1], sharex=ax_b2)

            self.attach_electric_potential_plot_to_axis(
                ax_pot, show_vector_potential=show_vector_potential, time_unit=time_unit
            )

            ax_b2.plot(self.times / t_scale_unit, self.b2, color="black", linewidth=2)

            if log:
                ax_b2.set_yscale("log")
                min_overlap = np.min(self.b2)
                ax_b2.set_ylim(bottom=max(1e-9, min_overlap * 0.1), top=1.0)
                ax_b2.grid(True, which="both", **si.vis.GRID_KWARGS)
            else:
                ax_b2.set_ylim(0.0, 1.0)
                ax_b2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                ax_b2.grid(True, **si.vis.GRID_KWARGS)

            ax_b2.set_xlim(
                self.spec.time_initial / t_scale_unit,
                self.spec.time_final / t_scale_unit,
            )

            ax_b2.set_ylabel(r"$\left| b(t) \right|^2$", fontsize=13)

            # Find at most n+1 ticks on the y-axis at 'nice' locations
            max_yticks = 4
            yloc = plt.MaxNLocator(max_yticks, prune="upper")
            ax_pot.yaxis.set_major_locator(yloc)

            max_xticks = 6
            xloc = plt.MaxNLocator(max_xticks, prune="both")
            ax_pot.xaxis.set_major_locator(xloc)

            ax_pot.tick_params(axis="x", which="major", labelsize=10)
            ax_pot.tick_params(axis="y", which="major", labelsize=10)
            ax_b2.tick_params(axis="both", which="major", labelsize=10)

            ax_b2.tick_params(
                labelleft=True,
                labelright=True,
                labeltop=True,
                labelbottom=False,
                bottom=True,
                top=True,
                left=True,
                right=True,
            )
            ax_pot.tick_params(
                labelleft=True,
                labelright=True,
                labeltop=False,
                labelbottom=True,
                bottom=True,
                top=True,
                left=True,
                right=True,
            )

            if show_title:
                title = ax_b2.set_title(self.name)
                title.set_y(1.15)

            postfix = ""
            if log:
                postfix += "__log"

            figman.name += postfix


class IntegroDifferentialEquationSpecification(si.Specification):
    """
    A Specification for an :class:`IntegroDifferentialEquationSimulation`.
    """

    simulation_type = IntegroDifferentialEquationSimulation

    def __init__(
        self,
        name: str,
        time_initial: float = 0 * u.asec,
        time_final: float = 200 * u.asec,
        time_step: float = 1 * u.asec,
        test_mass: float = u.electron_mass,
        test_charge: float = u.electron_charge,
        b_initial: Union[float, complex] = 1,
        integral_prefactor: float = -((u.electron_charge / u.hbar) ** 2),
        electric_potential: potentials.PotentialEnergy = potentials.NoElectricPotential(),
        electric_potential_dc_correction: bool = False,
        electric_potential_fluence_correction: bool = False,
        kernel: kernels.Kernel = kernels.LengthGaugeHydrogenKernel(),
        integration_method: str = "simpson",
        evolution_method: evolution_methods.EvolutionMethod = evolution_methods.RungeKuttaFourMethod(),
        evolution_gauge: core.Gauge = core.Gauge.LENGTH,
        checkpoints: bool = False,
        checkpoint_every: datetime.timedelta = datetime.timedelta(hours=1),
        checkpoint_dir: str = None,
        store_data_every: int = 1,
        tunneling_model: tunneling.TunnelingModel = tunneling.NoTunneling(),
        ionization_potential=-u.rydberg,
        **kwargs,
    ):
        """

        Parameters
        ----------
        name
            The name of the specification/simulation.
        time_initial
            The time to begin the simulation at.
        time_final
            The time to end the simulation at.
        time_step
            The amount of time to evolve by on each evolution step.
        test_mass
            The mass of the test particle.
        test_charge
            The charge of the test particle.
        b_initial
            The initial bound state amplitude.
        integral_prefactor
            The prefactor of the integral term of the IDE.
        electric_potential
            The possibly-time varying external electric field.
        electric_potential_dc_correction
            If ``True``, perform DC correction on the ``electric_potential``.
        kernel
            The :class:`Kernel` to use for the simulation.
        integration_method : {'trapezoid', 'simpson'}
            The kind of integration rule to use.
        evolution_method
            The :class:`EvolutionMethod` to use.
        evolution_gauge
            The :class:`Gauge` to work in.
        checkpoints
            If ``True``, the simulation will save checkpoints to ``checkpoint_dir`` every ``checkpoint_every``.
        checkpoint_every
            The time between checkpoints.
        checkpoint_dir
            The directory to save checkpoints to.
        store_data_every
            Data will be stored every ``store_data_every`` time steps.
            The special value ``store_data_every = -1`` causes data to be stored only on the first and last time steps.
        tunneling_model
            A :class:`TunnelingModel` that will be applied to the wavefunction after every time step.
        ionization_potential
            The ionization pot
        kwargs
            Any additional keyword arguments are passed to the :class:`simulacra.Specification` constructor.
        """
        super().__init__(name, **kwargs)

        self.time_initial = time_initial
        self.time_final = time_final
        self.time_step = time_step

        self.test_mass = test_mass
        self.test_charge = test_charge

        self.b_initial = b_initial

        self.integral_prefactor = integral_prefactor

        self.electric_potential = electric_potential
        self.electric_potential_dc_correction = electric_potential_dc_correction
        self.electric_potential_fluence_correction = (
            electric_potential_fluence_correction
        )

        self.kernel = kernel

        self.integration_method = integration_method
        self.evolution_method = evolution_method
        self.evolution_gauge = evolution_gauge

        self.checkpoints = checkpoints
        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir = checkpoint_dir

        self.store_data_every = store_data_every

        self.tunneling_model = tunneling_model
        self.ionization_potential = ionization_potential

    def info(self) -> si.Info:
        info = super().info()

        info_checkpoint = si.Info(header="Checkpointing")
        if self.checkpoints:
            if self.checkpoint_dir is not None:
                working_in = self.checkpoint_dir
            else:
                working_in = "cwd"
            info_checkpoint.header += ": every {} time steps, working in {}".format(
                self.checkpoint_every, working_in
            )
        else:
            info_checkpoint.header += ": disabled"

        info.add_info(info_checkpoint)

        info_evolution = si.Info(header="Time Evolution")
        info_evolution.add_field(
            "Initial Time", utils.fmt_quantity(self.time_initial, utils.TIME_UNITS)
        )
        info_evolution.add_field(
            "Final Time", utils.fmt_quantity(self.time_final, utils.TIME_UNITS)
        )
        info_evolution.add_field(
            "Time Step", utils.fmt_quantity(self.time_step, utils.TIME_UNITS)
        )
        info.add_info(info_evolution)

        info_algorithm = si.Info(header="Evolution Algorithm")
        info_algorithm.add_field("Integration Method", self.integration_method)
        info_algorithm.add_info(self.evolution_method.info())
        info_algorithm.add_field("Evolution Gauge", self.evolution_gauge)
        info.add_info(info_algorithm)

        info_ide = si.Info(header="IDE Parameters")
        info_ide.add_field("Initial State", f"b = {self.b_initial}")
        info_ide.add_field("Prefactor", self.integral_prefactor)
        info.add_info(info_ide)

        info.add_info(self.kernel.info())
        info.add_info(self.tunneling_model.info())

        info_fields = si.Info(header="Electric Fields")
        info_fields.add_field(
            "DC Correct Electric Field",
            "yes" if self.electric_potential_dc_correction else "no",
        )
        for x in itertools.chain(self.electric_potential):
            info_fields.add_info(x.info())
        info.add_info(info_fields)

        return info
