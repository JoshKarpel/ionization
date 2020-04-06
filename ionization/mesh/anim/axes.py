import logging
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np

import simulacra as si
import simulacra.units as u

from ... import vis, states

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

COLORMESH_GRID_KWARGS = {
    **si.vis.DEFAULT_COLORMESH_GRID_KWARGS,
    **dict(linestyle=":", linewidth=1.5, alpha=0.6),
}


class ElectricPotentialPlotAxis(si.vis.AxisManager):
    def __init__(
        self,
        time_unit: u.Unit = "asec",
        show_electric_field=True,
        electric_field_unit="atomic_electric_field",
        show_vector_potential=False,
        vector_potential_unit="atomic_momentum",
        linewidth=3,
        show_y_label=False,
        show_ticks_bottom=True,
        show_ticks_top=False,
        show_ticks_right=True,
        show_ticks_left=True,
        grid_kwargs=None,
        legend_kwargs=None,
    ):
        self.show_electric_field = show_electric_field
        self.show_vector_potential = show_vector_potential

        if not show_electric_field and not show_vector_potential:
            logger.warning(
                f"{self} has both show_electric_field and show_vector_potential set to False"
            )

        self.time_unit = time_unit
        self.time_unit_value, self.time_unit_latex = u.get_unit_value_and_latex(
            time_unit
        )
        self.electric_field_unit = electric_field_unit
        (
            self.electric_field_unit_value,
            self.electric_field_unit_latex,
        ) = u.get_unit_value_and_latex(electric_field_unit)
        self.vector_potential_unit = vector_potential_unit
        (
            self.vector_potential_unit_value,
            self.vector_potential_unit_latex,
        ) = u.get_unit_value_and_latex(vector_potential_unit)

        self.show_y_label = show_y_label
        self.show_ticks_bottom = show_ticks_bottom
        self.show_ticks_top = show_ticks_top
        self.show_ticks_right = show_ticks_right
        self.show_ticks_left = show_ticks_left

        self.linewidth = linewidth

        if legend_kwargs is None:
            legend_kwargs = dict()
        legend_defaults = dict(
            loc="lower left", fontsize=30, fancybox=True, framealpha=0
        )
        self.legend_kwargs = {**legend_defaults, **legend_kwargs}

        if grid_kwargs is None:
            grid_kwargs = {}
        self.grid_kwargs = {**si.vis.DEFAULT_GRID_KWARGS, **grid_kwargs}

        super().__init__()

    def initialize_axis(self):
        self.time_line = self.axis.axvline(
            x=self.sim.time / self.time_unit_value, color="gray", animated=True
        )
        self.redraw.append(self.time_line)

        if self.show_electric_field:
            (self.electric_field_line,) = self.axis.plot(
                self.sim.data_times / self.time_unit_value,
                self.sim.data.electric_field_amplitude / self.electric_field_unit_value,
                label=fr"${vis.LATEX_EFIELD}(t)$",
                color=vis.COLOR_EFIELD,
                linewidth=self.linewidth,
                animated=True,
            )

            self.redraw.append(self.electric_field_line)

        if self.show_vector_potential:
            (self.vector_potential_line,) = self.axis.plot(
                self.sim.data_times / self.time_unit_value,
                u.proton_charge
                * self.sim.data.vector_potential_amplitude
                / self.vector_potential_unit_value,
                label=fr"$q \, {vis.LATEX_AFIELD}(t)$",
                color=vis.COLOR_AFIELD,
                linewidth=self.linewidth,
                linestyle="--",
                animated=True,
            )

            self.redraw.append(self.vector_potential_line)

        self.legend = self.axis.legend(**self.legend_kwargs)
        self.redraw.append(self.legend)

        self.axis.grid(True, **self.grid_kwargs)

        self.axis.set_xlabel(fr"Time $t$ (${self.time_unit_latex}$)", fontsize=26)

        if self.show_y_label:
            self.axis.set_ylabel("Wavefunction Metric", fontsize=26)

        self.axis.tick_params(
            labelbottom=self.show_ticks_bottom,
            labeltop=self.show_ticks_top,
            labelright=self.show_ticks_right,
            labelleft=self.show_ticks_left,
        )

        self.axis.set_xlim(
            self.sim.data_times[0] / self.time_unit_value,
            self.sim.data_times[-1] / self.time_unit_value,
        )

        data = []
        if self.show_electric_field:
            data.append(
                self.spec.electric_potential.get_electric_field_amplitude(
                    self.sim.times
                )
                / self.electric_field_unit_value
            )
        if self.show_vector_potential:
            data.append(
                u.proton_charge
                * self.spec.electric_potential.get_vector_potential_amplitude_numeric_cumulative(
                    self.sim.times
                )
                / self.vector_potential_unit_value
            )

        y_lower_limit, y_upper_limit = si.vis.set_axis_limits_and_scale(
            self.axis, *data, pad=0.05, direction="y"
        )

        self.axis.tick_params(axis="both", which="major", labelsize=16)

        self.redraw += [
            *self.axis.xaxis.get_gridlines(),
            *self.axis.yaxis.get_gridlines(),
        ]

        super().initialize_axis()

    def update_axis(self):
        if self.show_electric_field:
            self.electric_field_line.set_ydata(
                self.sim.data.electric_field_amplitude / self.electric_field_unit_value
            )
        if self.show_vector_potential:
            self.vector_potential_line.set_ydata(
                u.proton_charge
                * self.sim.data.vector_potential_amplitude
                / self.vector_potential_unit_value
            )

        self.time_line.set_xdata(self.sim.time / self.time_unit_value)

        super().update_axis()

    def info(self) -> si.Info:
        info = super().info()

        info.add_field("Show Electric Field", self.show_electric_field)
        if self.show_electric_field:
            info.add_field("Electric Field Unit", self.electric_field_unit)
        info.add_field("Show Vector Potential", self.show_vector_potential)
        if self.show_vector_potential:
            info.add_field("Vector Potential Unit", self.vector_potential_unit)
        info.add_field("Time Unit", self.time_unit)

        return info


class StackplotAxis(si.vis.AxisManager):
    def __init__(
        self,
        show_norm=True,
        time_unit: u.Unit = "asec",
        y_label=None,
        show_ticks_bottom=True,
        show_ticks_top=False,
        show_ticks_right=True,
        show_ticks_left=True,
        grid_kwargs=None,
        legend_kwargs=None,
    ):
        self.show_norm = show_norm

        self.time_unit = time_unit
        self.time_unit_value, self.time_unit_latex = u.get_unit_value_and_latex(
            time_unit
        )

        self.y_label = y_label
        self.show_ticks_bottom = show_ticks_bottom
        self.show_ticks_top = show_ticks_top
        self.show_ticks_right = show_ticks_right
        self.show_ticks_left = show_ticks_left

        if legend_kwargs is None:
            legend_kwargs = {}
        legend_defaults = dict(
            loc="lower left", fontsize=30, fancybox=True, framealpha=0
        )
        self.legend_kwargs = {**legend_defaults, **legend_kwargs}

        if grid_kwargs is None:
            grid_kwargs = {}
        self.grid_kwargs = {**si.vis.DEFAULT_GRID_KWARGS, **grid_kwargs}

        super().__init__()

    def initialize_axis(self):
        if self.show_norm:
            (self.norm_line,) = self.axis.plot(
                self.sim.data.times / self.time_unit_value,
                self.sim.data.norm,
                label=r"$\left\langle \Psi|\psi \right\rangle$",
                color="black",
                linewidth=3,
            )

            self.redraw.append(self.norm_line)

        self._initialize_stackplot()

        self.time_line = self.axis.axvline(
            x=self.sim.time / self.time_unit_value,
            color="gray",
            linewidth=1,
            animated=True,
        )
        self.redraw.append(self.time_line)

        self.legend = self.axis.legend(**self.legend_kwargs)
        self.redraw.append(self.legend)

        self.axis.grid(True, **self.grid_kwargs)

        self.axis.set_xlabel(fr"Time $t$ (${self.time_unit_latex}$)", fontsize=26)

        if self.y_label is not None:
            self.axis.set_ylabel(self.y_label, fontsize=26)

        self.axis.tick_params(
            labelbottom=self.show_ticks_bottom,
            labeltop=self.show_ticks_top,
            labelright=self.show_ticks_right,
            labelleft=self.show_ticks_left,
        )

        self.axis.set_xlim(
            self.sim.data_times[0] / self.time_unit_value,
            self.sim.data_times[-1] / self.time_unit_value,
        )
        self.axis.set_ylim(0, 1.05)

        self.axis.tick_params(axis="both", which="major", labelsize=16)

        self.redraw += [
            *self.axis.xaxis.get_gridlines(),
            *self.axis.yaxis.get_gridlines(),
        ]

        super().initialize_axis()

    def _get_stackplot_data_and_labels(self):
        raise NotImplementedError

    def _initialize_stackplot(self):
        data, labels = self._get_stackplot_data_and_labels()

        self.overlaps_stackplot = self.axis.stackplot(
            self.sim.data_times / self.time_unit_value,
            *data,
            labels=labels,
            animated=True,
        )

        self.redraw += [*self.overlaps_stackplot]

    def _update_stackplot_lines(self):
        for x in self.overlaps_stackplot:
            self.redraw.remove(x)
            x.remove()

        self.axis.set_prop_cycle(None)

        data, labels = self._get_stackplot_data_and_labels()

        self.overlaps_stackplot = self.axis.stackplot(
            self.sim.data_times / self.time_unit_value,
            *data,
            labels=labels,
            animated=True,
        )

        self.redraw = [*self.overlaps_stackplot] + self.redraw

    def update_axis(self):
        if self.show_norm:
            self.norm_line.set_ydata(self.sim.data.norm)

        self._update_stackplot_lines()

        self.time_line.set_xdata(self.sim.time / self.time_unit_value)

        super().update_axis()

    def info(self) -> si.Info:
        info = super().info()

        info.add_field("Show Norm", self.show_norm)
        info.add_field("Time Unit", self.time_unit)

        return info


class TestStateStackplotAxis(StackplotAxis):
    def __init__(self, states=None, **kwargs):
        self.states = states
        if not callable(self.states):
            self.states = tuple(sorted(states))
        if len(self.states) > 8:
            logger.warning(
                f"Using more than 8 states in a {self.__class__.__name__} is ill-advised"
            )

        super().__init__(**kwargs)

    def _get_stackplot_data_and_labels(self):
        state_overlaps = self.sim.state_overlaps_vs_time
        if self.states is not None:
            if callable(self.states):
                data = (
                    overlap
                    for state, overlap in sorted(state_overlaps.items())
                    if self.states(state)
                )
            else:
                states = set(self.states)
                data = (
                    overlap
                    for state, overlap in sorted(state_overlaps.items())
                    if state in states
                    or (state.numeric and state.analytic_state in states)
                )
        else:
            data = (overlap for state, overlap in sorted(state_overlaps.items()))

        labels = (
            r"$\left| \left\langle \Psi| {} \right\rangle \right|^2$".format(state.tex)
            for state in sorted(state_overlaps)
        )

        return data, labels


class WavefunctionStackplotAxis(StackplotAxis):
    def __init__(
        self, states: Optional[Iterable[states.QuantumState]] = None, **kwargs
    ):
        if states is None:
            states = ()
        self.states = sorted(states)

        super().__init__(**kwargs)

    def _get_stackplot_data_and_labels(self):
        state_overlaps = self.sim.data.state_overlaps

        selected_state_overlaps = {
            state: overlap
            for state, overlap in sorted(state_overlaps.items())
            if state in self.states
            or (state.numeric and state.analytic_state in self.states)
        }
        overlap_len = len(
            list(state_overlaps.values())[0]
        )  # ugly, but I don't see a way around it

        data = [
            *(overlap for state, overlap in sorted(selected_state_overlaps.items())),
            sum(
                (
                    overlap
                    for state, overlap in state_overlaps.items()
                    if state.bound
                    and (
                        state not in self.states
                        and (
                            not state.numeric or state.analytic_state not in self.states
                        )
                    )
                ),
                np.zeros(overlap_len),
            ),
            sum(
                (
                    overlap
                    for state, overlap in state_overlaps.items()
                    if state.free
                    and (
                        state not in self.states
                        and (
                            not state.numeric or state.analytic_state not in self.states
                        )
                    )
                ),
                np.zeros(overlap_len),
            ),
        ]

        labels = (
            *(
                r"$ \left| \left\langle \Psi | {} \right\rangle \right|^2 $".format(
                    state.tex
                )
                for state, overlap in sorted(selected_state_overlaps.items())
            ),
            r"$ \sum_{\mathrm{other \, bound}} \; \left| \left\langle \Psi | \psi_{{n, \, \ell}} \right\rangle \right|^2 $",
            fr"$ \sum_{{ \mathrm{{other \, free}} }} \; \left| \left\langle \Psi | \phi_{{E, \, \ell}} \right\rangle \right|^2 $",
        )

        return data, labels


class AngularMomentumDecompositionAxis(si.vis.AxisManager):
    def __init__(
        self, renormalize_l_decomposition: bool = False, maximum_l: Optional[int] = None
    ):
        self.renormalize_l_decomposition = renormalize_l_decomposition
        self.maximum_l = maximum_l
        self.slice = slice(self.maximum_l + 1)

        super().__init__()

    def initialize_axis(self):
        l_plot = self.sim.mesh.norm_by_l
        if self.renormalize_l_decomposition:
            l_plot /= self.sim.mesh.norm()
        self.ang_mom_bar = self.axis.bar(
            self.sim.mesh.l[self.slice],
            l_plot[self.slice],
            align="center",
            color=".5",
            animated=True,
        )

        self.redraw += [*self.ang_mom_bar]

        self.axis.yaxis.grid(True, **si.vis.DEFAULT_GRID_KWARGS)

        self.axis.set_xlabel(r"Orbital Angular Momentum $\ell$", fontsize=22)
        l_label = r"$\left| \left\langle \Psi | Y^{\ell}_0 \right\rangle \right|^2$"
        if self.renormalize_l_decomposition:
            l_label += r"$/\left\langle\Psi|\Psi\right\rangle$"
        self.axis.set_ylabel(l_label, fontsize=22)
        self.axis.yaxis.set_label_position("right")

        self.axis.set_ylim(0, 1)
        self.axis.set_xlim(
            np.min(self.sim.mesh.l[self.slice]) - 0.4,
            np.max(self.sim.mesh.l[self.slice]) + 0.4,
        )

        self.axis.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        self.axis.tick_params(labelleft=False, labelright=True)
        self.axis.tick_params(axis="both", which="major", labelsize=20)

        self.redraw += [*self.axis.yaxis.get_gridlines()]

        super().initialize_axis()

    def update_axis(self):
        l_plot = self.sim.mesh.norm_by_l[self.slice]
        if self.renormalize_l_decomposition:
            l_plot /= self.sim.norm_vs_time[self.sim.data_time_index]
        for bar, height in zip(self.ang_mom_bar, l_plot):
            bar.set_height(height)

        super().update_axis()


class ColorBarAxis(si.vis.AxisManager):
    def assign_colorable(self, colorable, fontsize=14):
        self.colorable = colorable
        self.fontsize = fontsize

    def initialize_axis(self):
        self.cbar = plt.colorbar(mappable=self.colorable, cax=self.axis)
        self.cbar.ax.tick_params(labelsize=self.fontsize)

        super().initialize_axis()
