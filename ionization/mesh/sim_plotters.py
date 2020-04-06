import collections
import functools
import itertools
import logging
from copy import copy
from typing import Optional, Iterable

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
from cycler import cycler
import numpy as np

import simulacra as si
import simulacra.units as u

from .. import potentials, states, vis

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MeshSimulationPlotter:
    def __init__(self, sim):
        self.sim = sim
        self.spec = sim.spec

    def group_free_states_by_continuous_attr(
        self,
        attr="energy",
        divisions=10,
        cutoff_value=None,
        label_format_str=r"\phi_{{    {} \; \mathrm{{to}} \; {} \, {}, \ell   }}",
        attr_unit: u.Unit = "eV",
    ):
        spectrum = set(getattr(s, attr) for s in self.sim.free_states)

        grouped_states = collections.defaultdict(list)
        group_labels = {}

        try:
            attr_min, attr_max = min(spectrum), max(spectrum)
        except ValueError:
            return [], []

        if cutoff_value is None:
            boundaries = np.linspace(attr_min, attr_max, num=divisions + 1)
        else:
            boundaries = np.linspace(attr_min, cutoff_value, num=divisions)
            boundaries = np.concatenate((boundaries, [attr_max]))

        label_unit_value, label_unit_latex = u.get_unit_value_and_latex(attr_unit)

        free_states = list(self.sim.free_states)

        for ii, lower_boundary in enumerate(boundaries[:-1]):
            upper_boundary = boundaries[ii + 1]

            label = label_format_str.format(
                f"{lower_boundary / label_unit_value:.2f}",
                f"{upper_boundary / label_unit_value:.2f}",
                label_unit_latex,
            )
            group_labels[(lower_boundary, upper_boundary)] = label

            for s in copy(free_states):
                if lower_boundary <= getattr(s, attr) <= upper_boundary:
                    grouped_states[(lower_boundary, upper_boundary)].append(s)
                    free_states.remove(s)

        return grouped_states, group_labels

    def group_free_states_by_discrete_attr(
        self, attr="l", cutoff_value=9, label_format_str=r"\phi_{{ E, {} }}"
    ):
        grouped_states = collections.defaultdict(list)

        cutoff = []

        for s in self.sim.free_states:
            s_attr = getattr(s, attr)
            if s_attr < cutoff_value:
                grouped_states[getattr(s, attr)].append(s)
            else:
                cutoff.append(s)

        group_labels = {k: label_format_str.format(k) for k in grouped_states}

        try:
            cutoff_key = (
                max(grouped_states) + 1
            )  # get max key, make sure cutoff key is larger for sorting purposes
        except ValueError:
            cutoff_key = ""

        grouped_states[cutoff_key] = cutoff
        group_labels[cutoff_key] = label_format_str.format(rf"\geq {cutoff_value}")

        return grouped_states, group_labels

    def attach_electric_potential_plot_to_axis(
        self,
        axis: plt.Axes,
        show_electric_field: bool = True,
        show_vector_potential: bool = True,
        time_unit: u.Unit = "asec",
        legend_kwargs: Optional[dict] = None,
        show_y_label: bool = False,
    ):
        time_unit_value, time_unit_latex = u.get_unit_value_and_latex(time_unit)

        if legend_kwargs is None:
            legend_kwargs = dict()
        legend_defaults = dict(
            loc="lower left", fontsize=10, fancybox=True, framealpha=0.3
        )
        legend_kwargs = {**legend_defaults, **legend_kwargs}

        if show_electric_field:
            axis.plot(
                self.sim.data_times / time_unit_value,
                self.sim.data.electric_field_amplitude / u.atomic_electric_field,
                color=vis.COLOR_EFIELD,
                linewidth=1.5,
                label=fr"$ {vis.LATEX_EFIELD}(t) $",
            )
        if show_vector_potential:
            axis.plot(
                self.sim.data_times / time_unit_value,
                u.proton_charge
                * self.sim.data.vector_potential_amplitude
                / u.atomic_momentum,
                color=vis.COLOR_AFIELD,
                linewidth=1.5,
                label=fr"$ e \, {vis.LATEX_AFIELD}(t) $",
            )

        if show_y_label:
            axis.set_ylabel(
                rf"${vis.LATEX_EFIELD}(t)$", fontsize=13, color=vis.COLOR_EFIELD
            )

        axis.set_xlabel(rf"Time $t$ (${time_unit_latex}$)", fontsize=13)

        axis.tick_params(labelright=True)

        axis.set_xlim(
            self.sim.times[0] / time_unit_value, self.sim.times[-1] / time_unit_value
        )

        axis.legend(**legend_kwargs)

        axis.grid(True, **si.vis.DEFAULT_GRID_KWARGS)

    def state_overlaps_vs_time(
        self,
        states: Iterable[states.QuantumState] = None,
        log: bool = False,
        time_unit: u.Unit = "asec",
        show_electric_field: bool = True,
        show_vector_potential: bool = True,
        **kwargs,
    ):
        with si.vis.FigureManager(name=f"{self.spec.name}", **kwargs) as figman:
            time_unit_value, time_unit_latex = u.get_unit_value_and_latex(time_unit)

            grid_spec = matplotlib.gridspec.GridSpec(
                2, 1, height_ratios=[4, 1], hspace=0.07
            )
            ax_overlaps = plt.subplot(grid_spec[0])
            ax_field = plt.subplot(grid_spec[1], sharex=ax_overlaps)

            self.attach_electric_potential_plot_to_axis(
                ax_field,
                show_electric_field=show_electric_field,
                show_vector_potential=show_vector_potential,
                # legend_kwargs = dict(
                #     bbox_to_anchor = (1.1, .9),
                #     loc = 'upper left',
                #     borderaxespad = 0.1,
                #     fontsize = 10,
                # ),
            )

            ax_overlaps.plot(
                self.sim.data_times / time_unit_value,
                self.sim.data.norm,
                label=r"$\left\langle \psi|\psi \right\rangle$",
                color="black",
                linewidth=2,
            )

            state_overlaps = self.sim.data.state_overlaps
            if states is not None:
                if callable(states):
                    state_overlaps = {
                        state: overlap
                        for state, overlap in state_overlaps.items()
                        if states(state)
                    }
                else:
                    states = set(states)
                    state_overlaps = {
                        state: overlap
                        for state, overlap in state_overlaps.items()
                        if state in states
                        or (state.numeric and state.analytic_state in states)
                    }

            overlaps = [overlap for state, overlap in sorted(state_overlaps.items())]
            labels = [
                rf"$ \left| \left\langle \psi | {{{state.tex}}} \right\rangle \right|^2 $"
                for state, overlap in sorted(state_overlaps.items())
            ]

            ax_overlaps.stackplot(
                self.sim.data_times / time_unit_value,
                *overlaps,
                labels=labels,
                # colors = colors,
            )

            if log:
                ax_overlaps.set_yscale("log")
                min_overlap = min(
                    [np.min(overlap) for overlap in state_overlaps.values()]
                )
                ax_overlaps.set_ylim(bottom=max(1e-9, min_overlap * 0.1), top=1.0)
                ax_overlaps.grid(True, which="both", **si.vis.DEFAULT_GRID_KWARGS)
            else:
                ax_overlaps.set_ylim(0.0, 1.0)
                ax_overlaps.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                ax_overlaps.grid(True, **si.vis.DEFAULT_GRID_KWARGS)

            ax_overlaps.set_xlim(
                self.sim.times[0] / time_unit_value,
                self.sim.times[-1] / time_unit_value,
            )

            ax_overlaps.set_ylabel("Wavefunction Metric", fontsize=13)

            ax_overlaps.legend(
                bbox_to_anchor=(1.1, 1.1),
                loc="upper left",
                borderaxespad=0.075,
                fontsize=9,
                ncol=1 + (len(overlaps) // 10),
            )

            ax_overlaps.tick_params(labelright=True)

            ax_overlaps.xaxis.tick_top()

            # plt.rcParams['xtick.major.pad'] = 5
            # plt.rcParams['ytick.major.pad'] = 5

            # Find at most n+1 ticks on the y-axis at 'nice' locations
            max_yticks = 4
            yloc = plt.MaxNLocator(max_yticks, prune="upper")
            ax_field.yaxis.set_major_locator(yloc)

            max_xticks = 6
            xloc = plt.MaxNLocator(max_xticks, prune="both")
            ax_field.xaxis.set_major_locator(xloc)

            ax_field.tick_params(axis="both", which="major", labelsize=10)
            ax_overlaps.tick_params(axis="both", which="major", labelsize=10)

            postfix = ""
            if log:
                postfix += "__log"

            figman.name += postfix

    def wavefunction_vs_time(
        self,
        log: bool = False,
        time_unit: u.Unit = "asec",
        bound_state_max_n: int = 5,
        collapse_bound_state_angular_momenta: bool = True,
        grouped_free_states=None,
        group_free_states_labels=None,
        show_title: bool = False,
        plot_name_from: str = "file_name",
        show_electric_field: bool = True,
        show_vector_potential: bool = True,
        **kwargs,
    ):
        with si.vis.FigureManager(
            name=getattr(self, plot_name_from) + "__wavefunction_vs_time", **kwargs
        ) as figman:
            time_unit_value, time_unit_latex = u.get_unit_value_and_latex(time_unit)

            grid_spec = matplotlib.gridspec.GridSpec(
                2, 1, height_ratios=[4, 1], hspace=0.07
            )
            ax_overlaps = plt.subplot(grid_spec[0])
            ax_field = plt.subplot(grid_spec[1], sharex=ax_overlaps)

            self.attach_electric_potential_plot_to_axis(
                ax_field,
                show_electric_field=show_electric_field,
                show_vector_potential=show_vector_potential,
                # legend_kwargs = dict(
                #     bbox_to_anchor = (1.1, .9),
                #     loc = 'upper left',
                #     borderaxespad = 0.1,
                #     fontsize = 10)
            )

            ax_overlaps.plot(
                self.sim.data_times / time_unit_value,
                self.sim.data.norm,
                label=r"$\left\langle \Psi | \Psi \right\rangle$",
                color="black",
                linewidth=2,
            )

            if grouped_free_states is None:
                (
                    grouped_free_states,
                    group_free_states_labels,
                ) = self.group_free_states_by_continuous_attr("energy", attr_unit="eV")
            overlaps = []
            labels = []
            colors = []

            state_overlaps = (
                self.sim.data.state_overlaps
            )  # it's a property that would otherwise get evaluated every time we asked for it

            extra_bound_overlap = np.zeros(self.sim.data_time_steps)
            if collapse_bound_state_angular_momenta:
                overlaps_by_n = {
                    n: np.zeros(self.sim.data_time_steps)
                    for n in range(1, bound_state_max_n + 1)
                }  # prepare arrays to sum over angular momenta in, one for each n
                for state in sorted(self.sim.bound_states):
                    if state.n <= bound_state_max_n:
                        overlaps_by_n[state.n] += state_overlaps[state]
                    else:
                        extra_bound_overlap += state_overlaps[state]
                overlaps += [overlap for n, overlap in sorted(overlaps_by_n.items())]
                labels += [
                    rf"$ \left| \left\langle \Psi | \psi_{{ {n}, \ell }} \right\rangle \right|^2 $"
                    for n in sorted(overlaps_by_n)
                ]
                colors += [
                    matplotlib.colors.to_rgba("C" + str(n - 1), alpha=1)
                    for n in sorted(overlaps_by_n)
                ]
            else:
                for state in sorted(self.sim.bound_states):
                    if state.n <= bound_state_max_n:
                        overlaps.append(state_overlaps[state])
                        labels.append(
                            rf"$ \left| \left\langle \Psi | {{{state.tex}}} \right\rangle \right|^2 $"
                        )
                        colors.append(
                            matplotlib.colors.to_rgba(
                                "C" + str((state.n - 1) % 10),
                                alpha=1 - state.l / state.n,
                            )
                        )
                    else:
                        extra_bound_overlap += state_overlaps[state]

            overlaps.append(extra_bound_overlap)
            labels.append(
                rf"$ \left| \left\langle \Psi | \psi_{{n \geq {bound_state_max_n + 1} }}  \right\rangle \right|^2 $"
            )
            colors.append(".4")

            free_state_color_cycle = itertools.cycle(
                [
                    "#8dd3c7",
                    "#ffffb3",
                    "#bebada",
                    "#fb8072",
                    "#80b1d3",
                    "#fdb462",
                    "#b3de69",
                    "#fccde5",
                    "#d9d9d9",
                    "#bc80bd",
                    "#ccebc5",
                    "#ffed6f",
                ]
            )
            for group, states in sorted(grouped_free_states.items()):
                if len(states) != 0:
                    overlaps.append(np.sum(state_overlaps[s] for s in states))
                    labels.append(
                        rf"$\left| \left\langle \Psi | {{{group_free_states_labels[group]}}}  \right\rangle \right|^2$"
                    )
                    colors.append(free_state_color_cycle.__next__())

            overlaps = [overlap for overlap in overlaps]

            ax_overlaps.stackplot(
                self.sim.data_times / time_unit_value,
                *overlaps,
                labels=labels,
                colors=colors,
            )

            if log:
                ax_overlaps.set_yscale("log")
                min_overlap = min(
                    [np.min(overlap) for overlap in state_overlaps.values()]
                )
                ax_overlaps.set_ylim(bottom=max(1e-9, min_overlap * 0.1), top=1.0)
                ax_overlaps.grid(True, which="both", **si.vis.DEFAULT_GRID_KWARGS)
            else:
                ax_overlaps.set_ylim(0.0, 1.0)
                ax_overlaps.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                ax_overlaps.grid(True, **si.vis.DEFAULT_GRID_KWARGS)

            ax_overlaps.set_xlim(
                self.spec.time_initial / time_unit_value,
                self.spec.time_final / time_unit_value,
            )

            ax_overlaps.set_ylabel("Wavefunction Metric", fontsize=13)

            ax_overlaps.legend(
                bbox_to_anchor=(1.1, 1.1),
                loc="upper left",
                borderaxespad=0.075,
                fontsize=9,
                ncol=1 + (len(overlaps) // 12),
            )

            ax_overlaps.tick_params(
                labelleft=True,
                labelright=True,
                labeltop=True,
                labelbottom=False,
                bottom=True,
                top=True,
                left=True,
                right=True,
            )
            ax_field.tick_params(
                labelleft=True,
                labelright=True,
                labeltop=False,
                labelbottom=True,
                bottom=True,
                top=True,
                left=True,
                right=True,
            )

            # Find at most n+1 ticks on the y-axis at 'nice' locations
            max_yticks = 4
            yloc = plt.MaxNLocator(max_yticks, prune="upper")
            ax_field.yaxis.set_major_locator(yloc)

            max_xticks = 6
            xloc = plt.MaxNLocator(max_xticks, prune="both")
            ax_field.xaxis.set_major_locator(xloc)

            ax_field.tick_params(axis="both", which="major", labelsize=10)
            ax_overlaps.tick_params(axis="both", which="major", labelsize=10)

            if show_title:
                title = ax_overlaps.set_title(self.sim.name)
                title.set_y(1.15)

            postfix = ""
            if log:
                postfix += "__log"

            figman.name += postfix

    def energy_spectrum(
        self,
        states: str = "all",
        time_index: int = -1,
        energy_scale: str = "eV",
        time_scale: str = "asec",
        bins: int = 100,
        log: bool = False,
        energy_lower_bound: Optional[float] = None,
        energy_upper_bound: Optional[float] = None,
        group_angular_momentum: bool = True,
        angular_momentum_cutoff: Optional[int] = None,
        **kwargs,
    ):
        energy_unit, energy_unit_str = u.get_unit_value_and_latex(energy_scale)
        time_unit, time_unit_str = u.get_unit_value_and_latex(time_scale)

        if states == "all":
            state_list = self.spec.test_states
        elif states == "bound":
            state_list = self.sim.bound_states
        elif states == "free":
            state_list = self.sim.free_states
        else:
            raise ValueError("states must be one of 'all', 'bound', or 'free'")

        state_overlaps = self.sim.data.state_overlaps
        state_overlaps = {
            k: state_overlaps[k] for k in state_list
        }  # filter down to just states in state_list

        if group_angular_momentum:
            overlap_by_angular_momentum_by_energy = collections.defaultdict(
                functools.partial(collections.defaultdict, float)
            )

            for state, overlap_vs_time in state_overlaps.items():
                overlap_by_angular_momentum_by_energy[state.l][
                    state.energy
                ] += overlap_vs_time[time_index]

            energies = []
            overlaps = []
            cutoff_energies = np.array([])
            cutoff_overlaps = np.array([])
            for l, overlap_by_energy in sorted(
                overlap_by_angular_momentum_by_energy.items()
            ):
                if l < angular_momentum_cutoff:
                    e, o = si.utils.dict_to_arrays(overlap_by_energy)
                    energies.append(e / energy_unit)
                    overlaps.append(o)
                else:
                    e, o = si.utils.dict_to_arrays(overlap_by_energy)
                    cutoff_energies = np.append(cutoff_energies, e)
                    cutoff_overlaps = np.append(cutoff_overlaps, o)

            if len(cutoff_energies) != 0:
                energies.append(cutoff_energies)
                overlaps.append(cutoff_overlaps)

            if energy_lower_bound is None:
                energy_lower_bound = min([np.nanmin(e) for e in energies])
            if energy_upper_bound is None:
                energy_upper_bound = max([np.nanmax(e) for e in energies])

            labels = [rf"$ \ell = {l} $" for l in range(angular_momentum_cutoff)] + [
                rf"$ \ell \geq {angular_momentum_cutoff} $"
            ]
        else:
            overlap_by_energy = collections.defaultdict(float)
            for state, overlap_vs_time in state_overlaps.items():
                overlap_by_energy[state.energy] += overlap_vs_time[time_index]

            energies, overlaps = si.utils.dict_to_arrays(overlap_by_energy)
            energies /= energy_unit

            if energy_lower_bound is None:
                energy_lower_bound = np.nanmin(energies)
            if energy_upper_bound is None:
                energy_upper_bound = np.nanmax(energies)

            labels = None

        with si.vis.FigureManager(
            self.sim.name + "__energy_spectrum", **kwargs
        ) as figman:
            fig = figman.fig
            ax = fig.add_subplot(111)

            hist_n, hist_bins, hist_patches = ax.hist(
                x=energies,
                weights=overlaps,
                bins=bins,
                stacked=True,
                log=log,
                range=(energy_lower_bound, energy_upper_bound),
                label=labels,
            )

            ax.grid(True, **si.vis.DEFAULT_GRID_KWARGS)

            x_range = energy_upper_bound - energy_lower_bound
            ax.set_xlim(
                energy_lower_bound - 0.05 * x_range, energy_upper_bound + 0.05 * x_range
            )

            ax.set_xlabel(rf"Energy $E$ (${energy_unit_str}$)")
            ax.set_ylabel(r"Wavefunction Overlap")
            ax.set_title(
                rf"Wavefunction Overlap by Energy at $ t = {self.sim.times[time_index]/ time_unit:.3f} \, {time_unit_str} $"
            )

            if group_angular_momentum:
                ax.legend(loc="best", ncol=1 + len(energies) // 8)

            ax.tick_params(axis="both", which="major", labelsize=10)

            figman.name += f"__{states}_states__index={time_index}"

            if log:
                figman.name += "__log"
            if group_angular_momentum:
                figman.name += "__grouped"

    def radial_position_expectation_value_vs_time(
        self, use_name: bool = False, **kwargs
    ):
        if not use_name:
            prefix = self.sim.file_name
        else:
            prefix = self.sim.name

        si.vis.xy_plot(
            prefix + "__radial_position_vs_time",
            self.sim.data.times,
            self.sim.data.radial_position_expectation_value,
            x_label=r"Time $t$",
            x_unit="asec",
            y_label=r"Radial Position $\left\langle r(t) \right\rangle$",
            y_unit="bohr_radius",
            **kwargs,
        )

    def dipole_moment_expectation_value_vs_time(self, use_name: bool = False, **kwargs):
        if not use_name:
            prefix = self.sim.file_name
        else:
            prefix = self.sim.name

        si.vis.xy_plot(
            prefix + "__dipole_moment_vs_time",
            self.sim.data.times,
            self.sim.data.electric_dipole_moment_expectation_value,
            x_label=r"Time $t$",
            x_unit="asec",
            y_label=r"Dipole Moment $\left\langle d(t) \right\rangle$",
            y_unit="atomic_electric_dipole_moment",
            **kwargs,
        )

    def energy_expectation_value_vs_time(self, use_name: bool = False, **kwargs):
        if not use_name:
            prefix = self.sim.file_name
        else:
            prefix = self.sim.name

        si.vis.xy_plot(
            prefix + "__energy_vs_time",
            self.sim.data.times,
            self.sim.data.internal_energy_expectation_value,
            self.sim.data.total_energy_expectation_value,
            line_labels=[
                r"$\mathcal{H}_0$",
                r"$\mathcal{H}_0 + \mathcal{H}_{\mathrm{int}}$",
            ],
            x_label=r"Time $t$",
            x_unit="asec",
            y_label=r"Energy $\left\langle E(t) \right\rangle$",
            y_unit="eV",
            **kwargs,
        )

    def dipole_moment_vs_frequency(
        self,
        use_name: bool = False,
        gauge: str = "length",
        frequency_range: float = 10000 * u.THz,
        first_time: Optional[float] = None,
        last_time: Optional[float] = None,
        **kwargs,
    ):
        prefix = self.sim.file_name
        if use_name:
            prefix = self.sim.name

        frequency, dipole_moment = self.sim.dipole_moment_vs_frequency(
            gauge=gauge, first_time=first_time, last_time=last_time
        )

        si.vis.xy_plot(
            prefix + "__dipole_moment_vs_frequency",
            frequency,
            np.abs(dipole_moment) ** 2,
            x_unit_value="THz",
            y_unit_value=u.atomic_electric_dipole_moment ** 2,
            x_label="Frequency $f$",
            y_label=r"Dipole Moment $\left| d(\omega) \right|^2$ $\left( e^2 \, a_0^2 \right)$",
            x_lower_limit=0,
            x_upper_limit=frequency_range,
            y_log_axis=True,
            **kwargs,
        )


class SphericalHarmonicSimulationPlotter(MeshSimulationPlotter):
    def radial_probability_current_vs_time(
        self,
        time_unit: u.Unit = "asec",
        time_lower_limit: Optional[float] = None,
        time_upper_limit: Optional[float] = None,
        r_lower_limit: Optional[float] = None,
        r_upper_limit: Optional[float] = None,
        distance_unit: str = "bohr_radius",
        z_unit: str = "per_asec",
        z_limit: Optional[float] = None,
        use_name: bool = False,
        which: str = "sum",
        **kwargs,
    ):
        if which == "sum":
            z = self.radial_probability_current_vs_time
        elif which == "pos":
            z = self.radial_probability_current_vs_time__pos_z
        elif which == "neg":
            z = self.radial_probability_current_vs_time__neg_z
        else:
            raise AttributeError("which must be one of 'sum', 'pos', or 'neg'")

        prefix = self.sim.file_name
        if use_name:
            prefix = self.sim.name

        if z_limit is None:
            z_limit = np.nanmax(np.abs(self.radial_probability_current_vs_time))

        if time_lower_limit is None:
            time_lower_limit = self.sim.data_times[0]
        if time_upper_limit is None:
            time_upper_limit = self.sim.data_times[-1]

        try:
            r = self.mesh.r
        except AttributeError:
            r = np.linspace(0, self.spec.r_bound, self.spec.r_points)
            delta_r = r[1] - r[0]
            r += delta_r / 2

        if r_lower_limit is None:
            r_lower_limit = r[0]
        if r_upper_limit is None:
            r_upper_limit = r[-1]

        t_mesh, r_mesh = np.meshgrid(self.sim.data_times, r, indexing="ij")

        si.vis.xyz_plot(
            prefix + f"__radial_probability_current_{which}_vs_time",
            t_mesh,
            r_mesh,
            z,
            x_label=r"Time $t$",
            x_unit=time_unit,
            x_lower_limit=time_lower_limit,
            x_upper_limit=time_upper_limit,
            y_label=r"Radius $r$",
            y_unit=distance_unit,
            y_lower_limit=r_lower_limit,
            y_upper_limit=r_upper_limit,
            z_unit=z_unit,
            z_lower_limit=-z_limit,
            z_upper_limit=z_limit,
            z_label=r"$J_r$",
            colormap=plt.get_cmap("RdBu_r"),
            title=rf"Radial Probability Current vs. Time ({which})",
            **kwargs,
        )

    def radial_probability_current_vs_time__combined(
        self,
        r_upper_limit: Optional[float] = None,
        t_lower_limit: Optional[float] = None,
        t_upper_limit: Optional[float] = None,
        distance_unit: str = "bohr_radius",
        time_unit: u.Unit = "asec",
        current_unit: str = "per_asec",
        z_cut: float = 0.7,
        colormap=plt.get_cmap("coolwarm"),
        overlay_electric_field: bool = True,
        efield_unit: str = "atomic_electric_field",
        efield_color: str = "black",
        efield_label_fontsize: float = 12,
        title_fontsize: float = 12,
        y_axis_label_fontsize: float = 14,
        x_axis_label_fontsize: float = 12,
        cbar_label_fontsize: float = 12,
        aspect_ratio: float = 1.2,
        shading: str = "flat",
        use_name: bool = False,
        **kwargs,
    ):
        prefix = self.sim.file_name
        if use_name:
            prefix = self.sim.name

        distance_unit_value, distance_unit_latex = u.get_unit_value_and_latex(
            distance_unit
        )
        time_unit_value, time_unit_latex = u.get_unit_value_and_latex(time_unit)
        current_unit_value, current_unit_latex = u.get_unit_value_and_latex(
            current_unit
        )
        efield_unit_value, efield_unit_latex = u.get_unit_value_and_latex(efield_unit)

        if t_lower_limit is None:
            t_lower_limit = self.sim.data_times[0]
        if t_upper_limit is None:
            t_upper_limit = self.sim.data_times[-1]

        with si.vis.FigureManager(
            prefix + "__radial_probability_current_vs_time__combined",
            aspect_ratio=aspect_ratio,
            **kwargs,
        ) as figman:
            fig = figman.fig

            plt.set_cmap(colormap)

            gridspec = plt.GridSpec(2, 1, hspace=0.0)
            ax_pos = fig.add_subplot(gridspec[0])
            ax_neg = fig.add_subplot(gridspec[1], sharex=ax_pos)

            # TICKS, LEGEND, LABELS, and TITLE
            ax_pos.tick_params(
                labeltop=True,
                labelright=False,
                labelbottom=False,
                labelleft=True,
                bottom=False,
                right=False,
            )
            ax_neg.tick_params(
                labeltop=False,
                labelright=False,
                labelbottom=True,
                labelleft=True,
                top=False,
                right=False,
            )

            # pos_label = ax_pos.set_ylabel(f"$ r, \; z > 0 \; ({distance_unit_latex}) $", fontsize = y_axis_label_fontsize)
            # neg_label = ax_neg.set_ylabel(f"$ -r, \; z < 0 \; ({distance_unit_latex}) $", fontsize = y_axis_label_fontsize)
            pos_label = ax_pos.set_ylabel(f"$ z > 0 $", fontsize=y_axis_label_fontsize)
            neg_label = ax_neg.set_ylabel(f"$ z < 0 $", fontsize=y_axis_label_fontsize)
            ax_pos.yaxis.set_label_coords(-0.12, 0.65)
            ax_neg.yaxis.set_label_coords(-0.12, 0.35)
            r_label = ax_pos.text(
                -0.22,
                0.325,
                fr"Radius $ \pm r \; ({distance_unit_latex}) $",
                fontsize=y_axis_label_fontsize,
                rotation="vertical",
                transform=ax_pos.transAxes,
            )
            ax_neg.set_xlabel(
                rf"Time $ t \; ({time_unit_latex}) $", fontsize=x_axis_label_fontsize
            )
            suptitle = fig.suptitle(
                "Radial Probability Current vs. Time and Radius",
                fontsize=title_fontsize,
            )
            suptitle.set_x(0.6)
            suptitle.set_y(1.01)

            # COLORMESHES
            try:
                r = self.sim.mesh.r
            except AttributeError:
                r = np.linspace(0, self.spec.r_bound, self.spec.r_points)
                delta_r = r[1] - r[0]
                r += delta_r / 2

            t_mesh, r_mesh = np.meshgrid(self.sim.data_times, r, indexing="ij")

            # slicer = (slice(), slice(0, 50, 1))

            z_max = max(
                np.nanmax(
                    np.abs(self.sim.data.radial_probability_current_vs_time__pos_z)
                ),
                np.nanmax(np.abs(self.radial_probability_current_vs_time__neg_z)),
            )
            norm = matplotlib.colors.Normalize(
                vmin=-z_cut * z_max / current_unit_value,
                vmax=z_cut * z_max / current_unit_value,
            )

            pos_mesh = ax_pos.pcolormesh(
                t_mesh / time_unit_value,
                r_mesh / distance_unit_value,
                self.sim.data.radial_probability_current_vs_time__pos_z
                / current_unit_value,
                norm=norm,
                shading=shading,
            )
            neg_mesh = ax_neg.pcolormesh(
                t_mesh / time_unit_value,
                -r_mesh / distance_unit_value,
                self.sim.data.radial_probability_current_vs_time__neg_z
                / current_unit_value,
                norm=norm,
                shading=shading,
            )

            # LIMITS AND GRIDS
            grid_kwargs = si.vis.DEFAULT_GRID_KWARGS
            for ax in [ax_pos, ax_neg]:
                ax.set_xlim(
                    t_lower_limit / time_unit_value, t_upper_limit / time_unit_value
                )
                ax.grid(True, which="major", **grid_kwargs)

            if r_upper_limit is None:
                r_upper_limit = r[-1]
            ax_pos.set_ylim(0, r_upper_limit / distance_unit_value)
            ax_neg.set_ylim(-r_upper_limit / distance_unit_value, 0)

            y_ticks_neg = ax_neg.yaxis.get_major_ticks()
            y_ticks_neg[-1].label1.set_visible(False)

            # COLORBAR
            ax_pos_position = ax_pos.get_position()
            ax_neg_position = ax_neg.get_position()
            left, bottom, width, height = (
                ax_neg_position.x0,
                ax_neg_position.y0,
                ax_neg_position.x1 - ax_neg_position.x0,
                ax_pos_position.y1 - ax_neg_position.y0,
            )
            ax_cbar = fig.add_axes([left + width + 0.175, bottom, 0.05, height])
            cbar = plt.colorbar(mappable=pos_mesh, cax=ax_cbar, extend="both")
            z_label = cbar.set_label(
                rf"Radial Probability Current $ J_r \; ({current_unit_latex}) $",
                fontsize=cbar_label_fontsize,
            )

            # ELECTRIC FIELD OVERLAY
            if overlay_electric_field:
                ax_efield = fig.add_axes((left, bottom, width, height))

                ax_efield.tick_params(
                    labeltop=False,
                    labelright=True,
                    labelbottom=False,
                    labelleft=False,
                    left=False,
                    top=False,
                    bottom=False,
                    right=True,
                )
                ax_efield.tick_params(axis="y", colors=efield_color)
                ax_efield.tick_params(axis="x", colors=efield_color)

                (efield,) = ax_efield.plot(
                    self.sim.data_times / time_unit_value,
                    self.sim.data.electric_field_amplitude / efield_unit_value,
                    color=efield_color,
                    linestyle="-",
                )

                efield_grid_kwargs = {
                    **si.vis.DEFAULT_GRID_KWARGS,
                    **{"color": efield_color, "linestyle": "--"},
                }
                ax_efield.yaxis.grid(True, **efield_grid_kwargs)

                max_efield = np.nanmax(np.abs(self.sim.data.electric_field_amplitude))

                ax_efield.set_xlim(
                    t_lower_limit / time_unit_value, t_upper_limit / time_unit_value
                )
                ax_efield.set_ylim(
                    -1.05 * max_efield / efield_unit_value,
                    1.05 * max_efield / efield_unit_value,
                )
                ax_efield.set_ylabel(
                    rf"Electric Field Amplitude $ {vis.LATEX_EFIELD}(t) \; ({efield_unit_latex}) $",
                    color=efield_color,
                    fontsize=efield_label_fontsize,
                )
                ax_efield.yaxis.set_label_position("right")

    def angular_momentum_vs_time(
        self,
        use_name: bool = False,
        log: bool = False,
        renormalize: bool = False,
        **kwargs,
    ):
        fig = plt.figure(figsize=(7, 7 * 2 / 3), dpi=600)

        grid_spec = matplotlib.gridspec.GridSpec(
            2, 1, height_ratios=[4, 1], hspace=0.06
        )
        ax_momentums = plt.subplot(grid_spec[0])
        ax_field = plt.subplot(grid_spec[1], sharex=ax_momentums)

        if not isinstance(self.spec.electric_potential, potentials.NoPotentialEnergy):
            ax_field.plot(
                self.sim.times / u.asec,
                self.sim.data.electric_field_amplitude / u.atomic_electric_field,
                color="black",
                linewidth=2,
            )

        if renormalize:
            overlaps = [
                self.sim.data.norm_by_l[sph_harm] / self.sim.data.norm
                for sph_harm in self.spec.spherical_harmonics
            ]
            l_labels = [
                rf"$\left| \left\langle \Psi| {{{sph_harm.tex}}} \right\rangle \right|^2 / \left\langle \psi| \psi \right\rangle$"
                for sph_harm in self.spec.spherical_harmonics
            ]
        else:
            overlaps = [
                self.sim.data.norm_by_l[sph_harm]
                for sph_harm in self.spec.spherical_harmonics
            ]
            l_labels = [
                rf"$\left| \left\langle \Psi| {{{sph_harm.tex}}} \right\rangle \right|^2$"
                for sph_harm in self.spec.spherical_harmonics
            ]
        num_colors = len(overlaps)
        ax_momentums.set_prop_cycle(
            cycler(
                "color",
                [
                    plt.get_cmap("gist_rainbow")(n / num_colors)
                    for n in range(num_colors)
                ],
            )
        )
        ax_momentums.stackplot(
            self.sim.times / u.asec, *overlaps, alpha=1, labels=l_labels
        )

        if log:
            ax_momentums.set_yscale("log")
            ax_momentums.set_ylim(top=1.0)
            ax_momentums.grid(True, which="both")
        else:
            ax_momentums.set_ylim(0, 1.0)
            ax_momentums.set_yticks(
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            )
            ax_momentums.grid(True)
        ax_momentums.set_xlim(
            self.spec.time_initial / u.asec, self.spec.time_final / u.asec
        )

        ax_field.grid(True)

        ax_field.set_xlabel("Time $t$ (as)", fontsize=15)
        y_label = r"$\left| \left\langle \Psi | Y^l_0 \right\rangle \right|^2$"
        if renormalize:
            y_label += r"$/\left\langle \Psi|\Psi \right\rangle$"
        ax_momentums.set_ylabel(y_label, fontsize=15)
        ax_field.set_ylabel(rf"${vis.LATEX_EFIELD}(t)$ (a.u.)", fontsize=11)

        ax_momentums.legend(
            bbox_to_anchor=(1.1, 1),
            loc="upper left",
            borderaxespad=0.0,
            fontsize=10,
            ncol=1 + (len(self.spec.spherical_harmonics) // 17),
        )

        ax_momentums.tick_params(labelright=True)
        ax_field.tick_params(labelright=True)
        ax_momentums.xaxis.tick_top()

        plt.rcParams["xtick.major.pad"] = 5
        plt.rcParams["ytick.major.pad"] = 5

        # Find at most n+1 ticks on the y-axis at 'nice' locations
        max_yticks = 6
        yloc = plt.MaxNLocator(max_yticks, prune="upper")
        ax_field.yaxis.set_major_locator(yloc)

        max_xticks = 6
        xloc = plt.MaxNLocator(max_xticks, prune="both")
        ax_field.xaxis.set_major_locator(xloc)

        ax_field.tick_params(axis="x", which="major", labelsize=10)
        ax_field.tick_params(axis="y", which="major", labelsize=10)
        ax_momentums.tick_params(axis="both", which="major", labelsize=10)

        postfix = ""
        if renormalize:
            postfix += "_renorm"
        prefix = self.sim.file_name
        if use_name:
            prefix = self.sim.name
        si.vis.save_current_figure(
            name=prefix + f"__angular_momentum_vs_time{postfix}", **kwargs
        )

        plt.close()
