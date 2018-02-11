import collections
import functools
import itertools
import datetime
import logging
from copy import copy, deepcopy
from typing import Union, Optional, Iterable, Dict, Callable

import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np
import numpy.fft as nfft
from tqdm import tqdm

import simulacra as si
import simulacra.units as u

from .. import potentials, states, vis, core
from . import meshes, anim, snapshots, data

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MeshSimulation(si.Simulation):
    def __init__(self, spec: 'MeshSpecification'):
        super().__init__(spec)

        self.latest_checkpoint_time = datetime.datetime.utcnow()

        self.times = self.get_times()

        if self.spec.electric_potential_dc_correction:
            old_pot = self.spec.electric_potential
            self.spec.electric_potential = potentials.DC_correct_electric_potential(self.spec.electric_potential, self.times)

            logger.warning('Replaced electric potential {} --> {} for {} {}'.format(old_pot, self.spec.electric_potential, self.__class__.__name__, self.name))

        self.time_index = 0
        self.data_time_index = 0
        self.time_steps = len(self.times)

        self.initialize_mesh()

        # simulation data storage
        time_indices = np.array(range(0, self.time_steps))
        self.data_mask = np.equal(time_indices, 0) + np.equal(time_indices, self.time_steps - 1)
        if self.spec.store_data_every >= 1:
            self.data_mask += np.equal(time_indices % self.spec.store_data_every, 0)
        self.data_times = self.times[self.data_mask]
        self.data_indices = time_indices[self.data_mask]
        self.data_time_steps = len(self.data_times)

        # data storage initialization
        self.norm_vs_time = np.zeros(self.data_time_steps, dtype = np.float64) * np.NaN

        self.inner_products_vs_time = {state: np.zeros(self.data_time_steps, dtype = np.complex128) * np.NaN
                                       for state in self.spec.test_states}

        self.electric_field_amplitude_vs_time = np.zeros(self.data_time_steps, dtype = np.float64) * np.NaN
        self.vector_potential_amplitude_vs_time = np.zeros(self.data_time_steps, dtype = np.float64) * np.NaN

        # optional data storage initialization
        if self.spec.store_norm_diff_mask:
            self.norm_diff_mask_vs_time = np.zeros(self.data_time_steps, dtype = np.float64) * np.NaN

        self.data = data.Data(self)
        self.datastores = {datastore_type.__name__: datastore_type(self) for datastore_type in self.spec.datastore_types}

        # populate the snapshot times from the two ways of entering snapshot times in the spec (by index or by time)
        self.snapshot_times = set()

        for time in self.spec.snapshot_times:
            time_index, time_target, _ = si.utils.find_nearest_entry(self.times, time)
            self.snapshot_times.add(time_target)

        for index in self.spec.snapshot_indices:
            self.snapshot_times.add(self.times[index])

        self.snapshots = dict()

        self.warnings = collections.defaultdict(list)

    def get_blank_data(self, dtype = np.float64) -> np.array:
        return np.zeros(self.data_time_steps, dtype = dtype) * np.NaN

    def info(self) -> si.Info:
        info = super().info()

        mem_mesh = self.mesh.g.nbytes if self.mesh is not None else 0

        mem_matrix_operators = 6 * mem_mesh
        mem_numeric_eigenstates = sum(state.g.nbytes for state in self.spec.test_states if state.numeric and state.g is not None)
        # mem_inner_products = sum(overlap.nbytes for overlap in self.data.inner_products_vs_time.values())
        #
        # mem_other_time_data = sum(x.nbytes for x in (
        #     self.electric_field_amplitude_vs_time,
        #     self.vector_potential_amplitude_vs_time,
        #     self.norm_vs_time,
        # ))
        #
        # for attr in (
        #         'radial_position_expectation_value_vs_time',
        #         'internal_energy_expectation_value_vs_time',
        #         'total_energy_expectation_value_vs_time',
        #         'electric_dipole_moment_expectation_value_vs_time'
        #         'norm_diff_mask_vs_time',
        #         'radial_probability_current_vs_time__pos_z',
        #         'radial_probability_current_vs_time__neg_z',
        # ):
        #     try:
        #         mem_other_time_data += getattr(self, attr).nbytes
        #     except AttributeError:  # apparently we're not storing that data
        #         pass

        # try:
        #     mem_other_time_data += sum(h.nbytes for h in self.norm_by_harmonic_vs_time.values())
        # except AttributeError:
        #     pass

        mem_misc = sum(x.nbytes for x in (
            self.times,
            self.data_times,
            self.data_mask,
            self.data_indices,
        ))

        mem_total = sum((
            mem_mesh,
            mem_matrix_operators,
            mem_numeric_eigenstates,
            mem_inner_products,
            mem_other_time_data,
            mem_misc,
        ))

        info_mem = si.Info(header = f'Memory Usage (approx.): {si.utils.bytes_to_str(mem_total)}')
        info_mem.add_field('g', si.utils.bytes_to_str(mem_mesh))
        info_mem.add_field('Matrix Operators', si.utils.bytes_to_str(mem_matrix_operators))
        info_mem.add_field('Numeric Eigenstates', si.utils.bytes_to_str(mem_numeric_eigenstates))
        info_mem.add_field('State Inner Products', si.utils.bytes_to_str(mem_inner_products))
        info_mem.add_field('Other Time-Indexed Data', si.utils.bytes_to_str(mem_other_time_data))
        info_mem.add_field('Miscellaneous', si.utils.bytes_to_str(mem_misc))

        info.add_info(info_mem)

        return info

    @property
    def available_animation_frames(self) -> int:
        return self.time_steps

    @property
    def time(self) -> float:
        return self.times[self.time_index]

    @property
    def times_to_current(self) -> np.array:
        return self.times[:self.time_index + 1]

    @property
    def state_overlaps_vs_time(self) -> Dict[states.QuantumState, np.array]:
        return {state: np.abs(inner_product) ** 2 for state, inner_product in self.inner_products_vs_time.items()}

    @property
    def total_overlap_vs_time(self) -> np.array:
        return np.sum(overlap for overlap in self.state_overlaps_vs_time.values())

    @property
    def total_bound_state_overlap_vs_time(self) -> np.array:
        return np.sum(overlap for state, overlap in self.state_overlaps_vs_time.items() if state.bound)

    @property
    def total_free_state_overlap_vs_time(self) -> np.array:
        return np.sum(overlap for state, overlap in self.state_overlaps_vs_time.items() if state.free)

    def get_times(self) -> np.array:
        if not callable(self.spec.time_step):
            total_time = self.spec.time_final - self.spec.time_initial
            times = np.linspace(self.spec.time_initial, self.spec.time_final, int(total_time / self.spec.time_step) + 1)
        else:
            t = self.spec.time_initial
            times = [t]

            while t < self.spec.time_final:
                t += self.spec.time_step(t, self.spec)

                if t > self.spec.time_final:
                    t = self.spec.time_final

                times.append(t)

            times = np.array(times)

        return times

    def initialize_mesh(self):
        self.mesh = self.spec.mesh_type(self)

        logger.debug('Initialized mesh for {} {}'.format(self.__class__.__name__, self.name))

    def store_data(self):
        """Update the time-indexed data arrays with the current values."""
        for datastore in self.datastores.values():
            datastore.store()

        norm = self.mesh.norm()
        self.norm_vs_time[self.data_time_index] = norm
        if norm > 1.001 * self.norm_vs_time[0]:
            logger.warning(f'Wavefunction norm ({norm}) has exceeded initial norm ({self.norm_vs_time[0]}) by more than .1% for {self.__class__.__name__} {self.name}')
        try:
            if norm > 1.001 * self.norm_vs_time[self.data_time_index - 1]:
                logger.warning(f'Wavefunction norm ({norm}) at time_index = {self.data_time_index} has exceeded norm from previous time step ({self.norm_vs_time[self.data_time_index - 1]}) by more than .1% for {self.__class__.__name__} {self.name}')
        except IndexError:
            pass

        logger.debug(f'{self.__class__.__name__} {self.name} stored data for time index {self.time_index} (data time index {self.data_time_index})')

    def take_snapshot(self):
        snapshot = self.spec.snapshot_type(self, self.time_index, **self.spec.snapshot_kwargs)

        snapshot.take_snapshot()

        self.snapshots[self.time_index] = snapshot

        logger.info(f'Stored {snapshot.__class__.__name__} of {self.name} at time {u.uround(self.time, u.asec)} as (time index {self.time_index})')

    def run_simulation(self, progress_bar: bool = False, callback: Callable = None):
        """
        Run the simulation by repeatedly evolving the mesh by the time step and recovering various data from it.
        """
        logger.info(f'Performing time evolution on {self}, starting from time index {self.time_index}')
        try:
            self.status = si.Status.RUNNING

            for animator in self.spec.animators:
                animator.initialize(self)

            if progress_bar:
                pbar = tqdm(total = self.time_steps - 1, ascii = True)

            while True:
                if self.time in self.data_times:
                    self.store_data()

                if self.time in self.snapshot_times:
                    self.take_snapshot()

                for animator in self.spec.animators:
                    if self.time_index == 0 or self.time_index == self.time_steps or self.time_index % animator.decimation == 0:
                        animator.send_frame_to_ffmpeg()

                if self.time in self.data_times:  # having to repeat this is clunky, but I need the data for the animators to work and I can't change the data index until the animators are done
                    self.data_time_index += 1

                if callback is not None:
                    callback(self)

                if self.time_index == self.time_steps - 1:
                    break

                self.time_index += 1

                norm_diff_mask = self.mesh.evolve(self.times[self.time_index] - self.times[self.time_index - 1])  # evolve the mesh forward to the next time step
                if self.spec.store_norm_diff_mask:
                    self.norm_diff_mask_vs_time[self.data_time_index] = norm_diff_mask  # move to store data so it has the right index?

                logger.debug(f'{self.__class__.__name__} {self.name} ({self.file_name}) evolved to time index {self.time_index} / {self.time_steps - 1} ({np.around(100 * (self.time_index + 1) / self.time_steps, 2)}%)')

                if self.spec.checkpoints:
                    now = datetime.datetime.utcnow()
                    if (now - self.latest_checkpoint_time) > self.spec.checkpoint_every:
                        self.save(target_dir = self.spec.checkpoint_dir, save_mesh = True)
                        self.latest_checkpoint_time = now
                        logger.info(f'{self} checkpointed at time index {self.time_index} / {self.time_steps - 1} ({np.around(100 * (self.time_index + 1) / self.time_steps, 2)}%)')
                        self.status = si.Status.RUNNING

                try:
                    pbar.update(1)
                except NameError:
                    pass

            try:
                pbar.close()
            except NameError:
                pass

            self.status = si.Status.FINISHED
            logger.info(f'Finished performing time evolution on {self.name} ({self.file_name})')
        except Exception as e:
            raise e
        finally:
            # make sure the animators get cleaned up if there's some kind of error during time evolution
            for animator in self.spec.animators:
                animator.cleanup()

            self.spec.animators = ()

    @property
    def bound_states(self) -> Iterable[states.QuantumState]:
        yield from [s for s in self.spec.test_states if s.bound]

    @property
    def free_states(self) -> Iterable[states.QuantumState]:
        yield from [s for s in self.spec.test_states if not s.bound]

    def group_free_states_by_continuous_attr(self,
                                             attr = 'energy',
                                             divisions = 10,
                                             cutoff_value = None,
                                             label_format_str = r'\phi_{{    {} \; \mathrm{{to}} \; {} \, {}, \ell   }}',
                                             attr_unit = 'eV'):
        spectrum = set(getattr(s, attr) for s in self.free_states)

        grouped_states = collections.defaultdict(list)
        group_labels = {}

        try:
            attr_min, attr_max = min(spectrum), max(spectrum)
        except ValueError:
            return [], []

        if cutoff_value is None:
            boundaries = np.linspace(attr_min, attr_max, num = divisions + 1)
        else:
            boundaries = np.linspace(attr_min, cutoff_value, num = divisions)
            boundaries = np.concatenate((boundaries, [attr_max]))

        label_unit_value, label_unit_latex = u.get_unit_value_and_latex_from_unit(attr_unit)

        free_states = list(self.free_states)

        for ii, lower_boundary in enumerate(boundaries[:-1]):
            upper_boundary = boundaries[ii + 1]

            label = label_format_str.format(u.uround(lower_boundary, label_unit_value, 2), u.uround(upper_boundary, label_unit_value, 2), label_unit_latex)
            group_labels[(lower_boundary, upper_boundary)] = label

            for s in copy(free_states):
                if lower_boundary <= getattr(s, attr) <= upper_boundary:
                    grouped_states[(lower_boundary, upper_boundary)].append(s)
                    free_states.remove(s)

        return grouped_states, group_labels

    def group_free_states_by_discrete_attr(self, attr = 'l', cutoff_value = 9, label_format_str = r'\phi_{{ E, {} }}'):
        grouped_states = collections.defaultdict(list)

        cutoff = []

        for s in self.free_states:
            s_attr = getattr(s, attr)
            if s_attr < cutoff_value:
                grouped_states[getattr(s, attr)].append(s)
            else:
                cutoff.append(s)

        group_labels = {k: label_format_str.format(k) for k in grouped_states}

        try:
            cutoff_key = max(grouped_states) + 1  # get max key, make sure cutoff key is larger for sorting purposes
        except ValueError:
            cutoff_key = ''

        grouped_states[cutoff_key] = cutoff
        group_labels[cutoff_key] = label_format_str.format(r'\geq {}'.format(cutoff_value))

        return grouped_states, group_labels

    def attach_electric_potential_plot_to_axis(self,
                                               axis: plt.Axes,
                                               show_electric_field: bool = True,
                                               show_vector_potential: bool = True,
                                               time_unit: str = 'asec',
                                               legend_kwargs: Optional[dict] = None,
                                               show_y_label: bool = False, ):
        time_unit_value, time_unit_latex = u.get_unit_value_and_latex_from_unit(time_unit)

        if legend_kwargs is None:
            legend_kwargs = dict()
        legend_defaults = dict(
            loc = 'lower left',
            fontsize = 10,
            fancybox = True,
            framealpha = .3,
        )
        legend_kwargs = {**legend_defaults, **legend_kwargs}

        if show_electric_field:
            axis.plot(
                self.data_times / time_unit_value,
                self.electric_field_amplitude_vs_time / u.atomic_electric_field,
                color = vis.COLOR_EFIELD,
                linewidth = 1.5,
                label = fr'$ {vis.LATEX_EFIELD}(t) $',
            )
        if show_vector_potential:
            axis.plot(
                self.data_times / time_unit_value,
                u.proton_charge * self.vector_potential_amplitude_vs_time / u.atomic_momentum,
                color = vis.COLOR_AFIELD,
                linewidth = 1.5,
                label = fr'$ e \, {vis.LATEX_AFIELD}(t) $',
            )

        if show_y_label:
            axis.set_ylabel('${}(t)$'.format(vis.LATEX_EFIELD), fontsize = 13, color = vis.COLOR_EFIELD)

        axis.set_xlabel('Time $t$ (${}$)'.format(time_unit_latex), fontsize = 13)

        axis.tick_params(labelright = True)

        axis.set_xlim(self.times[0] / time_unit_value, self.times[-1] / time_unit_value)

        axis.legend(**legend_kwargs)

        axis.grid(True, **si.vis.GRID_KWARGS)

    def plot_state_overlaps_vs_time(self,
                                    states: Iterable[states.QuantumState] = None,
                                    log: bool = False,
                                    time_unit: str = 'asec',
                                    show_electric_field: bool = True,
                                    show_vector_potential: bool = True,
                                    **kwargs):
        with si.vis.FigureManager(name = f'{self.spec.name}', **kwargs) as figman:
            time_unit_value, time_unit_latex = u.get_unit_value_and_latex_from_unit(time_unit)

            grid_spec = matplotlib.gridspec.GridSpec(2, 1, height_ratios = [4, 1], hspace = 0.07)  # TODO: switch to fixed axis construction
            ax_overlaps = plt.subplot(grid_spec[0])
            ax_field = plt.subplot(grid_spec[1], sharex = ax_overlaps)

            self.attach_electric_potential_plot_to_axis(ax_field,
                                                        show_electric_field = show_electric_field,
                                                        show_vector_potential = show_vector_potential,
                                                        # legend_kwargs = dict(
                                                        #     bbox_to_anchor = (1.1, .9),
                                                        #     loc = 'upper left',
                                                        #     borderaxespad = 0.1,
                                                        #     fontsize = 10)
                                                        )

            ax_overlaps.plot(self.data_times / time_unit_value, self.norm_vs_time, label = r'$\left\langle \psi|\psi \right\rangle$', color = 'black', linewidth = 2)

            state_overlaps = self.state_overlaps_vs_time
            if states is not None:
                if callable(states):
                    state_overlaps = {state: overlap for state, overlap in state_overlaps.items() if states(state)}
                else:
                    states = set(states)
                    state_overlaps = {state: overlap for state, overlap in state_overlaps.items() if state in states or (state.numeric and state.analytic_state in states)}

            overlaps = [overlap for state, overlap in sorted(state_overlaps.items())]
            labels = [r'$\left| \left\langle \psi|{} \right\rangle \right|^2$'.format(state.latex) for state, overlap in sorted(state_overlaps.items())]

            ax_overlaps.stackplot(self.data_times / time_unit_value,
                                  *overlaps,
                                  labels = labels,
                                  # colors = colors,
                                  )

            if log:
                ax_overlaps.set_yscale('log')
                min_overlap = min([np.min(overlap) for overlap in state_overlaps.values()])
                ax_overlaps.set_ylim(bottom = max(1e-9, min_overlap * .1), top = 1.0)
                ax_overlaps.grid(True, which = 'both', **si.vis.GRID_KWARGS)
            else:
                ax_overlaps.set_ylim(0.0, 1.0)
                ax_overlaps.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                ax_overlaps.grid(True, **si.vis.GRID_KWARGS)

            ax_overlaps.set_xlim(self.times[0] / time_unit_value, self.times[-1] / time_unit_value)

            ax_overlaps.set_ylabel('Wavefunction Metric', fontsize = 13)

            ax_overlaps.legend(
                bbox_to_anchor = (1.1, 1.1),
                loc = 'upper left', borderaxespad = 0.075,
                fontsize = 9,
                ncol = 1 + (len(overlaps) // 10))

            ax_overlaps.tick_params(labelright = True)

            ax_overlaps.xaxis.tick_top()

            # plt.rcParams['xtick.major.pad'] = 5
            # plt.rcParams['ytick.major.pad'] = 5

            # Find at most n+1 ticks on the y-axis at 'nice' locations
            max_yticks = 4
            yloc = plt.MaxNLocator(max_yticks, prune = 'upper')
            ax_field.yaxis.set_major_locator(yloc)

            max_xticks = 6
            xloc = plt.MaxNLocator(max_xticks, prune = 'both')
            ax_field.xaxis.set_major_locator(xloc)

            ax_field.tick_params(axis = 'both', which = 'major', labelsize = 10)
            ax_overlaps.tick_params(axis = 'both', which = 'major', labelsize = 10)

            postfix = ''
            if log:
                postfix += '__log'

            figman.name += postfix

    def plot_wavefunction_vs_time(self,
                                  log: bool = False,
                                  time_unit: str = 'asec',
                                  bound_state_max_n: int = 5,
                                  collapse_bound_state_angular_momenta: bool = True,
                                  grouped_free_states = None,
                                  group_free_states_labels = None,
                                  show_title: bool = False,
                                  plot_name_from: str = 'file_name',
                                  show_electric_field: bool = True,
                                  show_vector_potential: bool = True,
                                  **kwargs):
        with si.vis.FigureManager(name = getattr(self, plot_name_from) + '__wavefunction_vs_time', **kwargs) as figman:
            time_unit_value, time_unit_latex = u.get_unit_value_and_latex_from_unit(time_unit)

            grid_spec = matplotlib.gridspec.GridSpec(2, 1, height_ratios = [4, 1], hspace = 0.07)
            ax_overlaps = plt.subplot(grid_spec[0])
            ax_field = plt.subplot(grid_spec[1], sharex = ax_overlaps)

            self.attach_electric_potential_plot_to_axis(ax_field,
                                                        show_electric_field = show_electric_field,
                                                        show_vector_potential = show_vector_potential,
                                                        # legend_kwargs = dict(
                                                        #     bbox_to_anchor = (1.1, .9),
                                                        #     loc = 'upper left',
                                                        #     borderaxespad = 0.1,
                                                        #     fontsize = 10)
                                                        )

            ax_overlaps.plot(self.data_times / time_unit_value, self.norm_vs_time, label = r'$\left\langle \Psi | \Psi \right\rangle$', color = 'black', linewidth = 2)

            if grouped_free_states is None:
                grouped_free_states, group_free_states_labels = self.group_free_states_by_continuous_attr('energy', attr_unit = 'eV')
            overlaps = []
            labels = []
            colors = []

            state_overlaps = self.state_overlaps_vs_time  # it's a property that would otherwise get evaluated every time we asked for it

            extra_bound_overlap = np.zeros(self.data_time_steps)
            if collapse_bound_state_angular_momenta:
                overlaps_by_n = {n: np.zeros(self.data_time_steps) for n in range(1, bound_state_max_n + 1)}  # prepare arrays to sum over angular momenta in, one for each n
                for state in sorted(self.bound_states):
                    if state.n <= bound_state_max_n:
                        overlaps_by_n[state.n] += state_overlaps[state]
                    else:
                        extra_bound_overlap += state_overlaps[state]
                overlaps += [overlap for n, overlap in sorted(overlaps_by_n.items())]
                labels += [r'$\left| \left\langle \Psi | \psi_{{ {}, \ell }} \right\rangle \right|^2$'.format(n) for n in sorted(overlaps_by_n)]
                colors += [matplotlib.colors.to_rgba('C' + str(n - 1), alpha = 1) for n in sorted(overlaps_by_n)]
            else:
                for state in sorted(self.bound_states):
                    if state.n <= bound_state_max_n:
                        overlaps.append(state_overlaps[state])
                        labels.append(r'$\left| \left\langle \Psi | {} \right\rangle \right|^2$'.format(state.latex))
                        colors.append(matplotlib.colors.to_rgba('C' + str((state.n - 1) % 10), alpha = 1 - state.l / state.n))
                    else:
                        extra_bound_overlap += state_overlaps[state]

            overlaps.append(extra_bound_overlap)
            labels.append(r'$\left| \left\langle \Psi | \psi_{{n \geq {} }}  \right\rangle \right|^2$'.format(bound_state_max_n + 1))
            colors.append('.4')

            free_state_color_cycle = itertools.cycle(['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5', '#ffed6f'])
            for group, states in sorted(grouped_free_states.items()):
                if len(states) != 0:
                    overlaps.append(np.sum(state_overlaps[s] for s in states))
                    labels.append(r'$\left| \left\langle \Psi | {}  \right\rangle \right|^2$'.format(group_free_states_labels[group]))
                    colors.append(free_state_color_cycle.__next__())

            overlaps = [overlap for overlap in overlaps]

            ax_overlaps.stackplot(
                self.data_times / time_unit_value,
                *overlaps,
                labels = labels,
                colors = colors,
            )

            if log:
                ax_overlaps.set_yscale('log')
                min_overlap = min([np.min(overlap) for overlap in state_overlaps.values()])
                ax_overlaps.set_ylim(bottom = max(1e-9, min_overlap * .1), top = 1.0)
                ax_overlaps.grid(True, which = 'both', **si.vis.GRID_KWARGS)
            else:
                ax_overlaps.set_ylim(0.0, 1.0)
                ax_overlaps.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                ax_overlaps.grid(True, **si.vis.GRID_KWARGS)

            ax_overlaps.set_xlim(self.spec.time_initial / time_unit_value, self.spec.time_final / time_unit_value)

            ax_overlaps.set_ylabel('Wavefunction Metric', fontsize = 13)

            ax_overlaps.legend(
                bbox_to_anchor = (1.1, 1.1),
                loc = 'upper left',
                borderaxespad = 0.075,
                fontsize = 9,
                ncol = 1 + (len(overlaps) // 12)
            )

            ax_overlaps.tick_params(
                labelleft = True,
                labelright = True,
                labeltop = True,
                labelbottom = False,
                bottom = True,
                top = True,
                left = True,
                right = True
            )
            ax_field.tick_params(
                labelleft = True,
                labelright = True,
                labeltop = False,
                labelbottom = True,
                bottom = True,
                top = True,
                left = True,
                right = True
            )

            # Find at most n+1 ticks on the y-axis at 'nice' locations
            max_yticks = 4
            yloc = plt.MaxNLocator(max_yticks, prune = 'upper')
            ax_field.yaxis.set_major_locator(yloc)

            max_xticks = 6
            xloc = plt.MaxNLocator(max_xticks, prune = 'both')
            ax_field.xaxis.set_major_locator(xloc)

            ax_field.tick_params(axis = 'both', which = 'major', labelsize = 10)
            ax_overlaps.tick_params(axis = 'both', which = 'major', labelsize = 10)

            if show_title:
                title = ax_overlaps.set_title(self.name)
                title.set_y(1.15)

            postfix = ''
            if log:
                postfix += '__log'

            figman.name += postfix

    def plot_energy_spectrum(self,
                             states: str = 'all',
                             time_index: int = -1,
                             energy_scale: str = 'eV',
                             time_scale: str = 'asec',
                             bins: int = 100,
                             log: bool = False,
                             energy_lower_bound: Optional[float] = None,
                             energy_upper_bound: Optional[float] = None,
                             group_angular_momentum: bool = True,
                             angular_momentum_cutoff: Optional[int] = None,
                             **kwargs):
        energy_unit, energy_unit_str = u.get_unit_value_and_latex_from_unit(energy_scale)
        time_unit, time_unit_str = u.get_unit_value_and_latex_from_unit(time_scale)

        if states == 'all':
            state_list = self.spec.test_states
        elif states == 'bound':
            state_list = self.bound_states
        elif states == 'free':
            state_list = self.free_states
        else:
            raise ValueError("states must be one of 'all', 'bound', or 'free'")

        state_overlaps = self.state_overlaps_vs_time
        state_overlaps = {k: state_overlaps[k] for k in state_list}  # filter down to just states in state_list

        if group_angular_momentum:
            overlap_by_angular_momentum_by_energy = collections.defaultdict(functools.partial(collections.defaultdict, float))

            for state, overlap_vs_time in state_overlaps.items():
                overlap_by_angular_momentum_by_energy[state.l][state.energy] += overlap_vs_time[time_index]

            energies = []
            overlaps = []
            cutoff_energies = np.array([])
            cutoff_overlaps = np.array([])
            for l, overlap_by_energy in sorted(overlap_by_angular_momentum_by_energy.items()):
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

            labels = [r'$\ell = {}$'.format(l) for l in range(angular_momentum_cutoff)] + [r'$\ell \geq {}$'.format(angular_momentum_cutoff)]
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

        with si.vis.FigureManager(self.name + '__energy_spectrum', **kwargs) as figman:
            fig = figman.fig
            ax = fig.add_subplot(111)

            hist_n, hist_bins, hist_patches = ax.hist(x = energies, weights = overlaps,
                                                      bins = bins,
                                                      stacked = True,
                                                      log = log,
                                                      range = (energy_lower_bound, energy_upper_bound),
                                                      label = labels,
                                                      )

            ax.grid(True, **si.vis.GRID_KWARGS)

            x_range = energy_upper_bound - energy_lower_bound
            ax.set_xlim(energy_lower_bound - .05 * x_range, energy_upper_bound + .05 * x_range)

            ax.set_xlabel('Energy $E$ (${}$)'.format(energy_unit_str))
            ax.set_ylabel('Wavefunction Overlap'.format(energy_unit_str))
            ax.set_title('Wavefunction Overlap by Energy at $t={} \, {}$'.format(u.uround(self.times[time_index], time_unit, 3), time_unit_str))

            if group_angular_momentum:
                ax.legend(loc = 'best', ncol = 1 + len(energies) // 8)

            ax.tick_params(axis = 'both', which = 'major', labelsize = 10)

            figman.name += '__{}_states__index={}'.format(states, time_index)

            if log:
                figman.name += '__log'
            if group_angular_momentum:
                figman.name += '__grouped'

    def plot_radial_position_expectation_value_vs_time(self, use_name: bool = False, **kwargs):
        if not use_name:
            prefix = self.file_name
        else:
            prefix = self.name

        si.vis.xy_plot(
            prefix + '__radial_position_vs_time',
            self.data_times,
            self.radial_position_expectation_value_vs_time,
            x_label = r'Time $t$', x_unit = 'asec',
            y_label = r'Radial Position $\left\langle r(t) \right\rangle$', y_unit = 'bohr_radius',
            **kwargs
        )

    def plot_dipole_moment_expectation_value_vs_time(self, use_name: bool = False, **kwargs):
        if not use_name:
            prefix = self.file_name
        else:
            prefix = self.name

        si.vis.xy_plot(
            prefix + '__dipole_moment_vs_time',
            self.data_times,
            self.electric_dipole_moment_expectation_value_vs_time,
            x_label = r'Time $t$', x_unit = 'asec',
            y_label = r'Dipole Moment $\left\langle d(t) \right\rangle$', y_unit = 'atomic_electric_dipole_moment',
            **kwargs
        )

    def plot_energy_expectation_value_vs_time(self, use_name: bool = False, **kwargs):
        if not use_name:
            prefix = self.file_name
        else:
            prefix = self.name

        si.vis.xy_plot(
            prefix + '__energy_vs_time',
            self.data_times,
            self.internal_energy_expectation_value_vs_time,
            self.total_energy_expectation_value_vs_time,
            line_labels = [r'$\mathcal{H}_0$', r'$\mathcal{H}_0 + \mathcal{H}_{\mathrm{int}}$'],
            x_label = r'Time $t$', x_unit = 'asec',
            y_label = r'Energy $\left\langle E(t) \right\rangle$', y_unit = 'eV',
            **kwargs
        )

    def dipole_moment_vs_frequency(self,
                                   gauge: str = 'length',
                                   first_time: Optional[float] = None,
                                   last_time: Optional[float] = None):
        logger.critical('ALERT: dipole_momentum_vs_frequency does not account for non-uniform time step!')

        if first_time is None:
            first_time_index, first_time = 0, self.times[0]
        else:
            first_time_index, first_time, _ = si.utils.find_nearest_entry(self.times, first_time)
        if last_time is None:
            last_time_index, last_time = self.time_steps - 1, self.times[self.time_steps - 1]
        else:
            last_time_index, last_time, _ = si.utils.find_nearest_entry(self.times, last_time)
        points = last_time_index - first_time_index
        frequency = nfft.fftshift(nfft.fftfreq(points, self.spec.time_step))
        dipole_moment = nfft.fftshift(nfft.fft(self.electric_dipole_moment_vs_time[gauge][first_time_index: last_time_index], norm = 'ortho'))

        return frequency, dipole_moment

    def plot_dipole_moment_vs_frequency(self,
                                        use_name: bool = False,
                                        gauge: str = 'length',
                                        frequency_range: float = 10000 * u.THz,
                                        first_time: Optional[float] = None,
                                        last_time: Optional[float] = None,
                                        **kwargs):
        prefix = self.file_name
        if use_name:
            prefix = self.name

        frequency, dipole_moment = self.dipole_moment_vs_frequency(gauge = gauge, first_time = first_time, last_time = last_time)

        si.vis.xy_plot(
            prefix + '__dipole_moment_vs_frequency',
            frequency, np.abs(dipole_moment) ** 2,
            x_unit_value = 'THz',
            y_unit_value = u.atomic_electric_dipole_moment ** 2,
            x_label = 'Frequency $f$',
            y_label = r'Dipole Moment $\left| d(\omega) \right|^2$ $\left( e^2 \, a_0^2 \right)$',
            x_lower_limit = 0,
            x_upper_limit = frequency_range,
            y_log_axis = True,
            **kwargs
        )

    def save(self, target_dir: Optional[str] = None, file_extension: str = '.sim', save_mesh: bool = False, **kwargs):
        """
        Atomically pickle the Simulation to {target_dir}/{self.file_name}.{file_extension}, and gzip it for reduced disk usage.

        :param target_dir: directory to save the Simulation to
        :param file_extension: file extension to name the Simulation with
        :param save_mesh: if True, save the mesh as well as the Simulation. If False, don't.
        :return: None
        """

        if not save_mesh:
            try:
                for state in self.spec.test_states:  # remove numeric eigenstate information
                    state.g = None

                mesh = self.mesh.copy()
                self.mesh = None

            except AttributeError:  # mesh is already None
                mesh = None

        if len(self.spec.animators) > 0:
            raise si.SimulacraException('Cannot pickle Simulation with Animators')

        out = super().save(target_dir = target_dir, file_extension = file_extension, **kwargs)

        if not save_mesh:
            self.mesh = mesh

        return out

    @classmethod
    def load(cls, file_path: str, initialize_mesh: bool = False):
        """Return a simulation loaded from the file_path."""
        sim = super().load(file_path)

        if initialize_mesh:
            sim.initialize_mesh()

        return sim


class MeshSpecification(si.Specification):
    """A base Specification for a Simulation with an electric field."""

    simulation_type = MeshSimulation
    mesh_type = meshes.QuantumMesh

    evolution_equations = si.utils.RestrictedValues({'LAG', 'HAM'})
    evolution_method = si.utils.RestrictedValues({'CN', 'SO', 'S'})
    evolution_gauge = si.utils.RestrictedValues({'LEN', 'VEL'})

    def __init__(self,
                 name: str,
                 test_mass: float = u.electron_mass_reduced,
                 test_charge: float = u.electron_charge,
                 initial_state: states.QuantumState = states.HydrogenBoundState(1, 0),
                 test_states: Iterable[states.QuantumState] = tuple(),
                 dipole_gauges = (),
                 internal_potential: potentials.PotentialEnergy = potentials.Coulomb(charge = u.proton_charge),
                 electric_potential: potentials.ElectricPotential = potentials.NoElectricPotential(),
                 electric_potential_dc_correction: bool = False,
                 mask: potentials.Mask = potentials.NoMask(),
                 evolution_method = 'SO',
                 evolution_equations = 'HAM',
                 evolution_gauge = 'LEN',
                 time_initial = 0 * u.asec,
                 time_final = 200 * u.asec,
                 time_step = 1 * u.asec,
                 checkpoints: bool = False,
                 checkpoint_every: datetime.timedelta = datetime.timedelta(hours = 1),
                 checkpoint_dir: Optional[str] = None,
                 animators: Iterable[anim.WavefunctionSimulationAnimator] = tuple(),
                 store_norm_diff_mask = False,
                 store_data_every: int = 1,
                 snapshot_times = (),
                 snapshot_indices = (),
                 snapshot_type = None,
                 snapshot_kwargs: Optional[dict] = None,
                 datastore_types: Iterable[data.Datastore] = data.DEFAULT_DATASTORES,
                 **kwargs):
        """
        Initialize an ElectricFieldSpecification instance from the given parameters.

        This class should generally not be instantiated directly - its subclasses contain information about the mesh that is necessary to create an actual ElectricFieldSimulation.

        :param name: the name of the Specification
        :param mesh_type: the type of QuantumMesh that will be used in the Simulation.
        :param test_mass: the mass of the test particle
        :param test_charge: the charge of the test particle
        :param initial_state: the initial QuantumState of the test particle
        :param test_states: a list of states to perform inner products with during time evolution
        :param dipole_gauges: a list of dipole gauges to check the expectation value of during time evolution
        :param internal_potential: the time-independent part of the potential the particle experiences
        :param electric_potential: the electric field (possibly time-dependent) that the particle experiences
        :param electric_potential_dc_correction: if True, perform DC correction on the electric field
        :param mask: a mask function to be applied to the QuantumMesh after every time step
        :param evolution_method: which evolution method to use. Options are 'CN' (Crank-Nicolson), 'SO' (Split-Operator), and 'S' (Spectral). Only certain options are available for certain meshes.
        :param evolution_equations: which form of the evolution equations to use. Options are 'L' (Lagrangian) and 'H' (Hamiltonian). Most meshes use 'H'.
        :param evolution_gauge: which form of the interaction operator to use. Options are 'L' (Length) or 'V' (Velocity).
        :param time_initial: the initial time
        :param time_final: the final time
        :param time_step: the time step
        :param minimum_time_final: the minimum final time the Simulation will run to (i.e., if this is longer than time_final, more time will be added to the Simulation)
        :param extra_time_step: the time step to use during extra time from minimum_final_time
        :param checkpoints: if True, a checkpoint file will be maintained during time evolution
        :param checkpoint_every: how many time steps to calculate between checkpoints
        :param checkpoint_dir: a directory path to store the checkpoint file in
        :param animators: a list of Animators which will be run during time evolution
        :param store_norm_diff_by_mask: if True, the Simulation will store the amount of norm removed by the mask at each data time step
        :param store_data_every: decimate by this is if >= 1, else store only initial and final
        :param snapshot_times:
        :param snapshot_indices:
        :param snapshot_types:
        :param kwargs:
        """
        super().__init__(name, **kwargs)

        self.test_mass = test_mass
        self.test_charge = test_charge
        self.initial_state = initial_state
        self.test_states = tuple(sorted(tuple(test_states)))  # consume input iterators
        if len(self.test_states) == 0:
            self.test_states = [self.initial_state]
        self.dipole_gauges = tuple(sorted(tuple(dipole_gauges)))

        self.internal_potential = internal_potential
        self.electric_potential = electric_potential
        self.electric_potential_dc_correction = electric_potential_dc_correction
        self.mask = mask

        self.evolution_method = evolution_method
        self.evolution_equations = evolution_equations
        self.evolution_gauge = evolution_gauge

        self.time_initial = time_initial
        self.time_final = time_final
        self.time_step = time_step

        self.checkpoints = checkpoints
        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir = checkpoint_dir

        self.animators = deepcopy(tuple(animators))

        self.store_norm_diff_mask = store_norm_diff_mask

        self.store_data_every = int(store_data_every)

        self.snapshot_times = set(snapshot_times)
        self.snapshot_indices = set(snapshot_indices)
        if snapshot_type is None:
            snapshot_type = snapshots.Snapshot
        self.snapshot_type = snapshot_type
        if snapshot_kwargs is None:
            snapshot_kwargs = dict()
        self.snapshot_kwargs = snapshot_kwargs

        self.datastore_types = datastore_types

    def info(self) -> si.Info:
        info = super().info()

        info_checkpoint = si.Info(header = 'Checkpointing')
        if self.checkpoints:
            if self.checkpoint_dir is not None:
                working_in = self.checkpoint_dir
            else:
                working_in = 'cwd'
            info_checkpoint.header += ': every {} time steps, working in {}'.format(self.checkpoint_every, working_in)
        else:
            info_checkpoint.header += ': disabled'

        info.add_info(info_checkpoint)

        info_animation = si.Info(header = 'Animation')
        if len(self.animators) > 0:
            for animator in self.animators:
                info_animation.add_info(animator.info())
        else:
            info_animation.header += ': none'

        info.add_info(info_animation)

        info_evolution = si.Info(header = 'Time Evolution')
        info_evolution.add_field('Initial State', str(self.initial_state))
        info_evolution.add_field('Initial Time', f'{u.uround(self.time_initial, u.asec, 3)} as | {u.uround(self.time_initial, u.fsec, 3)} fs | {u.uround(self.time_initial, u.atomic_time, 3)} a.u.')
        info_evolution.add_field('Final Time', f'{u.uround(self.time_final, u.asec, 3)} as | {u.uround(self.time_final, u.fsec, 3)} fs | {u.uround(self.time_final, u.atomic_time, 3)} a.u.')
        if not callable(self.time_step):
            info_evolution.add_field('Time Step', f'{u.uround(self.time_step, u.asec, 3)} as | {u.uround(self.time_step, u.atomic_time, 3)} a.u.')
        else:
            info_evolution.add_field('Time Step', f'determined by {self.time_step}')

        info.add_info(info_evolution)

        info_algorithm = si.Info(header = 'Evolution Algorithm')
        info_algorithm.add_field('Evolution Equations', self.evolution_equations)
        info_algorithm.add_field('Evolution Method', self.evolution_method)
        info_algorithm.add_field('Evolution Gauge', self.evolution_gauge)

        info.add_info(info_algorithm)

        info_potentials = si.Info(header = 'Potentials and Masks')
        info_potentials.add_field('DC Correct Electric Field', 'yes' if self.electric_potential_dc_correction else 'no')
        for x in itertools.chain(self.internal_potential, self.electric_potential, self.mask):
            info_potentials.add_info(x.info())

        info.add_info(info_potentials)

        info_analysis = si.Info(header = 'Analysis')

        info_test_particle = si.Info(header = 'Test Particle')
        info_test_particle.add_field('Charge', f'{u.uround(self.test_charge, u.proton_charge, 3)} e')
        info_test_particle.add_field('Mass', f'{u.uround(self.test_mass, u.electron_mass, 3)} m_e | {u.uround(self.test_mass, u.electron_mass_reduced, 3)} mu_e')
        info_analysis.add_info(info_test_particle)

        if len(self.test_states) > 10:
            info_analysis.add_field(f'Test States (first 5 of {len(self.test_states)})', ', '.join(str(s) for s in sorted(self.test_states)[:5]))
        else:
            info_analysis.add_field('Test States', ', '.join(str(s) for s in sorted(self.test_states)))

        info_analysis.add_field('Datastores', ', '.join(ds.__name__ for ds in self.datastore_types))

        info_analysis.add_field('Data Storage Decimation', self.store_data_every)
        info_analysis.add_field('Snapshot Indices', ', '.join(sorted(self.snapshot_indices)) if len(self.snapshot_indices) > 0 else 'none')
        info_analysis.add_field('Snapshot Times', (f'{u.uround(st, u.asec, 3)} as' for st in self.snapshot_times) if len(self.snapshot_times) > 0 else 'none')

        info.add_info(info_analysis)

        return info


class LineSpecification(MeshSpecification):
    def __init__(self,
                 name,
                 initial_state = states.QHOState(1 * u.N / u.m),
                 x_bound = 10 * u.nm,
                 x_points = 2 ** 9,
                 fft_cutoff_energy = 1000 * u.eV,
                 analytic_eigenstate_type = None,
                 use_numeric_eigenstates = False,
                 number_of_numeric_eigenstates = 100,
                 **kwargs):
        super().__init__(
            name,
            mesh_type = meshes.LineMesh,
            initial_state = initial_state,
            **kwargs
        )

        self.x_bound = x_bound
        self.x_points = int(x_points)

        self.fft_cutoff_energy = fft_cutoff_energy
        self.fft_cutoff_wavenumber = np.sqrt(2 * self.test_mass * self.fft_cutoff_energy) / u.hbar

        self.analytic_eigenstate_type = analytic_eigenstate_type
        self.use_numeric_eigenstates = use_numeric_eigenstates
        self.number_of_numeric_eigenstates = number_of_numeric_eigenstates

    def info(self):
        info = super().info()

        info_mesh = si.Info(header = f'Mesh: {self.mesh_type.__name__}')
        info_mesh.add_field('X Boundary', f'{u.uround(self.x_bound, u.bohr_radius, 3)} a_0 | {u.uround(self.x_bound, u.nm, 3)} nm')
        info_mesh.add_field('X Points', self.x_points)
        info_mesh.add_field('X Mesh Spacing', f'~{u.uround(self.x_bound / self.x_points, u.bohr_radius, 3)} a_0 | {u.uround(self.x_bound / self.x_points, u.nm, 3)} nm')

        info.add_info(info_mesh)

        info_eigenstates = si.Info(header = f'Numeric Eigenstates: {self.use_numeric_eigenstates}')
        if self.use_numeric_eigenstates:
            info_eigenstates.add_field('Number of Numeric Eigenstates', self.number_of_numeric_eigenstates)

        info.add_info(info_eigenstates)

        return info


class CylindricalSliceSpecification(MeshSpecification):
    mesh_type = meshes.CylindricalSliceMesh

    def __init__(self,
                 name: str,
                 z_bound: float = 20 * u.bohr_radius,
                 rho_bound: float = 20 * u.bohr_radius,
                 z_points: int = 2 ** 9,
                 rho_points: int = 2 ** 8,
                 evolution_equations = 'HAM',
                 evolution_method = 'CN',
                 evolution_gauge = 'LEN',
                 **kwargs):
        super().__init__(
            name,
            evolution_equations = evolution_equations,
            evolution_method = evolution_method,
            evolution_gauge = evolution_gauge,
            **kwargs
        )

        self.z_bound = z_bound
        self.rho_bound = rho_bound
        self.z_points = int(z_points)
        self.rho_points = int(rho_points)

    def info(self) -> si.Info:
        info = super().info()

        info_mesh = si.Info(header = f'Mesh: {self.mesh_type.__name__}')
        info_mesh.add_field('Z Boundary', f'{u.uround(self.z_bound, u.bohr_radius, 3)} a_0')
        info_mesh.add_field('Z Points', self.z_points)
        info_mesh.add_field('Z Mesh Spacing', f'~{u.uround(self.z_bound / self.z_points, u.bohr_radius, 3)} a_0')
        info_mesh.add_field('Rho Boundary', f'{u.uround(self.rho_bound, u.bohr_radius, 3)} a_0')
        info_mesh.add_field('Rho Points', self.rho_points)
        info_mesh.add_field('Rho Mesh Spacing', f'~{u.uround(self.rho_bound / self.rho_points, u.bohr_radius, 3)} a_0')
        info_mesh.add_field('Total Mesh Points', int(self.z_points * self.rho_points))

        info.add_info(info_mesh)

        return info


class WarpedCylindricalSliceSpecification(MeshSpecification):
    mesh_type = meshes.WarpedCylindricalSliceMesh

    def __init__(self,
                 name: str,
                 z_bound: float = 20 * u.bohr_radius,
                 rho_bound: float = 20 * u.bohr_radius,
                 z_points: int = 2 ** 9,
                 rho_points: int = 2 ** 8,
                 evolution_equations = 'HAM',
                 evolution_method = 'CN',
                 evolution_gauge = 'LEN',
                 warping: float = 1,
                 **kwargs):
        super().__init__(name,
                         evolution_equations = evolution_equations,
                         evolution_method = evolution_method,
                         evolution_gauge = evolution_gauge,
                         **kwargs)

        self.z_bound = z_bound
        self.rho_bound = rho_bound
        self.z_points = int(z_points)
        self.rho_points = int(rho_points)

        self.warping = warping

    def info(self):
        info = super().info()

        info_mesh = si.Info(header = f'Mesh: {self.mesh_type.__name__}')
        info_mesh.add_field('Z Boundary', f'{u.uround(self.z_bound, u.bohr_radius, 3)} a_0')
        info_mesh.add_field('Z Points', self.z_points)
        info_mesh.add_field('Z Mesh Spacing', f'~{u.uround(self.z_bound / self.z_points, u.bohr_radius, 3)} a_0')
        info_mesh.add_field('Rho Boundary', f'{u.uround(self.rho_bound, u.bohr_radius, 3)} a_0')
        info_mesh.add_field('Rho Points', self.rho_points)
        info_mesh.add_field('Rho Mesh Spacing', f'~{u.uround(self.rho_bound / self.rho_points, u.bohr_radius, 3)} a_0')
        info_mesh.add_field('Rho Warping', self.warping)
        info_mesh.add_field('Total Mesh Points', int(self.z_points * self.rho_points))

        info.add_info(info_mesh)

        return info


class SphericalSliceSpecification(MeshSpecification):
    mesh_type = meshes.SphericalSliceMesh

    def __init__(self,
                 name: str,
                 r_bound: float = 20 * u.bohr_radius,
                 r_points: int = 2 ** 10,
                 theta_points: int = 2 ** 10,
                 evolution_equations = 'HAM',
                 evolution_method = 'CN',
                 evolution_gauge = 'LEN',
                 **kwargs):
        super().__init__(
            name,
            evolution_equations = evolution_equations,
            evolution_method = evolution_method,
            evolution_gauge = evolution_gauge,
            **kwargs
        )

        self.r_bound = r_bound

        self.r_points = int(r_points)
        self.theta_points = int(theta_points)

    def info(self):
        info = super().info()

        info_mesh = si.Info(header = f'Mesh: {self.mesh_type.__name__}')
        info_mesh.add_field('R Boundary', f'{u.uround(self.r_bound, u.bohr_radius, 3)} a_0')
        info_mesh.add_field('R Points', self.r_points)
        info_mesh.add_field('R Mesh Spacing', f'~{u.uround(self.r_bound / self.r_points, u.bohr_radius, 3)} a_0')
        info_mesh.add_field('Theta Points', self.theta_points)
        info_mesh.add_field('Theta Mesh Spacing', f'~{u.uround(u.pi / self.theta_points, u.bohr_radius, 3)} a_0')
        info_mesh.add_field('Maximum Adjacent-Point Spacing', f'~{u.uround(u.pi * self.r_bound / self.theta_points, u.bohr_radius, 3)} a_0')
        info_mesh.add_field('Total Mesh Points', int(self.r_points * self.theta_points))

        info.add_info(info_mesh)

        return info


class SphericalHarmonicSimulation(MeshSimulation):
    """Adds options and data storage that are specific to SphericalHarmonicMesh-using simulations."""

    def __init__(self, spec: 'SphericalHarmonicSpecification'):
        super().__init__(spec)

        if self.spec.store_norm_by_l:
            self.norm_by_harmonic_vs_time = {sph_harm: np.zeros(self.data_time_steps, dtype = np.float64) * np.NaN for sph_harm in self.spec.spherical_harmonics}

        if self.spec.store_radial_probability_current:
            self.radial_probability_current_vs_time__pos_z = np.zeros((self.data_time_steps, self.spec.r_points), dtype = np.float64) * np.NaN
            self.radial_probability_current_vs_time__neg_z = np.zeros((self.data_time_steps, self.spec.r_points), dtype = np.float64) * np.NaN

    def store_data(self):
        super().store_data()

        if self.spec.store_norm_by_l:
            norm_by_l = self.mesh.norm_by_l
            for sph_harm, l_norm in zip(self.spec.spherical_harmonics, norm_by_l):
                self.norm_by_harmonic_vs_time[sph_harm][self.data_time_index] = l_norm

            norm_in_largest_l = self.norm_by_harmonic_vs_time[self.spec.spherical_harmonics[-1]][self.data_time_index]
        else:
            largest_l_mesh = self.mesh.g[-1]
            norm_in_largest_l = self.mesh.state_overlap(largest_l_mesh, largest_l_mesh)

        if norm_in_largest_l > self.norm_vs_time[self.data_time_index] / 1e9:
            msg = f'Wavefunction norm in largest angular momentum state is large at time index {self.time_index} (norm at bound = {norm_in_largest_l}, fraction of norm = {norm_in_largest_l / self.norm_vs_time[self.data_time_index]}), consider increasing l bound'
            logger.warning(msg)
            self.warnings['norm_in_largest_l'].append(core.warning_record(self.time_index, msg))

        if self.spec.store_radial_probability_current:
            radial_current_density = self.mesh.get_radial_probability_current_density_mesh__spatial()

            theta = self.mesh.theta_calc
            d_theta = np.abs(theta[1] - theta[0])
            sin_theta = np.sin(theta)
            mask = theta <= u.pi / 2

            integrand = radial_current_density * sin_theta * d_theta * u.twopi  # sin(theta) d_theta from theta integral, twopi from phi integral

            self.radial_probability_current_vs_time__pos_z[self.data_time_index] = np.sum(integrand[:, mask], axis = 1) * (self.mesh.r ** 2)
            self.radial_probability_current_vs_time__neg_z[self.data_time_index] = np.sum(integrand[:, ~mask], axis = 1) * (self.mesh.r ** 2)

    @property
    def radial_probability_current_vs_time(self) -> np.array:
        return self.radial_probability_current_vs_time__pos_z + self.radial_probability_current_vs_time__neg_z

    def plot_radial_probability_current_vs_time(
            self,
            time_unit: str = 'asec',
            time_lower_limit: Optional[float] = None,
            time_upper_limit: Optional[float] = None,
            r_lower_limit: Optional[float] = None,
            r_upper_limit: Optional[float] = None,
            distance_unit: str = 'bohr_radius',
            z_unit: str = 'per_asec',
            z_limit: Optional[float] = None,
            use_name: bool = False,
            which: str = 'sum',
            **kwargs):
        if which == 'sum':
            z = self.radial_probability_current_vs_time
        elif which == 'pos':
            z = self.radial_probability_current_vs_time__pos_z
        elif which == 'neg':
            z = self.radial_probability_current_vs_time__neg_z
        else:
            raise AttributeError("which must be one of 'sum', 'pos', or 'neg'")

        prefix = self.file_name
        if use_name:
            prefix = self.name

        if z_limit is None:
            z_limit = np.nanmax(np.abs(self.radial_probability_current_vs_time))

        if time_lower_limit is None:
            time_lower_limit = self.data_times[0]
        if time_upper_limit is None:
            time_upper_limit = self.data_times[-1]

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

        t_mesh, r_mesh = np.meshgrid(self.data_times, r, indexing = 'ij')

        si.vis.xyz_plot(
            prefix + f'__radial_probability_current_{which}_vs_time',
            t_mesh,
            r_mesh,
            z,
            x_label = r'Time $t$', x_unit = time_unit,
            x_lower_limit = time_lower_limit, x_upper_limit = time_upper_limit,
            y_label = r'Radius $r$', y_unit = distance_unit,
            y_lower_limit = r_lower_limit, y_upper_limit = r_upper_limit,
            z_unit = z_unit,
            z_lower_limit = -z_limit, z_upper_limit = z_limit,
            z_label = r'$J_r$',
            colormap = plt.get_cmap('RdBu_r'),
            title = rf'Radial Probability Current vs. Time ({which})',
            **kwargs,
        )

    def plot_radial_probability_current_vs_time__combined(
            self,
            r_upper_limit: Optional[float] = None,
            t_lower_limit: Optional[float] = None,
            t_upper_limit: Optional[float] = None,
            distance_unit: str = 'bohr_radius',
            time_unit: str = 'asec',
            current_unit: str = 'per_asec',
            z_cut: float = .7,
            colormap = plt.get_cmap('coolwarm'),
            overlay_electric_field: bool = True,
            efield_unit: str = 'atomic_electric_field',
            efield_color: str = 'black',
            efield_label_fontsize: float = 12,
            title_fontsize: float = 12,
            y_axis_label_fontsize: float = 14,
            x_axis_label_fontsize: float = 12,
            cbar_label_fontsize: float = 12,
            aspect_ratio: float = 1.2,
            shading: str = 'flat',
            use_name: bool = False,
            **kwargs):
        prefix = self.file_name
        if use_name:
            prefix = self.name

        distance_unit_value, distance_unit_latex = u.get_unit_value_and_latex_from_unit(distance_unit)
        time_unit_value, time_unit_latex = u.get_unit_value_and_latex_from_unit(time_unit)
        current_unit_value, current_unit_latex = u.get_unit_value_and_latex_from_unit(current_unit)
        efield_unit_value, efield_unit_latex = u.get_unit_value_and_latex_from_unit(efield_unit)

        if t_lower_limit is None:
            t_lower_limit = self.data_times[0]
        if t_upper_limit is None:
            t_upper_limit = self.data_times[-1]

        with si.vis.FigureManager(prefix + '__radial_probability_current_vs_time__combined', aspect_ratio = aspect_ratio, **kwargs) as figman:
            fig = figman.fig

            plt.set_cmap(colormap)

            gridspec = plt.GridSpec(2, 1, hspace = 0.0)
            ax_pos = fig.add_subplot(gridspec[0])
            ax_neg = fig.add_subplot(gridspec[1], sharex = ax_pos)

            # TICKS, LEGEND, LABELS, and TITLE
            ax_pos.tick_params(labeltop = True, labelright = False, labelbottom = False, labelleft = True, bottom = False, right = False)
            ax_neg.tick_params(labeltop = False, labelright = False, labelbottom = True, labelleft = True, top = False, right = False)

            # pos_label = ax_pos.set_ylabel(f"$ r, \; z > 0 \; ({distance_unit_latex}) $", fontsize = y_axis_label_fontsize)
            # neg_label = ax_neg.set_ylabel(f"$ -r, \; z < 0 \; ({distance_unit_latex}) $", fontsize = y_axis_label_fontsize)
            pos_label = ax_pos.set_ylabel(f"$ z > 0 $", fontsize = y_axis_label_fontsize)
            neg_label = ax_neg.set_ylabel(f"$ z < 0 $", fontsize = y_axis_label_fontsize)
            ax_pos.yaxis.set_label_coords(-0.12, .65)
            ax_neg.yaxis.set_label_coords(-0.12, .35)
            r_label = ax_pos.text(-.22, .325, fr'Radius $ \pm r \; ({distance_unit_latex}) $', fontsize = y_axis_label_fontsize, rotation = 'vertical', transform = ax_pos.transAxes)
            ax_neg.set_xlabel(rf"Time $ t \; ({time_unit_latex}) $", fontsize = x_axis_label_fontsize)
            suptitle = fig.suptitle('Radial Probability Current vs. Time and Radius', fontsize = title_fontsize)
            suptitle.set_x(.6)
            suptitle.set_y(1.01)

            # COLORMESHES
            try:
                r = self.mesh.r
            except AttributeError:
                r = np.linspace(0, self.spec.r_bound, self.spec.r_points)
                delta_r = r[1] - r[0]
                r += delta_r / 2

            t_mesh, r_mesh = np.meshgrid(self.data_times, r, indexing = 'ij')

            # slicer = (slice(), slice(0, 50, 1))

            z_max = max(np.nanmax(np.abs(self.radial_probability_current_vs_time__pos_z)), np.nanmax(np.abs(self.radial_probability_current_vs_time__neg_z)))
            norm = matplotlib.colors.Normalize(vmin = -z_cut * z_max / current_unit_value, vmax = z_cut * z_max / current_unit_value)

            pos_mesh = ax_pos.pcolormesh(
                t_mesh / time_unit_value,
                r_mesh / distance_unit_value,
                self.radial_probability_current_vs_time__pos_z / current_unit_value,
                norm = norm,
                shading = shading,
            )
            neg_mesh = ax_neg.pcolormesh(
                t_mesh / time_unit_value,
                -r_mesh / distance_unit_value,
                self.radial_probability_current_vs_time__neg_z / current_unit_value,
                norm = norm,
                shading = shading,
            )

            # LIMITS AND GRIDS
            grid_kwargs = si.vis.GRID_KWARGS
            for ax in [ax_pos, ax_neg]:
                ax.set_xlim(t_lower_limit / time_unit_value, t_upper_limit / time_unit_value)
                ax.grid(True, which = 'major', **grid_kwargs)

            if r_upper_limit is None:
                r_upper_limit = r[-1]
            ax_pos.set_ylim(0, r_upper_limit / distance_unit_value)
            ax_neg.set_ylim(-r_upper_limit / distance_unit_value, 0)

            y_ticks_neg = ax_neg.yaxis.get_major_ticks()
            y_ticks_neg[-1].label1.set_visible(False)

            # COLORBAR
            ax_pos_position = ax_pos.get_position()
            ax_neg_position = ax_neg.get_position()
            left, bottom, width, height = ax_neg_position.x0, ax_neg_position.y0, ax_neg_position.x1 - ax_neg_position.x0, ax_pos_position.y1 - ax_neg_position.y0
            ax_cbar = fig.add_axes([left + width + .175, bottom, .05, height])
            cbar = plt.colorbar(mappable = pos_mesh, cax = ax_cbar, extend = 'both')
            z_label = cbar.set_label(rf"Radial Probability Current $ J_r \; ({current_unit_latex}) $", fontsize = cbar_label_fontsize)

            # ELECTRIC FIELD OVERLAY
            if overlay_electric_field:
                ax_efield = fig.add_axes((left, bottom, width, height))

                ax_efield.tick_params(labeltop = False, labelright = True, labelbottom = False, labelleft = False,
                                      left = False, top = False, bottom = False, right = True)
                ax_efield.tick_params(axis = 'y', colors = efield_color)
                ax_efield.tick_params(axis = 'x', colors = efield_color)

                efield, = ax_efield.plot(
                    self.data_times / time_unit_value,
                    self.electric_field_amplitude_vs_time / efield_unit_value,
                    color = efield_color,
                    linestyle = '-',
                )

                efield_grid_kwargs = {**si.vis.GRID_KWARGS, **{'color': efield_color, 'linestyle': '--'}}
                ax_efield.yaxis.grid(True, **efield_grid_kwargs)

                max_efield = np.nanmax(np.abs(self.electric_field_amplitude_vs_time))

                ax_efield.set_xlim(t_lower_limit / time_unit_value, t_upper_limit / time_unit_value)
                ax_efield.set_ylim(-1.05 * max_efield / efield_unit_value, 1.05 * max_efield / efield_unit_value)
                ax_efield.set_ylabel(rf'Electric Field Amplitude $ {vis.LATEX_EFIELD}(t) \; ({efield_unit_latex}) $', color = efield_color, fontsize = efield_label_fontsize)
                ax_efield.yaxis.set_label_position('right')

    def plot_angular_momentum_vs_time(self, use_name: bool = False, log: bool = False, renormalize: bool = False, **kwargs):
        fig = plt.figure(figsize = (7, 7 * 2 / 3), dpi = 600)

        grid_spec = matplotlib.gridspec.GridSpec(2, 1, height_ratios = [4, 1], hspace = 0.06)
        ax_momentums = plt.subplot(grid_spec[0])
        ax_field = plt.subplot(grid_spec[1], sharex = ax_momentums)

        if not isinstance(self.spec.electric_potential, potentials.NoPotentialEnergy):
            ax_field.plot(self.times / u.asec, self.electric_field_amplitude_vs_time / u.atomic_electric_field, color = 'black', linewidth = 2)

        if renormalize:
            overlaps = [self.norm_by_harmonic_vs_time[sph_harm] / self.norm_vs_time for sph_harm in self.spec.spherical_harmonics]
            l_labels = [r'$\left| \left\langle \Psi| {} \right\rangle \right|^2 / \left\langle \psi| \psi \right\rangle$'.format(sph_harm.latex) for sph_harm in self.spec.spherical_harmonics]
        else:
            overlaps = [self.norm_by_harmonic_vs_time[sph_harm] for sph_harm in self.spec.spherical_harmonics]
            l_labels = [r'$\left| \left\langle \Psi| {} \right\rangle \right|^2$'.format(sph_harm.latex) for sph_harm in self.spec.spherical_harmonics]
        num_colors = len(overlaps)
        ax_momentums.set_prop_cycle(cycler('color', [plt.get_cmap('gist_rainbow')(n / num_colors) for n in range(num_colors)]))
        ax_momentums.stackplot(self.times / u.asec, *overlaps, alpha = 1, labels = l_labels)

        if log:
            ax_momentums.set_yscale('log')
            ax_momentums.set_ylim(top = 1.0)
            ax_momentums.grid(True, which = 'both')
        else:
            ax_momentums.set_ylim(0, 1.0)
            ax_momentums.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            ax_momentums.grid(True)
        ax_momentums.set_xlim(self.spec.time_initial / u.asec, self.spec.time_final / u.asec)

        ax_field.grid(True)

        ax_field.set_xlabel('Time $t$ (as)', fontsize = 15)
        y_label = r'$\left| \left\langle \Psi | Y^l_0 \right\rangle \right|^2$'
        if renormalize:
            y_label += r'$/\left\langle \Psi|\Psi \right\rangle$'
        ax_momentums.set_ylabel(y_label, fontsize = 15)
        ax_field.set_ylabel('${}(t)$ (a.u.)'.format(vis.LATEX_EFIELD), fontsize = 11)

        ax_momentums.legend(bbox_to_anchor = (1.1, 1), loc = 'upper left', borderaxespad = 0., fontsize = 10, ncol = 1 + (len(self.spec.spherical_harmonics) // 17))

        ax_momentums.tick_params(labelright = True)
        ax_field.tick_params(labelright = True)
        ax_momentums.xaxis.tick_top()

        plt.rcParams['xtick.major.pad'] = 5
        plt.rcParams['ytick.major.pad'] = 5

        # Find at most n+1 ticks on the y-axis at 'nice' locations
        max_yticks = 6
        yloc = plt.MaxNLocator(max_yticks, prune = 'upper')
        ax_field.yaxis.set_major_locator(yloc)

        max_xticks = 6
        xloc = plt.MaxNLocator(max_xticks, prune = 'both')
        ax_field.xaxis.set_major_locator(xloc)

        ax_field.tick_params(axis = 'x', which = 'major', labelsize = 10)
        ax_field.tick_params(axis = 'y', which = 'major', labelsize = 10)
        ax_momentums.tick_params(axis = 'both', which = 'major', labelsize = 10)

        postfix = ''
        if renormalize:
            postfix += '_renorm'
        prefix = self.file_name
        if use_name:
            prefix = self.name
        si.vis.save_current_figure(name = prefix + '__angular_momentum_vs_time{}'.format(postfix), **kwargs)

        plt.close()


class SphericalHarmonicSpecification(MeshSpecification):
    simulation_type = SphericalHarmonicSimulation
    mesh_type = meshes.SphericalHarmonicMesh

    def __init__(self,
                 name: str,
                 r_bound: float = 100 * u.bohr_radius,
                 r_points: int = 400,
                 l_bound: int = 100,
                 theta_points: int = 180,
                 evolution_equations = 'LAG',
                 evolution_method = 'SO',
                 evolution_gauge = 'LEN',
                 use_numeric_eigenstates: bool = False,
                 numeric_eigenstate_max_angular_momentum: float = 20,
                 numeric_eigenstate_max_energy: float = 100 * u.eV,
                 hydrogen_zero_angular_momentum_correction: bool = True,
                 store_radial_probability_current: bool = False,
                 store_norm_by_l: bool = False,
                 **kwargs):
        """
        Specification for an ElectricFieldSimulation using a SphericalHarmonicMesh.

        :param name:
        :param r_bound:
        :param r_points:
        :param l_bound:
        :param evolution_equations: 'L' (recommended) or 'H'
        :param evolution_method: 'SO' (recommended) or 'CN'
        :param evolution_gauge: 'V' (recommended) or 'L'
        :param kwargs: passed to ElectricFieldSpecification
        """
        super().__init__(
            name,
            evolution_equations = evolution_equations,
            evolution_method = evolution_method,
            evolution_gauge = evolution_gauge,
            **kwargs
        )

        self.r_bound = r_bound
        self.r_points = int(r_points)
        self.l_bound = l_bound
        self.theta_points = theta_points
        self.spherical_harmonics = tuple(si.math.SphericalHarmonic(l, 0) for l in range(self.l_bound))

        self.use_numeric_eigenstates = use_numeric_eigenstates
        self.numeric_eigenstate_max_angular_momentum = min(self.l_bound - 1, numeric_eigenstate_max_angular_momentum)
        self.numeric_eigenstate_max_energy = numeric_eigenstate_max_energy

        self.hydrogen_zero_angular_momentum_correction = hydrogen_zero_angular_momentum_correction

        self.store_radial_probability_current = store_radial_probability_current
        self.store_norm_by_l = store_norm_by_l

    def info(self) -> si.Info:
        info = super().info()

        info_mesh = si.Info(header = f'Mesh: {self.mesh_type.__name__}')
        info_mesh.add_field('R Boundary', f'{u.uround(self.r_bound, u.bohr_radius, 3)} a_0 | {u.uround(self.r_bound, u.nm, 3)} nm')
        info_mesh.add_field('R Points', self.r_points)
        info_mesh.add_field('R Mesh Spacing', f'~{u.uround(self.r_bound / self.r_points, u.bohr_radius, 3)} a_0')
        info_mesh.add_field('L Bound', self.l_bound)
        info_mesh.add_field('Total Mesh Points', self.r_points * self.l_bound)

        info.add_info(info_mesh)

        info_eigenstates = si.Info(header = f'Numeric Eigenstates: {self.use_numeric_eigenstates}')
        if self.use_numeric_eigenstates:
            info_eigenstates.add_field('Max Energy', f'{u.uround(self.numeric_eigenstate_max_energy, u.eV)} eV')
            info_eigenstates.add_field('Max Angular Momentum', self.numeric_eigenstate_max_angular_momentum)

        info.add_info(info_eigenstates)

        return info
