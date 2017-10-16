import collections
import functools as ft
import itertools as it
import datetime
import logging
from copy import copy, deepcopy

import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np
import numpy.fft as nfft
import scipy as sp
import scipy.sparse as sparse
import scipy.sparse.linalg as sparsealg
import scipy.special as special
import scipy.integrate as integ
from scipy.misc import factorial
from tqdm import tqdm

import simulacra as si
from simulacra.units import *
from . import potentials, states

from .cy import tdma

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

LATEX_EFIELD = r'\mathcal{E}'
LATEX_AFIELD = r'\mathcal{A}'

COLOR_ELECTRIC_FIELD = si.vis.RED
COLOR_VECTOR_POTENTIAL = si.vis.BLUE

COLORMAP_WAVEFUNCTION = plt.get_cmap('inferno')

DEFAULT_RICHARDSON_MAGNITUDE_DIVISOR = 3


def electron_energy_from_wavenumber(k):
    return (hbar * k) ** 2 / (2 * electron_mass)


def electron_wavenumber_from_energy(energy):
    return np.sqrt(2 * electron_mass * energy + 0j) / hbar


def c_l(l):
    "a particular set of 3j coefficients for SphericalHarmonicMesh"
    return (l + 1) / np.sqrt(((2 * l) + 1) * ((2 * l) + 3))


def triangle_coef(a, b, c):
    return factorial(a + b - c) * factorial(a - b + c) * factorial(-a + b + c) / factorial(a + b + c + 1)


@si.utils.memoize
def cg_coef(j1, m1, j2, m2, j, m):
    """
    Return the Clebsch-Gordan coefficient <j1, j2; m1, m2 | j1, j2; j, m> using the Racah formula.

    See:
    Rudnicki-Bujnowski, G. Explicit formulas for Clebsch-Gordan coefficients. Comput. Phys. Commun. 10, 245–250 (1975)


    Parameters
    ----------
    j1
    m1
    j2
    m2
    j
    m

    Returns
    -------
    float
        A Clebsch-Gordan coefficient.
    """
    if m1 + m2 != m or not abs(j1 - j2) <= j <= j1 + j2:
        return 0

    cg = np.sqrt(((2 * j) + 1) * triangle_coef(j1, j2, j) * factorial(j1 + m1) * factorial(j1 - m1) * factorial(j2 + m2) * factorial(j2 - m2) * factorial(j + m) * factorial(j - m))

    t_min = int(max(-j + j2 - m, -j + j1 + m2, 0))
    t_max = int(min(j1 + j2 - j, j1 - m1, j2 + m2))

    s = 0
    for t in range(t_min, t_max + 1):
        s += ((-1) ** t) / (factorial(t) * factorial(j - j2 + t + m1) * factorial(j - j1 + t - m2) * factorial(j1 + j2 - j - t) * factorial(j1 - t - m1) * factorial(j2 - t + m2))

    return cg * s


@si.utils.memoize
def triple_y_integral(j1, m1, j2, m2, j, m):
    """
    j, m is whichever angular momentum is complex-conjugated in the integrand

    Parameters
    ----------
    j1
    m1
    j2
    m2
    j
    m

    Returns
    -------

    """
    if m1 + m2 != m or not abs(j1 - j2) <= j <= j1 + j2:
        return 0

    y1 = si.math.SphericalHarmonic(j1, m1)
    y2 = si.math.SphericalHarmonic(j2, m2)
    y3 = si.math.SphericalHarmonic(j, m)

    def integrand(theta, phi):
        return y1(theta, phi) * y2(theta, phi) * np.conj(y3(theta, phi)) * np.sin(theta)

    result = si.math.complex_nquad(integrand, [(0, pi), (0, twopi)], opts = {'limit': 1000})
    logger.debug(result)

    return np.real(result[0])


warning_record = collections.namedtuple('warning_record', ['data_time_index', 'message'])


class Snapshot:
    def __init__(self, simulation, time_index):
        self.sim = simulation
        self.spec = self.sim.spec
        self.time_index = time_index

        self.data = dict()

    @property
    def time(self):
        return self.sim.times[self.time_index]

    def __str__(self):
        return 'Snapshot of {} at time {} as (time index = {})'.format(self.sim.name, uround(self.sim.times[self.time_index], asec, 3), self.time_index)

    def __repr__(self):
        return si.utils.field_str(self, 'sim', 'time_index')

    def take_snapshot(self):
        self.collect_norm()

    def collect_norm(self):
        self.data['norm'] = self.sim.mesh.norm()


class Snapshot:
    def __init__(self, simulation, time_index):
        self.sim = simulation
        self.spec = self.sim.spec
        self.time_index = time_index

        self.data = dict()

    @property
    def time(self):
        return self.sim.times[self.time_index]

    def __str__(self):
        return 'Snapshot of {} at time {} as (time index = {})'.format(self.sim.name, uround(self.sim.times[self.time_index], asec, 3), self.time_index)

    def __repr__(self):
        return si.utils.field_str(self, 'sim', 'time_index')

    def take_snapshot(self):
        self.collect_norm()

    def collect_norm(self):
        self.data['norm'] = self.sim.mesh.norm()


class ElectricFieldSimulation(si.Simulation):
    def __init__(self, spec):
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

        self.inner_products_vs_time = {state: np.zeros(self.data_time_steps, dtype = np.complex128) * np.NaN for state in self.spec.test_states}

        self.electric_field_amplitude_vs_time = np.zeros(self.data_time_steps, dtype = np.float64) * np.NaN
        self.vector_potential_amplitude_vs_time = np.zeros(self.data_time_steps, dtype = np.float64) * np.NaN

        # optional data storage initialization
        if self.spec.store_radial_position_expectation_value:
            self.radial_position_expectation_value_vs_time = np.zeros(self.data_time_steps, dtype = np.float64) * np.NaN

        if self.spec.store_electric_dipole_moment_expectation_value:
            self.electric_dipole_moment_expectation_value_vs_time = np.zeros(self.data_time_steps, dtype = np.float64) * np.NaN

        if self.spec.store_energy_expectation_value:
            self.internal_energy_expectation_value_vs_time = np.zeros(self.data_time_steps, dtype = np.float64) * np.NaN
            self.total_energy_expectation_value_vs_time = np.zeros(self.data_time_steps, dtype = np.float64) * np.NaN

        if self.spec.store_norm_diff_mask:
            self.norm_diff_mask_vs_time = np.zeros(self.data_time_steps, dtype = np.float64) * np.NaN

        # populate the snapshot times from the two ways of entering snapshot times in the spec (by index or by time)
        self.snapshot_times = set()

        for time in self.spec.snapshot_times:
            time_index, time_target, _ = si.utils.find_nearest_entry(self.times, time)
            self.snapshot_times.add(time_target)

        for index in self.spec.snapshot_indices:
            self.snapshot_times.add(self.times[index])

        self.snapshots = dict()

        self.warnings = collections.defaultdict(list)

    def info(self):
        info = super().info()

        mem_mesh = self.mesh.g.nbytes if self.mesh is not None else 0

        mem_matrix_operators = 6 * mem_mesh
        mem_numeric_eigenstates = sum(state.g.nbytes for state in self.spec.test_states if state.numeric and state.g is not None)
        mem_inner_products = sum(overlap.nbytes for overlap in self.inner_products_vs_time.values())

        mem_other_time_data = sum(x.nbytes for x in (
            self.electric_field_amplitude_vs_time,
            self.vector_potential_amplitude_vs_time,
            self.norm_vs_time,
        ))

        for attr in (
                'radial_position_expectation_value_vs_time',
                'internal_energy_expectation_value_vs_time',
                'total_energy_expectation_value_vs_time',
                'electric_dipole_moment_expectation_value_vs_time'
                'norm_diff_mask_vs_time',
                'radial_probability_current_vs_time__pos_z',
                'radial_probability_current_vs_time__neg_z',
        ):
            try:
                mem_other_time_data += getattr(self, attr).nbytes
            except AttributeError:  # apparently we're not storing that data
                pass

        try:
            mem_other_time_data += sum(h.nbytes for h in self.norm_by_harmonic_vs_time.values())
        except AttributeError:
            pass

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
    def available_animation_frames(self):
        return self.time_steps

    @property
    def time(self):
        return self.times[self.time_index]

    @property
    def times_to_current(self):
        return self.times[:self.time_index + 1]

    @property
    def state_overlaps_vs_time(self):
        return {state: np.abs(inner_product) ** 2 for state, inner_product in self.inner_products_vs_time.items()}

    @property
    def total_overlap_vs_time(self):
        return np.sum(overlap for overlap in self.state_overlaps_vs_time.values())

    @property
    def total_bound_state_overlap_vs_time(self):
        return np.sum(overlap for state, overlap in self.state_overlaps_vs_time.items() if state.bound)

    @property
    def total_free_state_overlap_vs_time(self):
        return np.sum(overlap for state, overlap in self.state_overlaps_vs_time.items() if state.free)

    def get_times(self):
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
        norm = self.mesh.norm()
        self.norm_vs_time[self.data_time_index] = norm
        if norm > 1.001 * self.norm_vs_time[0]:
            logger.warning('Wavefunction norm ({}) has exceeded initial norm ({}) by more than .1% for {} {}'.format(norm, self.norm_vs_time[0], self.__class__.__name__, self.name))
        try:
            if norm > 1.001 * self.norm_vs_time[self.data_time_index - 1]:
                logger.warning('Wavefunction norm ({}) at time_index = {} has exceeded norm from previous time step ({}) by more than .1% for {} {}'.format(norm, self.data_time_index,
                                                                                                                                                            self.norm_vs_time[self.data_time_index - 1],
                                                                                                                                                            self.__class__.__name__, self.name))
        except IndexError:
            pass

        if self.spec.store_radial_position_expectation_value:
            self.radial_position_expectation_value_vs_time[self.data_time_index] = np.real(self.mesh.inner_product(b = self.mesh.r_mesh * self.mesh.g)) / self.mesh.norm()

        if self.spec.store_energy_expectation_value:
            self.internal_energy_expectation_value_vs_time[self.data_time_index] = self.mesh.energy_expectation_value(include_interaction = False)
            self.total_energy_expectation_value_vs_time[self.data_time_index] = self.mesh.energy_expectation_value(include_interaction = True)

        if self.spec.store_electric_dipole_moment_expectation_value:
            self.electric_dipole_moment_expectation_value_vs_time[self.data_time_index] = np.real(self.mesh.dipole_moment_inner_product())

        for state in self.spec.test_states:
            self.inner_products_vs_time[state][self.data_time_index] = self.mesh.inner_product(state)

        self.electric_field_amplitude_vs_time[self.data_time_index] = self.spec.electric_potential.get_electric_field_amplitude(t = self.data_times[self.data_time_index])
        self.vector_potential_amplitude_vs_time[self.data_time_index] = self.spec.electric_potential.get_vector_potential_amplitude_numeric(times = self.times_to_current)

        for callback in self.spec.store_data_callbacks:
            callback(self)

        logger.debug('{} {} stored data for time index {} (data time index {})'.format(self.__class__.__name__, self.name, self.time_index, self.data_time_index))

    def take_snapshot(self):
        snapshot = self.spec.snapshot_type(self, self.time_index, **self.spec.snapshot_kwargs)

        snapshot.take_snapshot()

        self.snapshots[self.time_index] = snapshot

        logger.info('Stored {} of {} at time {} as (time index {})'.format(snapshot.__class__.__name__, self.name, uround(self.time, asec, 3), self.time_index))

    def run_simulation(self, progress_bar = False, callback = None):
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
                        logger.info(f'{self.__class__.__name__} {self.name} ({self.file_name}) checkpointed at time index {self.time_index} / {self.time_steps - 1} ({np.around(100 * (self.time_index + 1) / self.time_steps, 2)}%)')
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
    def bound_states(self):
        yield from [s for s in self.spec.test_states if s.bound]

    @property
    def free_states(self):
        yield from [s for s in self.spec.test_states if not s.bound]

    def group_free_states_by_continuous_attr(self, attr = 'energy', divisions = 10, cutoff_value = None,
                                             label_format_str = r'\phi_{{    {} \; \mathrm{{to}} \; {} \, {}, \ell   }}', attr_unit = 'eV'):
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

        label_unit_value, label_unit_latex = get_unit_value_and_latex_from_unit(attr_unit)

        free_states = list(self.free_states)

        for ii, lower_boundary in enumerate(boundaries[:-1]):
            upper_boundary = boundaries[ii + 1]

            label = label_format_str.format(uround(lower_boundary, label_unit_value, 2), uround(upper_boundary, label_unit_value, 2), label_unit_latex)
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
                                               axis,
                                               show_electric_field = True,
                                               show_vector_potential = True,
                                               time_unit = 'asec',
                                               legend_kwargs = None,
                                               show_y_label = False, ):
        time_unit_value, time_unit_latex = get_unit_value_and_latex_from_unit(time_unit)

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
            axis.plot(self.data_times / time_unit_value, self.electric_field_amplitude_vs_time / atomic_electric_field,
                      color = COLOR_ELECTRIC_FIELD,
                      linewidth = 1.5,
                      label = fr'$ {LATEX_EFIELD}(t) $')
        if show_vector_potential:
            axis.plot(self.data_times / time_unit_value, proton_charge * self.vector_potential_amplitude_vs_time / atomic_momentum,
                      color = COLOR_VECTOR_POTENTIAL,
                      linewidth = 1.5,
                      label = fr'$ e \, {LATEX_AFIELD}(t) $')

        if show_y_label:
            axis.set_ylabel('${}(t)$'.format(LATEX_EFIELD), fontsize = 13, color = COLOR_ELECTRIC_FIELD)

        axis.set_xlabel('Time $t$ (${}$)'.format(time_unit_latex), fontsize = 13)

        axis.tick_params(labelright = True)

        axis.set_xlim(self.times[0] / time_unit_value, self.times[-1] / time_unit_value)

        axis.legend(**legend_kwargs)

        axis.grid(True, **si.vis.GRID_KWARGS)

    def plot_state_overlaps_vs_time(self,
                                    states = None,
                                    log = False,
                                    time_unit = 'asec',
                                    show_electric_field = True,
                                    show_vector_potential = True,
                                    **kwargs):
        with si.vis.FigureManager(name = f'{self.spec.name}', **kwargs) as figman:
            time_unit_value, time_unit_latex = get_unit_value_and_latex_from_unit(time_unit)

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

    def plot_wavefunction_vs_time(self, log = False, time_unit = 'asec',
                                  bound_state_max_n = 5,
                                  collapse_bound_state_angular_momenta = True,
                                  grouped_free_states = None,
                                  group_free_states_labels = None,
                                  show_title = False,
                                  plot_name_from = 'file_name',
                                  show_electric_field = True,
                                  show_vector_potential = True,
                                  **kwargs):
        with si.vis.FigureManager(name = getattr(self, plot_name_from) + '__wavefunction_vs_time', **kwargs) as figman:
            time_unit_value, time_unit_latex = get_unit_value_and_latex_from_unit(time_unit)

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

            free_state_color_cycle = it.cycle(['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5', '#ffed6f'])
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
                ncol = 1 + (len(overlaps) // 12))

            ax_overlaps.tick_params(labelleft = True,
                                    labelright = True,
                                    labeltop = True,
                                    labelbottom = False,
                                    bottom = True,
                                    top = True,
                                    left = True,
                                    right = True)
            ax_field.tick_params(labelleft = True,
                                 labelright = True,
                                 labeltop = False,
                                 labelbottom = True,
                                 bottom = True,
                                 top = True,
                                 left = True,
                                 right = True)

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
                             states = 'all',
                             time_index = -1,
                             energy_scale = 'eV',
                             time_scale = 'asec',
                             bins = 100,
                             log = False,
                             energy_lower_bound = None, energy_upper_bound = None,
                             group_angular_momentum = True, angular_momentum_cutoff = None,
                             **kwargs):
        energy_unit, energy_unit_str = get_unit_value_and_latex_from_unit(energy_scale)
        time_unit, time_unit_str = get_unit_value_and_latex_from_unit(time_scale)

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
            overlap_by_angular_momentum_by_energy = collections.defaultdict(ft.partial(collections.defaultdict, float))

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
            ax.set_title('Wavefunction Overlap by Energy at $t={} \, {}$'.format(uround(self.times[time_index], time_unit, 3), time_unit_str))

            if group_angular_momentum:
                ax.legend(loc = 'best', ncol = 1 + len(energies) // 8)

            ax.tick_params(axis = 'both', which = 'major', labelsize = 10)

            figman.name += '__{}_states__index={}'.format(states, time_index)

            if log:
                figman.name += '__log'
            if group_angular_momentum:
                figman.name += '__grouped'

    def plot_radial_position_expectation_value_vs_time(self, use_name = False, **kwargs):
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

    def plot_dipole_moment_expectation_value_vs_time(self, use_name = False, **kwargs):
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

    def plot_energy_expectation_value_vs_time(self, use_name = False, **kwargs):
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

    def dipole_moment_vs_frequency(self, gauge = 'length', first_time = None, last_time = None):
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

    def plot_dipole_moment_vs_frequency(self, use_name = False, gauge = 'length', frequency_range = 10000 * THz, first_time = None, last_time = None, **kwargs):
        prefix = self.file_name
        if use_name:
            prefix = self.name

        frequency, dipole_moment = self.dipole_moment_vs_frequency(gauge = gauge, first_time = first_time, last_time = last_time)

        si.vis.xy_plot(prefix + '__dipole_moment_vs_frequency',
                       frequency, np.abs(dipole_moment) ** 2,
                       x_unit_value = 'THz', y_unit_value = atomic_electric_dipole_moment ** 2,
                       y_log_axis = True,
                       x_label = 'Frequency $f$', y_label = r'Dipole Moment $\left| d(\omega) \right|^2$ $\left( e^2 \, a_0^2 \right)$',
                       x_lower_limit = 0, x_upper_limit = frequency_range,
                       **kwargs)

    def save(self, target_dir = None, file_extension = '.sim', save_mesh = False, **kwargs):
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
    def load(cls, file_path, initialize_mesh = False):
        """Return a simulation loaded from the file_path."""
        sim = super().load(file_path)

        if initialize_mesh:
            sim.initialize_mesh()

        return sim


class ElectricFieldSpecification(si.Specification):
    """A base Specification for a Simulation with an electric field."""

    simulation_type = ElectricFieldSimulation

    evolution_equations = si.utils.RestrictedValues('evolution_equations', {'LAG', 'HAM'})
    evolution_method = si.utils.RestrictedValues('evolution_method', {'CN', 'SO', 'S'})
    evolution_gauge = si.utils.RestrictedValues('evolution_gauge', {'LEN', 'VEL'})

    def __init__(self, name,
                 mesh_type = None,
                 test_mass = electron_mass_reduced, test_charge = electron_charge,
                 initial_state = states.HydrogenBoundState(1, 0),
                 test_states = tuple(),
                 dipole_gauges = (),
                 internal_potential = potentials.Coulomb(charge = proton_charge),
                 electric_potential = potentials.NoElectricPotential(),
                 electric_potential_dc_correction = False,
                 mask = potentials.NoMask(),
                 evolution_method = 'SO', evolution_equations = 'HAM', evolution_gauge = 'LEN',
                 time_initial = 0 * asec, time_final = 200 * asec, time_step = 1 * asec,
                 checkpoints = False, checkpoint_every = datetime.timedelta(hours = 1), checkpoint_dir = None,
                 animators = tuple(),
                 store_radial_position_expectation_value = True,
                 store_electric_dipole_moment_expectation_value = True,
                 store_energy_expectation_value = True,
                 store_norm_diff_mask = False,
                 store_data_callbacks = (),
                 store_data_every = 1,
                 snapshot_times = (), snapshot_indices = (), snapshot_type = None, snapshot_kwargs = None,
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

        if mesh_type is None:
            raise ValueError('{} must have a mesh_type'.format(name))
        self.mesh_type = mesh_type

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

        self.store_radial_position_expectation_value = store_radial_position_expectation_value
        self.store_electric_dipole_moment_expectation_value = store_electric_dipole_moment_expectation_value
        self.store_energy_expectation_value = store_energy_expectation_value
        self.store_norm_diff_mask = store_norm_diff_mask

        self.store_data_callbacks = store_data_callbacks

        self.store_data_every = int(store_data_every)

        self.snapshot_times = set(snapshot_times)
        self.snapshot_indices = set(snapshot_indices)
        if snapshot_type is None:
            snapshot_type = Snapshot
        self.snapshot_type = snapshot_type
        if snapshot_kwargs is None:
            snapshot_kwargs = dict()
        self.snapshot_kwargs = snapshot_kwargs

    def info(self):
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
        info_evolution.add_field('Initial Time', f'{uround(self.time_initial, asec, 3)} as | {uround(self.time_initial, fsec, 3)} fs | {uround(self.time_initial, atomic_time, 3)} a.u.')
        info_evolution.add_field('Final Time', f'{uround(self.time_final, asec, 3)} as | {uround(self.time_final, fsec, 3)} fs | {uround(self.time_final, atomic_time, 3)} a.u.')
        if not callable(self.time_step):
            info_evolution.add_field('Time Step', f'{uround(self.time_step, asec, 3)} as | {uround(self.time_step, atomic_time, 3)} a.u.')
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
        for x in it.chain(self.internal_potential, self.electric_potential, self.mask):
            info_potentials.add_info(x.info())

        info.add_info(info_potentials)

        info_analysis = si.Info(header = 'Analysis')
        info_analysis.add_field('Test Charge', f'{uround(self.test_charge, proton_charge, 3)} e')
        info_analysis.add_field('Test Mass', f'{uround(self.test_mass, electron_mass, 3)} m_e | {uround(self.test_mass, electron_mass_reduced, 3)} mu_e')
        if len(self.test_states) > 10:
            info_analysis.add_field(f'Test States (first 5 of {len(self.test_states)})', ', '.join(str(s) for s in sorted(self.test_states)[:5]))
        else:
            info_analysis.add_field('Test States', ', '.join(str(s) for s in sorted(self.test_states)))
        info_analysis.add_field('Store Radial Position EV', self.store_radial_position_expectation_value)
        info_analysis.add_field('Store Dipole Moment EV', self.store_electric_dipole_moment_expectation_value)
        info_analysis.add_field('Store Energy EV', self.store_energy_expectation_value)
        info_analysis.add_field('Store Radial Probability Current', self.store_radial_probability_current)
        info_analysis.add_field('Data Storage Decimation', self.store_data_every)
        info_analysis.add_field('Snapshot Indices', ', '.join(sorted(self.snapshot_indices)) if len(self.snapshot_indices) > 0 else 'none')
        info_analysis.add_field('Snapshot Times', (f'{uround(st, asec, 3)} as' for st in self.snapshot_times) if len(self.snapshot_times) > 0 else 'none')

        info.add_info(info_analysis)

        return info


def add_to_diagonal_sparse_matrix_diagonal(dia_matrix, value = 1):
    s = dia_matrix.copy()
    s.setdiag(s.diagonal() + value)
    return s


def add_to_diagonal_sparse_matrix_diagonal_inplace(dia_matrix, value = 1):
    # dia_matrix.setdiag(dia_matrix.diagonal() + value)
    dia_matrix.data[1] += value
    return dia_matrix


class MeshOperator:
    def __init__(self, operator, *, wrapping_direction):
        self.operator = operator
        self.wrapping_direction = wrapping_direction

    def __repr__(self):
        return f"{self.__class__.__name__}(operator = {repr(self.operator)}, wrapping_direction = '{self.wrapping_direction}')"

    def apply(self, mesh, g, current_wrapping_direction):
        if current_wrapping_direction != self.wrapping_direction:
            g = mesh.flatten_mesh(mesh.wrap_vector(g, current_wrapping_direction), self.wrapping_direction)

        result = self._apply(g)

        return result, self.wrapping_direction

    def _apply(self, g):
        raise NotImplementedError


class DotOperator(MeshOperator):
    def _apply(self, g):
        return self.operator.dot(g)


class TDMAOperator(MeshOperator):
    def _apply(self, g):
        return tdma(self.operator, g)


class SimilarityOperator(DotOperator):
    def __init__(self, operator, *, wrapping_direction, parity):
        super().__init__(operator, wrapping_direction = wrapping_direction)

        self.parity = parity
        self.transform = getattr(self, f'u_{self.parity}_g')

    def __repr__(self):
        op_repr = repr(self.operator).replace('\n', '')
        return f"{self.__class__.__name__}({op_repr}, wrapping_direction = '{self.wrapping_direction}', parity = '{self.parity}')"

    def u_even_g(self, g):
        stack = []
        if len(g) % 2 == 0:
            for a, b in si.utils.grouper(g, 2, fill_value = 0):
                stack += (a + b, a - b)
        else:
            for a, b in si.utils.grouper(g[:-1], 2, fill_value = 0):
                stack += (a + b, a - b)
            stack.append(np.sqrt(2) * g[-1])

        return np.hstack(stack) / np.sqrt(2)

    def u_odd_g(self, g):
        stack = [np.sqrt(2) * g[0]]
        if len(g) % 2 == 0:
            for a, b in si.utils.grouper(g[1:-1], 2, fill_value = 0):
                stack += (a + b, a - b)
            stack.append(np.sqrt(2) * g[-1])
        else:
            for a, b in si.utils.grouper(g[1:], 2, fill_value = 0):
                stack += (a + b, a - b)

        return np.hstack(stack) / np.sqrt(2)

    def apply(self, mesh, g, current_wrapping_direction):
        g_wrapped = mesh.wrap_vector(g, current_wrapping_direction)
        g_transformed = self.transform(g_wrapped)  # this wraps the mesh along j!
        g_flat = mesh.flatten_mesh(g_transformed, self.wrapping_direction)
        g_flat = self._apply(g_flat)
        g_wrap = mesh.wrap_vector(g_flat, self.wrapping_direction)
        result = self.transform(g_wrap)  # this wraps the mesh along j!

        return result, self.wrapping_direction


def apply_operators(mesh, g, *operators):
    """Operators should be entered in operation (the order they would act on something on their right)"""
    current_wrapping_direction = None

    for operator in operators:
        g, current_wrapping_direction = operator.apply(mesh, g, current_wrapping_direction)

    return mesh.wrap_vector(g, current_wrapping_direction)


class QuantumMesh:
    def __init__(self, simulation):
        self.sim = simulation
        self.spec = simulation.spec

        self.g = None
        self.inner_product_multiplier = None

        try:
            self.evolution_method = getattr(self, f'_evolve_{self.spec.evolution_method}')
        except AttributeError:
            raise NotImplementedError(f'Evolution method {self.spec.evolution_method} is not implemented for {self.__class__.__name__}')

    def __eq__(self, other):
        """
        Return whether the provided meshes are equal. QuantumMeshes should evaluate equal if and only if their Simulations are equal and their g (the only thing which carries state information) are the same.

        :param other:
        :return:
        """
        return isinstance(other, self.__class__) and self.sim == other.sim and np.array_equal(self.g, other.g)

    def __hash__(self):
        """Return the hash of the QuantumMesh, which is the same as the hash of the associated Simulation."""
        return hash(self.sim)

    def __str__(self):
        return '{} for {}'.format(self.__class__.__name__, str(self.sim))

    def __repr__(self):
        return '{}(parameters = {}, simulation = {})'.format(self.__class__.__name__, repr(self.spec), repr(self.sim))

    def get_g_for_state(self, state):
        raise NotImplementedError

    def state_to_mesh(self, state_or_mesh):
        """Return the mesh associated with the given state, or simply passes the mesh through."""
        if state_or_mesh is None:
            return self.g
        elif isinstance(state_or_mesh, states.QuantumState):
            try:
                state_or_mesh = self.analytic_to_numeric[state_or_mesh]
            except (AttributeError, KeyError):
                pass
            return self.get_g_for_state(state_or_mesh)
        else:
            return state_or_mesh

    def get_g_with_states_removed(self, states, g = None):
        """
        Get a g mesh with the contributions from the states removed.

        :param states: a list of states to remove from g
        :param g: a g to remove the state contributions from. Defaults to self.g
        :return:
        """
        if g is None:
            g = self.g

        g = g.copy()  # always act on a copy of g, regardless of source

        for state in states:
            g -= self.inner_product(state, g) * self.get_g_for_state(state)

        return g

    def inner_product(self, a = None, b = None):
        """Inner product between two meshes. If either mesh is None, the state on the g is used for that state."""
        return np.sum(np.conj(self.state_to_mesh(a)) * self.state_to_mesh(b)) * self.inner_product_multiplier

    def state_overlap(self, a = None, b = None):
        """State overlap between two states. If either state is None, the state on the g is used for that state."""
        return np.abs(self.inner_product(a, b)) ** 2

    def norm(self, state = None):
        return np.abs(self.inner_product(a = state, b = state))

    def energy_expectation_value(self):
        raise NotImplementedError

    def dipole_moment_inner_product(self, a = None, b = None):
        raise NotImplementedError

    def __abs__(self):
        return self.norm()

    def copy(self):
        return deepcopy(self)

    @property
    def psi(self):
        return self.g / self.g_factor

    @property
    def g2(self):
        return np.abs(self.g) ** 2

    @property
    def psi2(self):
        return np.abs(self.psi) ** 2

    @si.utils.memoize
    def get_kinetic_energy_matrix_operators(self):
        try:
            return getattr(self, f'_get_kinetic_energy_matrix_operators_{self.spec.evolution_equations}')()
        except AttributeError:
            raise NotImplementedError

    def get_internal_hamiltonian_matrix_operators(self):
        raise NotImplementedError

    def get_interaction_hamiltonian_matrix_operators(self):
        try:
            return getattr(self, f'_get_interaction_hamiltonian_matrix_operators_{self.spec.evolution_gauge}')()
        except AttributeError:
            raise NotImplementedError

    def evolve(self, time_step):
        if self.spec.store_norm_diff_mask:
            pre_evolve_norm = self.norm()
            self.evolution_method(time_step)
            norm_diff_evolve = pre_evolve_norm - self.norm()
            if norm_diff_evolve / pre_evolve_norm > .001:
                logger.warning('Evolution may be dangerously non-unitary, norm decreased by {} ({} %) during evolution step'.format(norm_diff_evolve, norm_diff_evolve / pre_evolve_norm))

            pre_mask_norm = self.norm()
            self.g *= self.spec.mask(r = self.r_mesh)
            norm_diff_by_mask = pre_mask_norm - self.norm()
            logger.debug('Applied mask {} to g for {} {}, removing {} norm'.format(self.spec.mask, self.sim.__class__.__name__, self.sim.name, norm_diff_by_mask))
            return norm_diff_by_mask
        else:
            self.evolution_method(time_step)
            if not isinstance(self.spec.mask, potentials.NoMask):
                self.g *= self.spec.mask(r = self.r_mesh)
                logger.debug('Applied mask {} to g for {} {}'.format(self.spec.mask, self.sim.__class__.__name__, self.sim.name))

    def get_mesh_slicer(self, plot_limit):
        raise NotImplementedError

    def attach_mesh_to_axis(self, axis, mesh,
                            distance_unit = 'bohr_radius',
                            colormap = plt.get_cmap('inferno'),
                            norm = si.vis.AbsoluteRenormalize(),
                            shading = 'flat',
                            plot_limit = None,
                            slicer = 'get_mesh_slicer',
                            **kwargs):
        raise NotImplementedError

    def attach_g2_to_axis(self, axis, **kwargs):
        return self.attach_mesh_to_axis(axis, self.g2, **kwargs)

    def attach_psi2_to_axis(self, axis, **kwargs):
        return self.attach_mesh_to_axis(axis, self.psi2, **kwargs)

    def attach_g_to_axis(self, axis,
                         colormap = plt.get_cmap('richardson'),
                         norm = None,
                         **kwargs):
        if norm is None:
            norm = si.vis.RichardsonNormalization(np.max(np.abs(self.g) / DEFAULT_RICHARDSON_MAGNITUDE_DIVISOR))

        return self.attach_mesh_to_axis(axis, self.g,
                                        colormap = colormap,
                                        norm = norm,
                                        **kwargs)

    def attach_psi_to_axis(self, axis,
                           colormap = plt.get_cmap('richardson'),
                           norm = None,
                           **kwargs):
        if norm is None:
            norm = si.vis.RichardsonNormalization(np.max(np.abs(self.psi) / DEFAULT_RICHARDSON_MAGNITUDE_DIVISOR))

        return self.attach_mesh_to_axis(axis, self.psi,
                                        colormap = colormap,
                                        norm = norm,
                                        **kwargs)

    def update_mesh(self, colormesh, updated_mesh,
                    plot_limit = None,
                    shading = 'flat',
                    slicer = 'get_mesh_slicer',
                    **kwargs
                    ):
        _slice = getattr(self, slicer)(plot_limit)
        updated_mesh = updated_mesh[_slice]

        try:
            if shading == 'flat':
                updated_mesh = updated_mesh[:-1, :-1]
            colormesh.set_array(updated_mesh.ravel())
        except AttributeError:  # if the mesh is 1D we can't .ravel() it and instead should just set the y data with the mesh
            colormesh.set_ydata(updated_mesh)

    def update_g2_mesh(self, colormesh, **kwargs):
        self.update_mesh(colormesh, self.g2, **kwargs)

    def update_psi2_mesh(self, colormesh, **kwargs):
        self.update_mesh(colormesh, self.psi2, **kwargs)

    def update_g_mesh(self, colormesh, **kwargs):
        self.update_mesh(colormesh, self.g, **kwargs)

    def update_psi_mesh(self, colormesh, **kwargs):
        self.update_mesh(colormesh, self.psi, **kwargs)

    def plot_mesh(self, mesh,
                  name = '',
                  title = None,
                  distance_unit = 'bohr_radius',
                  colormap = COLORMAP_WAVEFUNCTION,
                  norm = si.vis.AbsoluteRenormalize(),
                  shading = 'flat',
                  plot_limit = None,
                  slicer = 'get_mesh_slicer',
                  **kwargs):
        """kwargs go to figman"""
        raise NotImplementedError

    def plot_g2(self, name_postfix = '', title = None, **kwargs):
        if title is None:
            title = r'$|g|^2$'
        name = 'g2' + name_postfix

        self.plot_mesh(self.g2, name = name, title = title, **kwargs)

    def plot_psi2(self, name_postfix = '', **kwargs):
        title = r'$|\Psi|^2$'
        name = 'psi2' + name_postfix

        self.plot_mesh(self.psi2, name = name, title = title, **kwargs)

    def plot_g(self,
               title = None,
               name_postfix = '',
               colormap = plt.get_cmap('richardson'),
               norm = None,
               **kwargs):
        if title is None:
            title = r'$g$'
        name = 'g' + name_postfix

        if norm is None:
            norm = si.vis.RichardsonNormalization(np.max(np.abs(self.g) / DEFAULT_RICHARDSON_MAGNITUDE_DIVISOR))

        self.plot_mesh(self.g, name = name, title = title,
                       colormap = colormap,
                       norm = norm,
                       show_colorbar = False,
                       **kwargs)

    def plot_psi(self,
                 name_postfix = '',
                 colormap = plt.get_cmap('richardson'),
                 norm = None,
                 **kwargs):
        title = r'$g$'
        name = 'g' + name_postfix

        if norm is None:
            norm = si.vis.RichardsonNormalization(np.max(np.abs(self.psi) / DEFAULT_RICHARDSON_MAGNITUDE_DIVISOR))

        self.plot_mesh(self.psi, name = name, title = title,
                       colormap = colormap,
                       norm = norm,
                       show_colorbar = False,
                       **kwargs)


class LineSpecification(ElectricFieldSpecification):
    def __init__(self, name,
                 initial_state = states.QHOState(1 * N / m),
                 x_bound = 10 * nm,
                 x_points = 2 ** 9,
                 fft_cutoff_energy = 1000 * eV,
                 analytic_eigenstate_type = None,
                 use_numeric_eigenstates = False,
                 number_of_numeric_eigenstates = 100,
                 **kwargs):
        super().__init__(name,
                         mesh_type = LineMesh,
                         initial_state = initial_state,
                         **kwargs)

        self.x_bound = x_bound
        self.x_points = int(x_points)

        self.fft_cutoff_energy = fft_cutoff_energy
        self.fft_cutoff_wavenumber = np.sqrt(2 * self.test_mass * self.fft_cutoff_energy) / hbar

        self.analytic_eigenstate_type = analytic_eigenstate_type
        self.use_numeric_eigenstates = use_numeric_eigenstates
        self.number_of_numeric_eigenstates = number_of_numeric_eigenstates

    def info(self):
        info = super().info()

        info_mesh = si.Info(header = f'Mesh: {self.mesh_type.__name__}')
        info_mesh.add_field('X Boundary', f'{uround(self.x_bound, bohr_radius, 3)} a_0 | {uround(self.x_bound, nm, 3)} nm')
        info_mesh.add_field('X Points', self.x_points)
        info_mesh.add_field('X Mesh Spacing', f'~{uround(self.x_bound / self.x_points, bohr_radius, 3)} a_0 | {uround(self.x_bound / self.x_points, nm, 3)} nm')

        info.add_info(info_mesh)

        info_eigenstates = si.Info(header = f'Numeric Eigenstates: {self.use_numeric_eigenstates}')
        if self.use_numeric_eigenstates:
            info_eigenstates.add_field('Number of Numeric Eigenstates', self.number_of_numeric_eigenstates)

        info.add_info(info_eigenstates)

        return info


class LineMesh(QuantumMesh):
    mesh_storage_method = ['x']

    def __init__(self, simulation):
        super().__init__(simulation)

        self.x_mesh = np.linspace(-self.spec.x_bound, self.spec.x_bound, self.spec.x_points)
        self.delta_x = np.abs(self.x_mesh[1] - self.x_mesh[0])
        self.x_center_index = si.utils.find_nearest_entry(self.x_mesh, 0).index

        self.wavenumbers = twopi * nfft.fftfreq(len(self.x_mesh), d = self.delta_x)
        self.delta_k = np.abs(self.wavenumbers[1] - self.wavenumbers[0])

        self.inner_product_multiplier = self.delta_x
        self.g_factor = 1

        if self.spec.use_numeric_eigenstates:
            logger.debug('Calculating numeric eigenstates')

            self.analytic_to_numeric = self._get_numeric_eigenstate_basis(self.spec.number_of_numeric_eigenstates)
            self.spec.test_states = sorted(list(self.analytic_to_numeric.values()), key = lambda x: x.energy)
            self.spec.initial_state = self.analytic_to_numeric[self.spec.initial_state]

            logger.warning('Replaced test states for {} with numeric eigenbasis'.format(self))

        self.g = self.get_g_for_state(self.spec.initial_state)

        self.free_evolution_prefactor = -1j * (hbar / (2 * self.spec.test_mass)) * (self.wavenumbers ** 2)  # hbar^2/2m / hbar
        self.wavenumber_mask = np.where(np.abs(self.wavenumbers) < self.spec.fft_cutoff_wavenumber, 1, 0)

    @property
    def r_mesh(self):
        return self.x_mesh

    @property
    def energies(self):
        return ((self.wavenumbers * hbar) ** 2) / (2 * self.spec.test_mass)

    def flatten_mesh(self, mesh, flatten_along):
        return mesh

    def wrap_vector(self, vector, wrap_along):
        return vector

    @si.utils.memoize
    def get_g_for_state(self, state):
        if state.analytic and self.spec.use_numeric_eigenstates:
            try:
                state = self.analytic_to_numeric[state]
            except (AttributeError, KeyError):
                logger.debug(f'Analytic to numeric eigenstate lookup failed for state {state}')

        g = state(self.x_mesh)
        g /= np.sqrt(self.norm(g))
        g *= state.amplitude

        return g

    def energy_expectation_value(self, include_interaction = False):
        potential = self.inner_product(b = self.spec.internal_potential(r = self.x_mesh, distance = self.x_mesh, test_charge = self.spec.test_charge) * self.g)

        power_spectrum = np.abs(self.fft(self.g)) ** 2
        kinetic = np.sum((((hbar * self.wavenumbers) ** 2) / (2 * self.spec.test_mass)) * power_spectrum) / np.sum(power_spectrum)

        energy = potential + kinetic

        if include_interaction:
            energy += self.inner_product(
                b = self.spec.electric_potential(t = self.sim.time, r = self.x_mesh, distance = self.x_mesh, distance_along_polarization = self.x_mesh, test_charge = self.spec.test_charge) * self.g)

        return np.real(energy) / self.norm()

    def dipole_moment_inner_product(self, a = None, b = None):
        return self.spec.test_charge * self.inner_product(a = a, b = self.x_mesh * self.state_to_mesh(b))

    def fft(self, mesh = None):
        if mesh is None:
            mesh = self.g

        return nfft.fft(mesh, norm = 'ortho')

    def ifft(self, mesh):
        return nfft.ifft(mesh, norm = 'ortho')

    def _evolve_potential(self, time_step):
        pot = self.spec.internal_potential(t = self.sim.time, r = self.x_mesh, distance = self.x_mesh, test_charge = self.spec.test_charge)
        pot += self.spec.electric_potential(t = self.sim.time, r = self.x_mesh, distance = self.x_mesh, distance_along_polarization = self.x_mesh, test_charge = self.spec.test_charge)
        self.g *= np.exp(-1j * time_step * pot / hbar)

    def _evolve_free(self, time_step):
        self.g = self.ifft(self.fft(self.g) * np.exp(self.free_evolution_prefactor * time_step) * self.wavenumber_mask)

    def _get_kinetic_energy_matrix_operators_HAM(self):
        prefactor = -(hbar ** 2) / (2 * self.spec.test_mass * (self.delta_x ** 2))

        diag = -2 * prefactor * np.ones(len(self.x_mesh), dtype = np.complex128)
        off_diag = prefactor * np.ones(len(self.x_mesh) - 1, dtype = np.complex128)

        return sparse.diags((off_diag, diag, off_diag), (-1, 0, 1))

    @si.utils.memoize
    def get_internal_hamiltonian_matrix_operators(self):
        kinetic_x = self.get_kinetic_energy_matrix_operators().copy()

        kinetic_x = add_to_diagonal_sparse_matrix_diagonal(kinetic_x, self.spec.internal_potential(t = self.sim.time, r = self.x_mesh, distance = self.x_mesh, test_charge = self.spec.test_charge))

        return kinetic_x

    def _get_interaction_hamiltonian_matrix_operators_LEN(self):
        epot = self.spec.electric_potential(t = self.sim.time, distance_along_polarization = self.x_mesh, test_charge = self.spec.test_charge)

        return sparse.diags([epot], offsets = [0])

    @si.utils.memoize
    def _get_interaction_hamiltonian_matrix_operators_without_field_VEL(self):
        prefactor = 1j * hbar * (self.spec.test_charge / self.spec.test_mass) / (2 * self.delta_x)
        offdiag = prefactor * np.ones(self.spec.x_points - 1, dtype = np.complex128)

        return sparse.diags([-offdiag, offdiag], offsets = [-1, 1])

    def _get_interaction_hamiltonian_matrix_operators_VEL(self):
        return self._get_interaction_hamiltonian_matrix_operators_without_field_VEL() * self.spec.electric_potential.get_vector_potential_amplitude_numeric(self.sim.times_to_current)

    def _evolve_CN(self, time_step):
        """Crank-Nicholson evolution in the Length gauge."""
        tau = time_step / (2 * hbar)

        interaction_operator = self.get_interaction_hamiltonian_matrix_operators()

        hamiltonian_x = self.get_internal_hamiltonian_matrix_operators()
        hamiltonian = 1j * tau * sparse.dia_matrix(hamiltonian_x + interaction_operator)

        ham_explicit = add_to_diagonal_sparse_matrix_diagonal(-hamiltonian, 1)
        ham_implicit = add_to_diagonal_sparse_matrix_diagonal(hamiltonian, 1)

        operators = [
            DotOperator(ham_explicit, wrapping_direction = None),
            TDMAOperator(ham_implicit, wrapping_direction = None),
        ]

        self.g = apply_operators(self, self.g, *operators)

    def _make_split_operator_evolution_operators(self, interaction_hamiltonians_matrix_operators, tau):
        return getattr(self, f'_make_split_operator_evolution_operators_{self.spec.evolution_gauge}')(interaction_hamiltonians_matrix_operators, tau)

    def _make_split_operator_evolution_operators_LEN(self, interaction_hamiltonians_matrix_operators, tau):
        return [DotOperator(sparse.diags([np.exp(-1j * interaction_hamiltonians_matrix_operators.data[0] * tau)], offsets = [0]), wrapping_direction = None)]

    def _make_split_operator_evolution_operators_VEL(self, interaction_hamiltonians_matrix_operators, tau):
        a = interaction_hamiltonians_matrix_operators.data[-1][1:] * tau * (-1j)

        a_even, a_odd = a[::2], a[1::2]

        even_diag = np.zeros(len(a) + 1, dtype = np.complex128)
        even_offdiag = np.zeros(len(a), dtype = np.complex128)
        odd_diag = np.zeros(len(a) + 1, dtype = np.complex128)
        odd_offdiag = np.zeros(len(a), dtype = np.complex128)

        if len(self.x_mesh) % 2 != 0:
            even_diag[:-1] = np.cos(a_even).repeat(2)
            even_diag[-1] = 1

            even_offdiag[::2] = np.sin(a_even)

            odd_diag[0] = 1
            odd_diag[1:] = np.cos(a_odd).repeat(2)

            odd_offdiag[1::2] = np.sin(a_odd)
        else:
            even_diag[:] = np.cos(a_even).repeat(2)

            even_offdiag[::2] = np.sin(a_even)

            odd_diag[0] = odd_diag[-1] = 1
            odd_diag[1:-1] = np.cos(a_odd).repeat(2)

            odd_offdiag[1::2] = np.sin(a_odd)

        even = sparse.diags([-even_offdiag, even_diag, even_offdiag], offsets = [-1, 0, 1])
        odd = sparse.diags([-odd_offdiag, odd_diag, odd_offdiag], offsets = [-1, 0, 1])

        return (
            DotOperator(even, wrapping_direction = None),
            DotOperator(odd, wrapping_direction = None),
        )

    def _evolve_SO(self, time_step):
        """Split-Operator evolution in the Length gauge."""
        tau = time_step / (2 * hbar)

        hamiltonian_x = self.get_internal_hamiltonian_matrix_operators()

        ham_x_explicit = add_to_diagonal_sparse_matrix_diagonal(-1j * tau * hamiltonian_x, 1)
        ham_x_implicit = add_to_diagonal_sparse_matrix_diagonal(1j * tau * hamiltonian_x, 1)

        split_operators = self._make_split_operator_evolution_operators(self.get_interaction_hamiltonian_matrix_operators(), tau)

        operators = [
            *split_operators,
            DotOperator(ham_x_explicit, wrapping_direction = None),
            TDMAOperator(ham_x_implicit, wrapping_direction = None),
            *reversed(split_operators),
        ]

        self.g = apply_operators(self, self.g, *operators)

    def _evolve_S(self, time_step):
        """Spectral evolution in the Length gauge."""
        self._evolve_potential(time_step / 2)
        self._evolve_free(time_step)  # splitting order chosen for computational efficiency (only one FFT per time step)
        self._evolve_potential(time_step / 2)

    def _get_numeric_eigenstate_basis(self, number_of_eigenstates):
        analytic_to_numeric = {}

        h = self.get_internal_hamiltonian_matrix_operators()

        eigenvalues, eigenvectors = sparsealg.eigsh(h, k = number_of_eigenstates, which = 'SA')

        for nn, (eigenvalue, eigenvector) in enumerate(zip(eigenvalues, eigenvectors.T)):
            eigenvector /= np.sqrt(self.inner_product_multiplier * np.sum(np.abs(eigenvector) ** 2))  # normalize

            try:
                bound = True
                analytic_state = self.spec.analytic_eigenstate_type.from_potential(self.spec.internal_potential, self.spec.test_mass,
                                                                                   n = nn + self.spec.analytic_eigenstate_type.smallest_n)
            except states.IllegalQuantumState:
                bound = False
                analytic_state = states.OneDFreeParticle.from_energy(eigenvalue, mass = self.spec.test_mass)

            numeric_state = states.NumericOneDState(eigenvector, eigenvalue, bound = bound, analytic_state = analytic_state)
            if analytic_state is None:
                analytic_state = numeric_state

            analytic_to_numeric[analytic_state] = numeric_state

        logger.debug(f'Generated numerical eigenbasis with {len(analytic_to_numeric)} states')

        return analytic_to_numeric

    def gauge_transformation(self, *, g = None, leaving_gauge = None):
        if g is None:
            g = self.g
        if leaving_gauge is None:
            leaving_gauge = self.spec.evolution_gauge

        vamp = self.spec.electric_potential.get_vector_potential_amplitude_numeric_cumulative(self.sim.times_to_current)
        integral = integ.simps(y = vamp ** 2,
                               x = self.sim.times_to_current)

        dipole_to_velocity = np.exp(1j * integral * (self.spec.test_charge ** 2) / (2 * self.spec.test_mass * hbar))
        dipole_to_length = np.exp(-1j * self.spec.test_charge * vamp[-1] * self.x_mesh / hbar)

        if leaving_gauge == 'LEN':
            return np.conj(dipole_to_length) * dipole_to_velocity * g
        elif leaving_gauge == 'VEL':
            return dipole_to_length * np.conj(dipole_to_velocity) * g

    def get_mesh_slicer(self, plot_limit):
        if plot_limit is None:
            mesh_slicer = slice(None, None, 1)
        else:
            x_lim_points = round(plot_limit / self.delta_x)
            mesh_slicer = slice(int(self.x_center_index - x_lim_points), int(self.x_center_index + x_lim_points + 1), 1)

        return mesh_slicer

    def attach_mesh_to_axis(self, axis, mesh,
                            distance_unit = 'bohr_radius',
                            colormap = plt.get_cmap('inferno'),
                            norm = si.vis.AbsoluteRenormalize(),
                            shading = 'flat',
                            plot_limit = None,
                            slicer = 'get_mesh_slicer',
                            **kwargs):
        unit_value, _ = get_unit_value_and_latex_from_unit(distance_unit)

        _slice = getattr(self, slicer)(plot_limit)

        line, = axis.plot(self.x_mesh[_slice] / unit_value, norm(mesh[_slice]), **kwargs)

        return line

    def plot_mesh(self, mesh, distance_unit = 'nm', **kwargs):
        si.vis.xy_plot(self.sim.name + '_' + kwargs.pop('name'),
                       self.x_mesh,
                       mesh,
                       x_label = 'Distance $x$',
                       x_unit_value = distance_unit,
                       **kwargs)

    def update_mesh(self, colormesh, updated_mesh, norm = None, **kwargs):
        if norm is not None:
            updated_mesh = norm(updated_mesh)

        super().update_mesh(colormesh, updated_mesh, **kwargs)

    def plot_fft(self):
        raise NotImplementedError

    def attach_fft_to_axis(self, axis,
                           distance_unit = 'per_nm',
                           colormap = plt.get_cmap('inferno'),
                           norm = si.vis.AbsoluteRenormalize(),
                           shading = 'gouraud',
                           plot_limit = None,
                           slicer = 'get_mesh_slicer',
                           **kwargs):
        unit_value, _ = get_unit_value_and_latex_from_unit(distance_unit)

        _slice = getattr(self, slicer)(plot_limit)

        line, = axis.plot(self.wavenumbers[_slice] / unit_value, norm(self.fft()[_slice]), **kwargs)

        return line

    def update_fft_mesh(self, colormesh, **kwargs):
        self.update_mesh(colormesh, self.fft(), **kwargs)

# class LineSnapshot(Snapshot):
#     def __init__(self):
#         pass


class CylindricalSliceSpecification(ElectricFieldSpecification):
    def __init__(self, name,
                 z_bound = 20 * bohr_radius, rho_bound = 20 * bohr_radius,
                 z_points = 2 ** 9, rho_points = 2 ** 8,
                 evolution_equations = 'HAM',
                 evolution_method = 'CN',
                 evolution_gauge = 'LEN',
                 **kwargs):
        super().__init__(name,
                         mesh_type = CylindricalSliceMesh,
                         evolution_equations = evolution_equations,
                         evolution_method = evolution_method,
                         evolution_gauge = evolution_gauge,
                         **kwargs)

        self.z_bound = z_bound
        self.rho_bound = rho_bound
        self.z_points = int(z_points)
        self.rho_points = int(rho_points)

    def info(self):
        info = super().info()

        info_mesh = si.Info(header = f'Mesh: {self.mesh_type.__name__}')
        info_mesh.add_field('Z Boundary', f'{uround(self.z_bound, bohr_radius, 3)} a_0')
        info_mesh.add_field('Z Points', self.z_points)
        info_mesh.add_field('Z Mesh Spacing', f'~{uround(self.z_bound / self.z_points, bohr_radius, 3)} a_0')
        info_mesh.add_field('Rho Boundary', f'{uround(self.rho_bound, bohr_radius, 3)} a_0')
        info_mesh.add_field('Rho Points', self.rho_points)
        info_mesh.add_field('Rho Mesh Spacing', f'~{uround(self.rho_bound / self.rho_points, bohr_radius, 3)} a_0')
        info_mesh.add_field('Total Mesh Points', int(self.z_points * self.rho_points))

        info.add_info(info_mesh)

        return info


class CylindricalSliceMesh(QuantumMesh):
    mesh_storage_method = ('z', 'rho')

    def __init__(self, simulation):
        super().__init__(simulation)

        self.z = np.linspace(-self.spec.z_bound, self.spec.z_bound, self.spec.z_points)
        self.rho = np.delete(np.linspace(0, self.spec.rho_bound, self.spec.rho_points + 1), 0)

        self.delta_z = self.z[1] - self.z[0]
        self.delta_rho = self.rho[1] - self.rho[0]
        self.inner_product_multiplier = self.delta_z * self.delta_rho

        self.rho -= self.delta_rho / 2

        self.z_center_index = int(self.spec.z_points // 2)
        self.z_max = np.max(self.z)
        self.rho_max = np.max(self.rho)

        # self.z_mesh, self.rho_mesh = np.meshgrid(self.z, self.rho, indexing = 'ij')
        self.g = self.get_g_for_state(self.spec.initial_state)

        self.mesh_points = len(self.z) * len(self.rho)
        self.matrix_operator_shape = (self.mesh_points, self.mesh_points)
        self.mesh_shape = np.shape(self.r_mesh)

    @property
    @si.utils.memoize
    def z_mesh(self):
        return np.meshgrid(self.z, self.rho, indexing = 'ij')[0]

    @property
    @si.utils.memoize
    def rho_mesh(self):
        return np.meshgrid(self.z, self.rho, indexing = 'ij')[1]

    @property
    def g_factor(self):
        return np.sqrt(twopi * self.rho_mesh)

    @property
    def r_mesh(self):
        return np.sqrt((self.z_mesh ** 2) + (self.rho_mesh ** 2))

    @property
    def theta_mesh(self):
        return np.arccos(self.z_mesh / self.r_mesh)  # either of these work
        # return np.arctan2(self.rho_mesh, self.z_mesh)  # I have a slight preference for arccos because it will never divide by zero

    @property
    def sin_theta_mesh(self):
        return np.sin(self.theta_mesh)

    @property
    def cos_theta_mesh(self):
        return np.cos(self.theta_mesh)

    def flatten_mesh(self, mesh, flatten_along):
        """Return a mesh flattened along one of the mesh coordinates ('z' or 'rho')."""
        if flatten_along == 'z':
            flat = 'F'
        elif flatten_along == 'rho':
            flat = 'C'
        elif flatten_along is None:
            return mesh
        else:
            raise ValueError("{} is not a valid specifier for flatten_along (valid specifiers: 'z', 'rho')".format(flatten_along))

        return mesh.flatten(flat)

    def wrap_vector(self, vector, wrap_along):
        if wrap_along == 'z':
            wrap = 'F'
        elif wrap_along == 'rho':
            wrap = 'C'
        elif wrap_along is None:
            return vector
        else:
            raise ValueError("{} is not a valid specifier for wrap_vector (valid specifiers: 'z', 'rho')".format(wrap_along))

        return np.reshape(vector, self.mesh_shape, wrap)

    @si.utils.memoize
    def get_g_for_state(self, state):
        g = self.g_factor * state(self.r_mesh, self.theta_mesh, 0)
        g /= np.sqrt(self.norm(g))
        g *= state.amplitude

        return g

    def dipole_moment_inner_product(self, a = None, b = None):
        return self.spec.test_charge * self.inner_product(a = a, b = self.z_mesh * self.state_to_mesh(b))

    def _get_kinetic_energy_matrix_operators_HAM(self):
        """Get the mesh kinetic energy operator matrices for z and rho."""
        z_prefactor = -(hbar ** 2) / (2 * self.spec.test_mass * (self.delta_z ** 2))
        rho_prefactor = -(hbar ** 2) / (2 * self.spec.test_mass * (self.delta_rho ** 2))

        z_diagonal = z_prefactor * (-2) * np.ones(self.mesh_points, dtype = np.complex128)
        z_offdiagonal = z_prefactor * np.array([1 if (z_index + 1) % self.spec.z_points != 0 else 0 for z_index in range(self.mesh_points - 1)], dtype = np.complex128)

        @si.utils.memoize
        def c(j):
            return j / np.sqrt((j ** 2) - 0.25)

        rho_diagonal = rho_prefactor * (-2) * np.ones(self.mesh_points, dtype = np.complex128)
        rho_offdiagonal = np.zeros(self.mesh_points - 1, dtype = np.complex128)
        for rho_index in range(self.mesh_points - 1):
            if (rho_index + 1) % self.spec.rho_points != 0:
                j = (rho_index % self.spec.rho_points) + 1  # get j for the upper diagonal
                rho_offdiagonal[rho_index] = c(j)
        rho_offdiagonal *= rho_prefactor

        z_kinetic = sparse.diags([z_offdiagonal, z_diagonal, z_offdiagonal], offsets = (-1, 0, 1))
        rho_kinetic = sparse.diags([rho_offdiagonal, rho_diagonal, rho_offdiagonal], offsets = (-1, 0, 1))

        return z_kinetic, rho_kinetic

    @si.utils.memoize
    def get_internal_hamiltonian_matrix_operators(self):
        """Get the mesh internal Hamiltonian matrix operators for z and rho."""
        kinetic_z, kinetic_rho = self.get_kinetic_energy_matrix_operators()
        potential_mesh = self.spec.internal_potential(r = self.r_mesh, test_charge = self.spec.test_charge)

        kinetic_z = add_to_diagonal_sparse_matrix_diagonal(kinetic_z, value = 0.5 * self.flatten_mesh(potential_mesh, 'z'))
        kinetic_rho = add_to_diagonal_sparse_matrix_diagonal(kinetic_rho, value = 0.5 * self.flatten_mesh(potential_mesh, 'rho'))

        return kinetic_z, kinetic_rho

    def _get_interaction_hamiltonian_matrix_operators_LEN(self):
        """Get the interaction term calculated from the Lagrangian evolution equations."""
        electric_potential_energy_mesh = self.spec.electric_potential(t = self.sim.time, distance_along_polarization = self.z_mesh, test_charge = self.spec.test_charge)

        interaction_hamiltonian_z = sparse.diags(self.flatten_mesh(electric_potential_energy_mesh, 'z'))
        interaction_hamiltonian_rho = sparse.diags(self.flatten_mesh(electric_potential_energy_mesh, 'rho'))

        return interaction_hamiltonian_z, interaction_hamiltonian_rho

    def _get_interaction_hamiltonian_matrix_operators_VEL(self):
        # vector_potential_amplitude = -self.spec.electric_potential.get_electric_field_integral_numeric_cumulative(self.sim.times_to_current)
        raise NotImplementedError

    def tg_mesh(self, use_abs_g = False):
        hamiltonian_z, hamiltonian_rho = self.get_kinetic_energy_matrix_operators()

        if use_abs_g:
            g = np.abs(self.g)
        else:
            g = self.g

        g_vector_z = self.flatten_mesh(g, 'z')
        hg_vector_z = hamiltonian_z.dot(g_vector_z)
        hg_mesh_z = self.wrap_vector(hg_vector_z, 'z')

        g_vector_rho = self.flatten_mesh(g, 'rho')
        hg_vector_rho = hamiltonian_rho.dot(g_vector_rho)
        hg_mesh_rho = self.wrap_vector(hg_vector_rho, 'rho')

        return hg_mesh_z + hg_mesh_rho

    def hg_mesh(self, include_interaction = False):
        hamiltonian_z, hamiltonian_rho = self.get_internal_hamiltonian_matrix_operators()

        g_vector_z = self.flatten_mesh(self.g, 'z')
        hg_vector_z = hamiltonian_z.dot(g_vector_z)
        hg_mesh_z = self.wrap_vector(hg_vector_z, 'z')

        g_vector_rho = self.flatten_mesh(self.g, 'rho')
        hg_vector_rho = hamiltonian_rho.dot(g_vector_rho)
        hg_mesh_rho = self.wrap_vector(hg_vector_rho, 'rho')

        if include_interaction:
            raise NotImplementedError

        return hg_mesh_z + hg_mesh_rho

    def energy_expectation_value(self, include_interaction = False):
        return np.real(self.inner_product(b = self.hg_mesh())) / self.norm()

    @si.utils.memoize
    def _get_probability_current_matrix_operators(self):
        """Get the mesh probability current operators for z and rho."""
        z_prefactor = hbar / (4 * pi * self.spec.test_mass * self.delta_rho * self.delta_z)
        rho_prefactor = hbar / (4 * pi * self.spec.test_mass * (self.delta_rho ** 2))

        # construct the diagonals of the z probability current matrix operator
        z_offdiagonal = np.zeros(self.mesh_points - 1, dtype = np.complex128)
        for z_index in range(0, self.mesh_points - 1):
            if (z_index + 1) % self.spec.z_points == 0:  # detect edge of mesh
                z_offdiagonal[z_index] = 0
            else:
                j = z_index // self.spec.z_points
                z_offdiagonal[z_index] = 1 / (j + 0.5)
        z_offdiagonal *= z_prefactor

        @si.utils.memoize
        def d(j):
            return 1 / np.sqrt((j ** 2) - 0.25)

        # construct the diagonals of the rho probability current matrix operator
        rho_offdiagonal = np.zeros(self.mesh_points - 1, dtype = np.complex128)
        for rho_index in range(0, self.mesh_points - 1):
            if (rho_index + 1) % self.spec.rho_points == 0:  # detect edge of mesh
                rho_offdiagonal[rho_index] = 0
            else:
                j = (rho_index % self.spec.rho_points) + 1
                rho_offdiagonal[rho_index] = d(j)
        rho_offdiagonal *= rho_prefactor

        z_current = sparse.diags([-z_offdiagonal, z_offdiagonal], offsets = [-1, 1])
        rho_current = sparse.diags([-rho_offdiagonal, rho_offdiagonal], offsets = [-1, 1])

        return z_current, rho_current

    def get_probability_current_vector_field(self):
        z_current, rho_current = self._get_probability_current_matrix_operators()

        g_vector_z = self.flatten_mesh(self.g, 'z')
        current_vector_z = z_current.dot(g_vector_z)
        gradient_mesh_z = self.wrap_vector(current_vector_z, 'z')
        current_mesh_z = np.imag(np.conj(self.g) * gradient_mesh_z)

        g_vector_rho = self.flatten_mesh(self.g, 'rho')
        current_vector_rho = rho_current.dot(g_vector_rho)
        gradient_mesh_rho = self.wrap_vector(current_vector_rho, 'rho')
        current_mesh_rho = np.imag(np.conj(self.g) * gradient_mesh_rho)

        return current_mesh_z, current_mesh_rho

    def get_spline_for_mesh(self, mesh):
        return sp.interp.RectBivariateSpline(self.z, self.rho, mesh)

    def _evolve_CN(self, time_step):
        """
        Evolve the mesh forward in time by time_step.

        Crank-Nicholson evolution in the Length gauge.

        :param time_step:
        :return:
        """
        tau = time_step / (2 * hbar)

        hamiltonian_z, hamiltonian_rho = self.get_internal_hamiltonian_matrix_operators()
        interaction_hamiltonian_z, interaction_hamiltonian_rho = self.get_interaction_hamiltonian_matrix_operators()

        hamiltonian_z = 1j * tau * add_to_diagonal_sparse_matrix_diagonal(hamiltonian_z, value = 0.5 * interaction_hamiltonian_z.diagonal())
        hamiltonian_rho = 1j * tau * add_to_diagonal_sparse_matrix_diagonal(hamiltonian_rho, value = 0.5 * interaction_hamiltonian_rho.diagonal())

        hamiltonian_rho_explicit = add_to_diagonal_sparse_matrix_diagonal(-hamiltonian_rho, value = 1)
        hamiltonian_z_implicit = add_to_diagonal_sparse_matrix_diagonal(hamiltonian_z, value = 1)
        hamiltonian_z_explicit = add_to_diagonal_sparse_matrix_diagonal(-hamiltonian_z, value = 1)
        hamiltonian_rho_implicit = add_to_diagonal_sparse_matrix_diagonal(hamiltonian_rho, value = 1)

        operators = [
            DotOperator(hamiltonian_rho_explicit, wrapping_direction = 'rho'),
            TDMAOperator(hamiltonian_z_implicit, wrapping_direction = 'z'),
            DotOperator(hamiltonian_z_explicit, wrapping_direction = 'z'),
            TDMAOperator(hamiltonian_rho_implicit, wrapping_direction = 'rho'),
        ]

        self.g = apply_operators(self, self.g, *operators)

    @si.utils.memoize
    def get_mesh_slicer(self, plot_limit = None):
        """Returns a slice object that slices a mesh to the given distance of the center."""
        if plot_limit is None:
            mesh_slicer = (slice(None, None, 1), slice(None, None, 1))
        else:
            z_lim_points = round(plot_limit / self.delta_z)
            rho_lim_points = round(plot_limit / self.delta_rho)
            mesh_slicer = (slice(int(self.z_center_index - z_lim_points), int(self.z_center_index + z_lim_points + 1), 1), slice(0, int(rho_lim_points + 1), 1))

        return mesh_slicer

    def attach_mesh_to_axis(self, axis, mesh,
                            distance_unit = 'bohr_radius',
                            colormap = plt.get_cmap('inferno'),
                            norm = si.vis.AbsoluteRenormalize(),
                            shading = 'flat',
                            plot_limit = None,
                            slicer = 'get_mesh_slicer',
                            **kwargs):
        unit_value, _ = get_unit_value_and_latex_from_unit(distance_unit)

        _slice = getattr(self, slicer)(plot_limit)

        color_mesh = axis.pcolormesh(self.z_mesh[_slice] / unit_value,
                                     self.rho_mesh[_slice] / unit_value,
                                     mesh[_slice],
                                     shading = shading,
                                     cmap = colormap,
                                     norm = norm,
                                     **kwargs)

        return color_mesh

    def attach_probability_current_to_axis(self, axis, plot_limit = None, distance_unit = 'bohr_radius'):
        unit_value, _ = get_unit_value_and_latex_from_unit(distance_unit)

        current_mesh_z, current_mesh_rho = self.get_probability_current_vector_field()

        current_mesh_z *= self.delta_z
        current_mesh_rho *= self.delta_rho

        skip_count = int(self.z_mesh.shape[0] / 50), int(self.z_mesh.shape[1] / 50)
        skip = (slice(None, None, skip_count[0]), slice(None, None, skip_count[1]))
        normalization = np.max(np.sqrt(current_mesh_z ** 2 + current_mesh_rho ** 2)[skip])
        if normalization == 0 or normalization is np.NaN:
            normalization = 1

        quiv = axis.quiver(self.z_mesh[self.get_mesh_slicer(plot_limit)][skip] / unit_value,
                           self.rho_mesh[self.get_mesh_slicer(plot_limit)][skip] / unit_value,
                           current_mesh_z[self.get_mesh_slicer(plot_limit)][skip] / normalization,
                           current_mesh_rho[self.get_mesh_slicer(plot_limit)][skip] / normalization,
                           pivot = 'middle', units = 'width', scale = 10, scale_units = 'width', width = 0.0015, alpha = 0.5)

        return quiv

    def plot_mesh(self, mesh,
                  name = '',
                  title = None,
                  distance_unit = 'bohr_radius',
                  colormap = COLORMAP_WAVEFUNCTION,
                  norm = si.vis.AbsoluteRenormalize(),
                  shading = 'flat',
                  plot_limit = None,
                  slicer = 'get_mesh_slicer',
                  show_colorbar = True,
                  show_title = True,
                  show_axes = True,
                  title_size = 12,
                  axis_label_size = 12,
                  tick_label_size = 10,
                  grid_kwargs = None,
                  title_y_adjust = 1.1,
                  # overlay_probability_current = False, probability_current_time_step = 0,
                  **kwargs):
        if grid_kwargs is None:
            grid_kwargs = {}

        with si.vis.FigureManager(name = f'{self.spec.name}__{name}', **kwargs) as figman:
            fig = figman.fig
            fig.set_tight_layout(True)
            axis = plt.subplot(111)

            unit_value, unit_name = get_unit_value_and_latex_from_unit(distance_unit)

            color_mesh = self.attach_mesh_to_axis(
                axis, mesh,
                distance_unit = distance_unit,
                colormap = colormap,
                norm = norm,
                shading = shading,
                plot_limit = plot_limit,
                slicer = slicer
            )
            # if overlay_probability_current:
            #     quiv = self.attach_probability_current_to_axis(axis, plot_limit = plot_limit, distance_unit = distance_unit)

            axis.set_xlabel(r'$z$ (${}$)'.format(unit_name), fontsize = axis_label_size)
            axis.set_ylabel(r'$\rho$ (${}$)'.format(unit_name), fontsize = axis_label_size)
            if title is not None and title != '' and show_axes and show_title:
                title = axis.set_title(title, fontsize = title_size)
                title.set_y(title_y_adjust)  # move title up a bit

            # make a colorbar
            if show_colorbar and show_axes:
                cbar = fig.colorbar(mappable = color_mesh, ax = axis, pad = .1)
                cbar.ax.tick_params(labelsize = tick_label_size)

            axis.axis('tight')  # removes blank space between color mesh and axes

            axis.grid(True, color = si.vis.CMAP_TO_OPPOSITE[colormap], **{**si.vis.COLORMESH_GRID_KWARGS, **grid_kwargs})  # change grid color to make it show up against the colormesh

            axis.tick_params(labelright = True, labeltop = True)  # ticks on all sides
            axis.tick_params(axis = 'both', which = 'major', labelsize = tick_label_size)  # increase size of tick labels
            # axis.tick_params(axis = 'both', which = 'both', length = 0)

            # set upper and lower y ticks to not display to avoid collisions with the x ticks at the edges
            # y_ticks = axis.yaxis.get_major_ticks()
            # y_ticks[0].label1.set_visible(False)
            # y_ticks[0].label2.set_visible(False)
            # y_ticks[-1].label1.set_visible(False)
            # y_ticks[-1].label2.set_visible(False)

            if not show_axes:
                axis.axis('off')


class SphericalSliceSpecification(ElectricFieldSpecification):
    def __init__(self, name,
                 r_bound = 20 * bohr_radius,
                 r_points = 2 ** 10, theta_points = 2 ** 10,
                 **kwargs):
        super().__init__(name, mesh_type = SphericalSliceMesh, **kwargs)

        self.r_bound = r_bound

        self.r_points = int(r_points)
        self.theta_points = int(theta_points)

    def info(self):
        info = super().info()

        info_mesh = si.Info(header = f'Mesh: {self.mesh_type.__name__}')
        info_mesh.add_field('R Boundary', f'{uround(self.r_bound, bohr_radius, 3)} a_0')
        info_mesh.add_field('R Points', self.r_points)
        info_mesh.add_field('R Mesh Spacing', f'~{uround(self.r_bound / self.z_points, bohr_radius, 3)} a_0')
        info_mesh.add_field('Theta Points', self.theta_points)
        info_mesh.add_field('Theta Mesh Spacing', f'~{uround(pi / self.theta_points, bohr_radius, 3)} a_0')
        info_mesh.add_field('Maximum Adjacent-Point Spacing', f'~{uround(pi * self.r_bound / self.theta_points, bohr_radius, 3)} a_0')
        info_mesh.add_field('Total Mesh Points', int(self.r_points * self.theta_points))

        info.add_info(info_mesh)

        return info


class SphericalSliceMesh(QuantumMesh):
    mesh_storage_method = ('r', 'theta')

    def __init__(self, simulation):
        super().__init__(simulation)

        self.r = np.linspace(0, self.spec.r_bound, self.spec.r_points)
        self.theta = np.delete(np.linspace(0, pi, self.spec.theta_points + 1), 0)

        self.delta_r = self.r[1] - self.r[0]
        self.delta_theta = self.theta[1] - self.theta[0]
        self.inner_product_multiplier = self.delta_r * self.delta_theta

        self.r += self.delta_r / 2
        self.theta -= self.delta_theta / 2

        self.r_max = np.max(self.r)

        # self.r_mesh, self.theta_mesh = np.meshgrid(self.r, self.theta, indexing = 'ij')
        self.g = self.get_g_for_state(self.spec.initial_state)

        self.mesh_points = len(self.r) * len(self.theta)
        self.matrix_operator_shape = (self.mesh_points, self.mesh_points)
        self.mesh_shape = np.shape(self.r_mesh)

    @property
    @si.utils.memoize
    def r_mesh(self):
        return np.meshgrid(self.r, self.theta, indexing = 'ij')[0]

    @property
    @si.utils.memoize
    def theta_mesh(self):
        return np.meshgrid(self.r, self.theta, indexing = 'ij')[1]

    @property
    def g_factor(self):
        return np.sqrt(twopi * np.sin(self.theta_mesh)) * self.r_mesh

    @property
    def z_mesh(self):
        return self.r_mesh * np.cos(self.theta_mesh)

    def flatten_mesh(self, mesh, flatten_along):
        """Return a mesh flattened along one of the mesh coordinates ('theta' or 'r')."""
        if flatten_along == 'r':
            flat = 'F'
        elif flatten_along == 'theta':
            flat = 'C'
        elif flatten_along is None:
            return mesh
        else:
            raise ValueError("{} is not a valid specifier for flatten_mesh (valid specifiers: 'r', 'theta')".format(flatten_along))

        return mesh.flatten(flat)

    def wrap_vector(self, vector, wrap_along):
        if wrap_along == 'r':
            wrap = 'F'
        elif wrap_along == 'theta':
            wrap = 'C'
        elif wrap_along is None:
            return vector
        else:
            raise ValueError("{} is not a valid specifier for wrap_vector (valid specifiers: 'r', 'theta')".format(wrap_along))

        return np.reshape(vector, self.mesh_shape, wrap)

    def state_overlap(self, state_a = None, state_b = None):
        """State overlap between two states. If either state is None, the state on the g is used for that state."""
        if state_a is None:
            mesh_a = self.g
        else:
            mesh_a = self.get_g_for_state(state_a)
        if state_b is None:
            b = self.g
        else:
            b = self.get_g_for_state(state_b)

        return np.abs(self.inner_product(mesh_a, b)) ** 2

    @si.utils.memoize
    def get_g_for_state(self, state):
        g = self.g_factor * state(self.r_mesh, self.theta_mesh, 0)
        g /= np.sqrt(self.norm(g))
        g *= state.amplitude

        return g

    def dipole_moment_inner_product(self, a = None, b = None):
        return self.spec.test_charge * self.inner_product(a = a, b = self.z_mesh * self.state_to_mesh(b))

    def _get_kinetic_energy_matrix_operators_HAM(self):
        r_prefactor = -(hbar ** 2) / (2 * electron_mass_reduced * (self.delta_r ** 2))
        theta_prefactor = -(hbar ** 2) / (2 * electron_mass_reduced * ((self.delta_r * self.delta_theta) ** 2))

        r_diagonal = r_prefactor * (-2) * np.ones(self.mesh_points, dtype = np.complex128)
        r_offdiagonal = r_prefactor * np.array([1 if (z_index + 1) % self.spec.r_points != 0 else 0 for z_index in range(self.mesh_points - 1)], dtype = np.complex128)

        @si.utils.memoize
        def theta_j_prefactor(x):
            return 1 / (x + 0.5) ** 2

        @si.utils.memoize
        def sink(x):
            return np.sin(x * self.delta_theta)

        @si.utils.memoize
        def sqrt_sink_ratio(x_num, x_den):
            return np.sqrt(sink(x_num) / sink(x_den))

        @si.utils.memoize
        def cotank(x):
            return 1 / np.tan(x * self.delta_theta)

        theta_diagonal = (-2) * np.ones(self.mesh_points, dtype = np.complex128)
        for theta_index in range(self.mesh_points):
            j = theta_index // self.spec.theta_points
            theta_diagonal[theta_index] *= theta_j_prefactor(j)
        theta_diagonal *= theta_prefactor

        theta_upper_diagonal = np.zeros(self.mesh_points - 1, dtype = np.complex128)
        theta_lower_diagonal = np.zeros(self.mesh_points - 1, dtype = np.complex128)
        for theta_index in range(self.mesh_points - 1):
            if (theta_index + 1) % self.spec.theta_points != 0:
                j = theta_index // self.spec.theta_points
                k = theta_index % self.spec.theta_points
                k_p = k + 1  # add 1 because the entry for the lower diagonal is really for the next point (k -> k + 1), not this one
                theta_upper_diagonal[theta_index] = theta_j_prefactor(j) * (1 + (self.delta_theta / 2) * cotank(k + 0.5)) * sqrt_sink_ratio(k + 0.5, k + 1.5)
                theta_lower_diagonal[theta_index] = theta_j_prefactor(j) * (1 - (self.delta_theta / 2) * cotank(k_p + 0.5)) * sqrt_sink_ratio(k_p + 0.5, k_p - 0.5)
        theta_upper_diagonal *= theta_prefactor
        theta_lower_diagonal *= theta_prefactor

        r_kinetic = sparse.diags([r_offdiagonal, r_diagonal, r_offdiagonal], offsets = (-1, 0, 1))
        theta_kinetic = sparse.diags([theta_lower_diagonal, theta_diagonal, theta_upper_diagonal], offsets = (-1, 0, 1))

        return r_kinetic, theta_kinetic

    @si.utils.memoize
    def get_internal_hamiltonian_matrix_operators(self):
        kinetic_r, kinetic_theta = self.get_kinetic_energy_matrix_operators()
        potential_mesh = self.spec.internal_potential(r = self.r_mesh, test_charge = self.spec.test_charge)

        kinetic_r = add_to_diagonal_sparse_matrix_diagonal(kinetic_r, value = 0.5 * self.flatten_mesh(potential_mesh, 'r'))
        kinetic_theta = add_to_diagonal_sparse_matrix_diagonal(kinetic_theta, value = 0.5 * self.flatten_mesh(potential_mesh, 'theta'))

        return kinetic_r, kinetic_theta

    def _get_interaction_hamiltonian_matrix_operators_LEN(self):
        """Get the angular momentum interaction term calculated from the Lagrangian evolution equations."""
        electric_potential_energy_mesh = self.spec.electric_potential(t = self.sim.time, distance_along_polarization = self.z_mesh, test_charge = self.spec.test_charge)

        interaction_hamiltonian_r = sparse.diags(self.flatten_mesh(electric_potential_energy_mesh, 'r'))
        interaction_hamiltonian_theta = sparse.diags(self.flatten_mesh(electric_potential_energy_mesh, 'theta'))

        return interaction_hamiltonian_r, interaction_hamiltonian_theta

    def _get_interaction_hamiltonian_matrix_operators_VEL(self):
        # vector_potential_amplitude = -self.spec.electric_potential.get_electric_field_integral_numeric_cumulative(self.sim.times_to_current)

        raise NotImplementedError

    def tg_mesh(self, use_abs_g = False):
        hamiltonian_r, hamiltonian_theta = self.get_kinetic_energy_matrix_operators()

        if use_abs_g:
            g = np.abs(self.g)
        else:
            g = self.g

        g_vector_r = self.flatten_mesh(g, 'r')
        hg_vector_r = hamiltonian_r.dot(g_vector_r)
        hg_mesh_r = self.wrap_vector(hg_vector_r, 'r')

        g_vector_theta = self.flatten_mesh(g, 'theta')
        hg_vector_theta = hamiltonian_theta.dot(g_vector_theta)
        hg_mesh_theta = self.wrap_vector(hg_vector_theta, 'theta')

        return hg_mesh_r + hg_mesh_theta

    def hg_mesh(self):
        hamiltonian_r, hamiltonian_theta = self.get_internal_hamiltonian_matrix_operators()

        g_vector_r = self.flatten_mesh(self.g, 'r')
        hg_vector_r = hamiltonian_r.dot(g_vector_r)
        hg_mesh_r = self.wrap_vector(hg_vector_r, 'r')

        g_vector_theta = self.flatten_mesh(self.g, 'theta')
        hg_vector_theta = hamiltonian_theta.dot(g_vector_theta)
        hg_mesh_theta = self.wrap_vector(hg_vector_theta, 'theta')

        raise NotImplementedError
        # TODO: not including interaction yet

        return hg_mesh_r + hg_mesh_theta

    def energy_expectation_value(self):
        return np.real(self.inner_product(b = self.hg_mesh())) / self.norm()

    @si.utils.memoize
    def get_probability_current_matrix_operators(self):
        raise NotImplementedError

    def get_probability_current_vector_field(self):
        raise NotImplementedError

    def get_spline_for_mesh(self, mesh):
        return sp.interp.RectBivariateSpline(self.r, self.theta, mesh)

    def _evolve_CN(self, time_step):
        """Crank-Nicholson evolution in the Length gauge."""
        tau = time_step / (2 * hbar)

        hamiltonian_r, hamiltonian_theta = self.get_internal_hamiltonian_matrix_operators()
        interaction_hamiltonian_r, interaction_hamiltonian_theta = self.get_interaction_hamiltonian_matrix_operators()

        hamiltonian_r = 1j * tau * add_to_diagonal_sparse_matrix_diagonal(hamiltonian_r, value = 0.5 * interaction_hamiltonian_r.diagonal())
        hamiltonian_theta = 1j * tau * add_to_diagonal_sparse_matrix_diagonal(hamiltonian_theta, value = 0.5 * interaction_hamiltonian_theta.diagonal())

        hamiltonian_theta_explicit = add_to_diagonal_sparse_matrix_diagonal(-hamiltonian_theta, value = 1)
        hamiltonian_r_implicit = add_to_diagonal_sparse_matrix_diagonal(hamiltonian_r, value = 1)
        hamiltonian_r_explicit = add_to_diagonal_sparse_matrix_diagonal(-hamiltonian_r, value = 1)
        hamiltonian_theta_implicit = add_to_diagonal_sparse_matrix_diagonal(hamiltonian_theta, value = 1)

        operators = [
            DotOperator(hamiltonian_theta_explicit, wrapping_direction = 'theta'),
            TDMAOperator(hamiltonian_r_implicit, wrapping_direction = 'r'),
            DotOperator(hamiltonian_r_explicit, wrapping_direction = 'r'),
            TDMAOperator(hamiltonian_theta_implicit, wrapping_direction = 'theta'),
        ]

        self.g = apply_operators(self, self.g, *operators)

    @si.utils.memoize
    def get_mesh_slicer(self, distance_from_center = None):
        """Returns a slice object that slices a mesh to the given distance of the center."""
        if distance_from_center is None:
            mesh_slicer = slice(None, None, 1)
        else:
            r_lim_points = round(distance_from_center / self.delta_r)
            mesh_slicer = slice(0, int(r_lim_points + 1), 1)

        return mesh_slicer

    def attach_mesh_to_axis(self, axis, mesh,
                            distance_unit = 'bohr_radius',
                            colormap = plt.get_cmap('inferno'),
                            norm = si.vis.AbsoluteRenormalize(),
                            shading = 'flat',
                            plot_limit = None,
                            slicer = 'get_mesh_slicer',
                            **kwargs):
        unit_value, _ = get_unit_value_and_latex_from_unit(distance_unit)

        _slice = getattr(self, slicer)(plot_limit)

        color_mesh = axis.pcolormesh(self.theta_mesh[_slice],
                                     self.r_mesh[_slice] / unit_value,
                                     mesh[_slice],
                                     shading = shading,
                                     cmap = colormap,
                                     norm = norm,
                                     **kwargs)
        color_mesh_mirror = axis.pcolormesh(twopi - self.theta_mesh[_slice],
                                            self.r_mesh[_slice] / unit_value,
                                            mesh[_slice],
                                            shading = shading,
                                            cmap = colormap,
                                            norm = norm,
                                            **kwargs)  # another colormesh, mirroring the first mesh onto pi to 2pi

        return color_mesh, color_mesh_mirror

    def attach_probability_current_to_axis(self, axis, plot_limit = None, distance_unit = 'bohr_radius'):
        raise NotImplementedError

    def plot_mesh(self, mesh,
                  name = '', title = None,
                  overlay_probability_current = False, probability_current_time_step = 0, plot_limit = None,
                  distance_unit = 'nm',
                  color_map = plt.get_cmap('inferno'),
                  **kwargs):
        plt.close()  # close any old figures

        plt.set_cmap(color_map)

        unit_value, unit_name = get_unit_value_and_latex_from_unit(distance_unit)

        fig = si.vis.get_figure('full')
        fig.set_tight_layout(True)
        axis = plt.subplot(111, projection = 'polar')
        axis.set_theta_zero_location('N')
        axis.set_theta_direction('clockwise')

        color_mesh, color_mesh_mirror = self.attach_mesh_to_axis(axis, mesh, plot_limit = plot_limit, distance_unit = distance_unit)
        # if overlay_probability_current:
        #     quiv = self.attach_probability_current_to_axis(axis, plot_limit = plot_limit, distance_unit = distance_unit)

        if title is not None:
            title = axis.set_title(title, fontsize = 15)
            title.set_x(.03)  # move title to the upper left corner
            title.set_y(.97)

        # make a colorbar
        cbar_axis = fig.add_axes([1.01, .1, .04, .8])  # add a new axis for the cbar so that the old axis can stay square
        cbar = plt.colorbar(mappable = color_mesh, cax = cbar_axis)
        cbar.ax.tick_params(labelsize = 10)

        axis.grid(True, color = si.vis.CMAP_TO_OPPOSITE[color_map], **si.vis.COLORMESH_GRID_KWARGS)  # change grid color to make it show up against the colormesh
        angle_labels = ['{}\u00b0'.format(s) for s in (0, 30, 60, 90, 120, 150, 180, 150, 120, 90, 60, 30)]  # \u00b0 is unicode degree symbol
        axis.set_thetagrids(np.arange(0, 359, 30), frac = 1.075, labels = angle_labels)

        axis.tick_params(axis = 'both', which = 'major', labelsize = 10)  # increase size of tick labels
        axis.tick_params(axis = 'y', which = 'major', colors = si.vis.COLOR_OPPOSITE_INFERNO, pad = 3)  # make r ticks a color that shows up against the colormesh
        axis.tick_params(axis = 'both', which = 'both', length = 0)

        axis.set_rlabel_position(80)

        max_yticks = 5
        yloc = plt.MaxNLocator(max_yticks, symmetric = False, prune = 'both')
        axis.yaxis.set_major_locator(yloc)

        fig.canvas.draw()  # must draw early to modify the axis text

        tick_labels = axis.get_yticklabels()
        for t in tick_labels:
            t.set_text(t.get_text() + r'${}$'.format(unit_name))
        axis.set_yticklabels(tick_labels)

        axis.set_rmax((self.r_max - (self.delta_r / 2)) / unit_value)

        si.vis.save_current_figure(name = '{}_{}'.format(self.spec.name, name), **kwargs)

        plt.close()


class SphericalHarmonicSimulation(ElectricFieldSimulation):
    """Adds options and data storage that are specific to SphericalHarmonicMesh-using simulations."""

    def __init__(self, spec):
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
            self.warnings['norm_in_largest_l'].append(warning_record(self.time_index, msg))

        if self.spec.store_radial_probability_current:
            radial_current_density = self.mesh.get_radial_probability_current_density_mesh__spatial()

            theta = self.mesh.theta_calc
            d_theta = np.abs(theta[1] - theta[0])
            sin_theta = np.sin(theta)
            mask = theta <= pi / 2

            integrand = radial_current_density * sin_theta * d_theta * twopi  # sin(theta) d_theta from theta integral, twopi from phi integral

            self.radial_probability_current_vs_time__pos_z[self.data_time_index] = np.sum(integrand[:, mask], axis = 1) * (self.mesh.r ** 2)
            self.radial_probability_current_vs_time__neg_z[self.data_time_index] = np.sum(integrand[:, ~mask], axis = 1) * (self.mesh.r ** 2)

    @property
    def radial_probability_current_vs_time(self):
        return self.radial_probability_current_vs_time__pos_z + self.radial_probability_current_vs_time__neg_z

    def plot_radial_probability_current_vs_time(
            self,
            time_unit = 'asec',
            time_lower_limit = None,
            time_upper_limit = None,
            r_lower_limit = None,
            r_upper_limit = None,
            distance_unit = 'bohr_radius',
            z_unit = 'per_asec',
            z_limit = None,
            use_name = False,
            which = 'sum',
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
            r_upper_limit = None,
            t_lower_limit = None,
            t_upper_limit = None,
            distance_unit = 'bohr_radius',
            time_unit = 'asec',
            current_unit = 'per_asec',
            z_cut = .7,
            colormap = plt.get_cmap('coolwarm'),
            overlay_electric_field = True,
            efield_unit = 'atomic_electric_field',
            efield_color = 'black',
            efield_label_fontsize = 12,
            title_fontsize = 12,
            y_axis_label_fontsize = 14,
            x_axis_label_fontsize = 12,
            cbar_label_fontsize = 12,
            aspect_ratio = 1.2,
            shading = 'flat',
            use_name = False,
            **kwargs):
        prefix = self.file_name
        if use_name:
            prefix = self.name

        distance_unit_value, distance_unit_latex = get_unit_value_and_latex_from_unit(distance_unit)
        time_unit_value, time_unit_latex = get_unit_value_and_latex_from_unit(time_unit)
        current_unit_value, current_unit_latex = get_unit_value_and_latex_from_unit(current_unit)
        efield_unit_value, efield_unit_latex = get_unit_value_and_latex_from_unit(efield_unit)

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
                ax_efield.set_ylabel(rf'Electric Field Amplitude $ {LATEX_EFIELD}(t) \; ({efield_unit_latex}) $', color = efield_color, fontsize = efield_label_fontsize)
                ax_efield.yaxis.set_label_position('right')

    def plot_angular_momentum_vs_time(self, use_name = False, log = False, renormalize = False, **kwargs):
        fig = plt.figure(figsize = (7, 7 * 2 / 3), dpi = 600)

        grid_spec = matplotlib.gridspec.GridSpec(2, 1, height_ratios = [4, 1], hspace = 0.06)
        ax_momentums = plt.subplot(grid_spec[0])
        ax_field = plt.subplot(grid_spec[1], sharex = ax_momentums)

        if not isinstance(self.spec.electric_potential, potentials.NoPotentialEnergy):
            ax_field.plot(self.times / asec, self.electric_field_amplitude_vs_time / atomic_electric_field, color = 'black', linewidth = 2)

        if renormalize:
            overlaps = [self.norm_by_harmonic_vs_time[sph_harm] / self.norm_vs_time for sph_harm in self.spec.spherical_harmonics]
            l_labels = [r'$\left| \left\langle \Psi| {} \right\rangle \right|^2 / \left\langle \psi| \psi \right\rangle$'.format(sph_harm.latex) for sph_harm in self.spec.spherical_harmonics]
        else:
            overlaps = [self.norm_by_harmonic_vs_time[sph_harm] for sph_harm in self.spec.spherical_harmonics]
            l_labels = [r'$\left| \left\langle \Psi| {} \right\rangle \right|^2$'.format(sph_harm.latex) for sph_harm in self.spec.spherical_harmonics]
        num_colors = len(overlaps)
        ax_momentums.set_prop_cycle(cycler('color', [plt.get_cmap('gist_rainbow')(n / num_colors) for n in range(num_colors)]))
        ax_momentums.stackplot(self.times / asec, *overlaps, alpha = 1, labels = l_labels)

        if log:
            ax_momentums.set_yscale('log')
            ax_momentums.set_ylim(top = 1.0)
            ax_momentums.grid(True, which = 'both')
        else:
            ax_momentums.set_ylim(0, 1.0)
            ax_momentums.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            ax_momentums.grid(True)
        ax_momentums.set_xlim(self.spec.time_initial / asec, self.spec.time_final / asec)

        ax_field.grid(True)

        ax_field.set_xlabel('Time $t$ (as)', fontsize = 15)
        y_label = r'$\left| \left\langle \Psi | Y^l_0 \right\rangle \right|^2$'
        if renormalize:
            y_label += r'$/\left\langle \Psi|\Psi \right\rangle$'
        ax_momentums.set_ylabel(y_label, fontsize = 15)
        ax_field.set_ylabel('${}(t)$ (a.u.)'.format(LATEX_EFIELD), fontsize = 11)

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


class SphericalHarmonicSpecification(ElectricFieldSpecification):
    simulation_type = SphericalHarmonicSimulation

    def __init__(self, name,
                 r_bound = 100 * bohr_radius,
                 r_points = 400,
                 l_bound = 100,
                 theta_points = 180,
                 evolution_equations = 'LAG',
                 evolution_method = 'SO',
                 evolution_gauge = 'LEN',
                 use_numeric_eigenstates = False,
                 numeric_eigenstate_max_angular_momentum = 20,
                 numeric_eigenstate_max_energy = 100 * eV,
                 hydrogen_zero_angular_momentum_correction = True,
                 store_radial_probability_current = False,
                 store_norm_by_l = False,
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
            mesh_type = SphericalHarmonicMesh,
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

    def info(self):
        info = super().info()

        info_mesh = si.Info(header = f'Mesh: {self.mesh_type.__name__}')
        info_mesh.add_field('R Boundary', f'{uround(self.r_bound, bohr_radius, 3)} a_0 | {uround(self.r_bound, nm, 3)} nm')
        info_mesh.add_field('R Points', self.r_points)
        info_mesh.add_field('R Mesh Spacing', f'~{uround(self.r_bound / self.r_points, bohr_radius, 3)} a_0')
        info_mesh.add_field('L Bound', self.l_bound)
        info_mesh.add_field('Total Mesh Points', self.r_points * self.l_bound)

        info.add_info(info_mesh)

        info_eigenstates = si.Info(header = f'Numeric Eigenstates: {self.use_numeric_eigenstates}')
        if self.use_numeric_eigenstates:
            info_eigenstates.add_field('Max Energy', f'{uround(self.numeric_eigenstate_max_energy, eV)} eV')
            info_eigenstates.add_field('Max Angular Momentum', self.numeric_eigenstate_max_angular_momentum)

        info.add_info(info_eigenstates)

        return info


class SphericalHarmonicMesh(QuantumMesh):
    mesh_storage_method = ('l', 'r')

    def __init__(self, simulation):
        super().__init__(simulation)

        self.r = np.linspace(0, self.spec.r_bound, self.spec.r_points)
        self.delta_r = self.r[1] - self.r[0]
        self.r += self.delta_r / 2
        self.r_max = np.max(self.r)
        self.inner_product_multiplier = self.delta_r

        self.l = np.array(range(self.spec.l_bound), dtype = int)
        self.theta_points = self.phi_points = self.spec.theta_points

        self.mesh_points = len(self.r) * len(self.l)
        self.mesh_shape = np.shape(self.r_mesh)

        if self.spec.use_numeric_eigenstates:
            logger.debug('Calculating numeric eigenstates')

            self.analytic_to_numeric = self.get_numeric_eigenstate_basis(self.spec.numeric_eigenstate_max_energy, self.spec.numeric_eigenstate_max_angular_momentum)
            self.spec.test_states = sorted(list(self.analytic_to_numeric.values()), key = lambda x: x.energy)
            self.spec.initial_state = self.analytic_to_numeric[self.spec.initial_state]

            logger.warning('Replaced test states for {} with numeric eigenbasis'.format(self))

        self.g = self.get_g_for_state(self.spec.initial_state)

    @property
    @si.utils.memoize
    def r_mesh(self):
        return np.meshgrid(self.l, self.r, indexing = 'ij')[1]

    @property
    @si.utils.memoize
    def l_mesh(self):
        return np.meshgrid(self.l, self.r, indexing = 'ij')[0]

    @property
    def g_factor(self):
        return self.r

    def flatten_mesh(self, mesh, flatten_along):
        """Return a mesh flattened along one of the mesh coordinates ('theta' or 'r')."""
        try:
            if flatten_along == 'l':
                flat = 'F'
            elif flatten_along == 'r':
                flat = 'C'
            elif flatten_along is None:
                return mesh
            else:
                raise ValueError("{} is not a valid specifier for flatten_mesh (valid specifiers: 'l', 'r')".format(flatten_along))

            return mesh.flatten(flat)
        except AttributeError:  # occurs if the "mesh" is actually an int or float, in which case we should should just return it
            return mesh

    def wrap_vector(self, vector, wrap_along):
        if wrap_along == 'l':
            wrap = 'F'
        elif wrap_along == 'r':
            wrap = 'C'
        elif wrap_along is None:
            return vector
        else:
            raise ValueError("{} is not a valid specifier for wrap_vector (valid specifiers: 'l', 'r')".format(wrap_along))

        return np.reshape(vector, self.mesh_shape, wrap)

    def get_g_for_state(self, state):
        """

        :param state:
        :return:
        """
        # don't memoize this, instead rely on the memoization in get_radial_function_for_state, more compact in memory (at the cost of some runtime in having to reassemble g occasionally)

        if isinstance(state, states.QuantumState) and all(hasattr(s, 'spherical_harmonic') for s in state):
            if state.analytic and self.spec.use_numeric_eigenstates:
                try:
                    state = self.analytic_to_numeric[state]
                except (AttributeError, KeyError):
                    logger.debug(f'Analytic to numeric eigenstate lookup failed for state {state}')
            g = np.zeros(self.mesh_shape, dtype = np.complex128)

            for s in state:
                g[s.l, :] += self.get_radial_g_for_state(s)  # fill in g state-by-state to improve runtime

            return g
        else:
            raise NotImplementedError('States with non-definite angular momentum components are not currently supported by SphericalHarmonicMesh')

    @si.utils.memoize
    def get_radial_g_for_state(self, state):
        """Return the radial g function evaluated on the radial mesh for a state that has a radial function."""
        # logger.debug('Calculating radial wavefunction for state {}'.format(state))
        g = state.radial_function(self.r) * self.g_factor
        g /= np.sqrt(self.norm(g))
        g *= state.amplitude

        return g

    def inner_product(self, a = None, b = None):
        """
        Return the inner product between two states (a and b) on the mesh.

        a and b can be QuantumStates or g_meshes.

        :param a: a QuantumState or g
        :param b: a QuantumState or g
        :return: the inner product between a and b
        """
        if isinstance(a, states.QuantumState) and all(hasattr(s, 'spherical_harmonic') for s in a) and b is None:  # shortcut
            ip = 0

            for s in a:
                ip += np.sum(np.conj(self.get_radial_g_for_state(s)) * self.g[s.l, :])  # calculate inner product state-by-state to improve runtime

            return ip * self.inner_product_multiplier
        else:
            return super().inner_product(a, b)

    @property
    def norm_by_l(self):
        return np.abs(np.sum(np.conj(self.g) * self.g, axis = 1) * self.delta_r)

    def dipole_moment_inner_product(self, a = None, b = None):
        operator = self._get_interaction_hamiltonian_matrix_operators_without_field_LEN()
        b = self.wrap_vector(operator.dot(self.flatten_mesh(self.state_to_mesh(b), 'l')), 'l')
        return -self.inner_product(a = a, b = b)

    def inner_product_with_plane_waves(self, thetas, wavenumbers, g = None):
        """
        Return the inner products for each plane wave state in the Cartesian product of thetas and wavenumbers.

        Parameters
        ----------
        thetas
        wavenumbers
        g

        Returns
        -------

        """
        if g is None:
            g = self.g

        l_mesh = self.l_mesh

        multiplier = np.sqrt(2 / pi) * self.g_factor * (-1j ** (l_mesh % 4)) * self.inner_product_multiplier * g

        thetas, wavenumbers = np.array(thetas), np.array(wavenumbers)
        theta_mesh, wavenumber_mesh = np.meshgrid(thetas, wavenumbers, indexing = 'ij')

        inner_product_mesh = np.zeros(np.shape(wavenumber_mesh), dtype = np.complex128)

        @si.utils.memoize
        def sph_harm(theta):
            return special.sph_harm(0, l_mesh, 0, theta)

        @si.utils.memoize
        def bessel(wavenumber):
            return special.spherical_jn(l_mesh, np.real(wavenumber * self.r_mesh))

        # for ii, theta in enumerate(thetas):
        #     for jj, wavenumber in enumerate(wavenumbers):
        #         inner_product_mesh[ii, jj] = np.sum(multiplier * sph_harm(theta) * bessel(wavenumber))

        for (ii, theta), (jj, wavenumber) in it.product(enumerate(thetas), enumerate(wavenumbers)):
            inner_product_mesh[ii, jj] = np.sum(multiplier * sph_harm(theta) * bessel(wavenumber))

        return theta_mesh, wavenumber_mesh, inner_product_mesh

    def inner_product_with_plane_waves_at_infinity(self, thetas, wavenumbers, g = None):
        """
        Return the inner products for each plane wave state in the Cartesian product of thetas and wavenumbers.

        WARNING: NOT WORKING

        Parameters
        ----------
        thetas
        wavenumbers
        g

        Returns
        -------

        """
        l_mesh = self.l_mesh

        # multiplier = np.sqrt(2 / pi) * self.g_factor * (-1j ** (l_mesh % 4)) * self.inner_product_multiplier * g

        thetas, wavenumbers = np.array(thetas), np.array(wavenumbers)
        theta_mesh, wavenumber_mesh = np.meshgrid(thetas, wavenumbers, indexing = 'ij')

        inner_product_mesh = np.zeros(np.shape(wavenumber_mesh), dtype = np.complex128)

        # @si.utils.memoize
        # def sph_harm(theta):
        #     return special.sph_harm(0, l_mesh, 0, theta)
        #
        # @si.utils.memoize
        # def bessel(wavenumber):
        #     return special.spherical_jn(l_mesh, np.real(wavenumber * self.r_mesh))

        # @si.utils.memoize
        # def poly(l, theta):
        #     return special.legendre(l)(np.cos(theta))
        #
        # @si.utils.memoize
        # def phase(l, k):
        #     return np.exp(1j * states.coulomb_phase_shift(l, k))
        #
        # # sqrt_mesh = np.sqrt((2 * l_mesh) + 1)
        #
        # for ii, theta in enumerate(thetas):
        #     for jj, wavenumber in enumerate(wavenumbers):
        #         print(ii, jj)
        #
        #         total = 0
        #         for l in self.l:
        #             total += phase(l, wavenumber) * np.sqrt((2 * l) + 1) * poly(l, theta) * self.inner_product(states.HydrogenCoulombState.from_wavenumber(wavenumber, l), g)
        #
        #         inner_product_mesh[ii, jj] = total / np.sqrt(4 * pi * wavenumber)

        if g is None:
            g = self.g

        sqrt_mesh = np.sqrt((2 * l_mesh) + 1)

        @si.utils.memoize
        def poly(theta):
            return special.lpn(l_mesh, np.cos(theta))

        @si.utils.memoize
        def phase(k):
            return np.exp(1j * states.coulomb_phase_shift(l_mesh, k))

        for ii, theta in enumerate(thetas):
            for jj, wavenumber in enumerate(wavenumbers):
                print(ii, jj)

                # total = 0
                # for l in self.l:
                #     total += phase(l, wavenumber) * np.sqrt((2 * l) + 1) * poly(l, theta) * self.inner_product(states.HydrogenCoulombState.from_wavenumber(wavenumber, l), g)

                state = states.HydrogenCoulombState.from_wavenumber(wavenumber, l = 0)
                for l in self.l[1:]:
                    state += states.HydrogenCoulombState.from_wavenumber(wavenumber, l)

                print(state)
                state_mesh = self.get_g_for_state(state)
                ip = self.inner_product(poly(theta) * phase(wavenumber) * sqrt_mesh * state_mesh, g)

                inner_product_mesh[ii, jj] = ip / np.sqrt(4 * pi * wavenumber)

        return theta_mesh, wavenumber_mesh, inner_product_mesh

    def _get_kinetic_energy_matrix_operators_HAM(self):
        r_prefactor = -(hbar ** 2) / (2 * electron_mass_reduced * (self.delta_r ** 2))

        r_diagonal = r_prefactor * (-2) * np.ones(self.mesh_points, dtype = np.complex128)
        r_offdiagonal = r_prefactor * np.ones(self.mesh_points - 1, dtype = np.complex128)

        effective_potential_mesh = ((hbar ** 2) / (2 * electron_mass_reduced)) * self.l_mesh * (self.l_mesh + 1) / (self.r_mesh ** 2)
        r_diagonal += self.flatten_mesh(effective_potential_mesh, 'r')

        r_kinetic = sparse.diags([r_offdiagonal, r_diagonal, r_offdiagonal], offsets = (-1, 0, 1))

        return r_kinetic

    def alpha(self, j):
        x = (j ** 2) + (2 * j)
        return (x + 1) / (x + 0.75)

    def beta(self, j):
        x = 2 * (j ** 2) + (2 * j)
        return (x + 1) / (x + 0.5)

    def gamma(self, j):
        """For radial probability current."""
        return 1 / ((j ** 2) - 0.25)

    def _get_kinetic_energy_matrix_operator_single_l(self, l):
        r_prefactor = -(hbar ** 2) / (2 * electron_mass_reduced * (self.delta_r ** 2))
        effective_potential = ((hbar ** 2) / (2 * electron_mass_reduced)) * l * (l + 1) / (self.r ** 2)

        r_beta = self.beta(np.array(range(len(self.r)), dtype = np.complex128))
        if l == 0 and self.spec.hydrogen_zero_angular_momentum_correction:
            dr = self.delta_r / bohr_radius
            r_beta[0] += dr * (1 + dr) / 8
        r_diagonal = (-2 * r_prefactor * r_beta) + effective_potential
        r_offdiagonal = r_prefactor * self.alpha(np.array(range(len(self.r) - 1), dtype = np.complex128))

        return sparse.diags([r_offdiagonal, r_diagonal, r_offdiagonal], offsets = (-1, 0, 1))

    def _get_internal_hamiltonian_matrix_operator_single_l(self, l):
        r_kinetic = self._get_kinetic_energy_matrix_operator_single_l(l)
        potential = self.spec.internal_potential(r = self.r, test_charge = self.spec.test_charge)

        r_kinetic.data[1] += potential

        return r_kinetic

    def _get_kinetic_energy_matrix_operators_LAG(self, include_effective_potential = True):
        """Get the radial kinetic energy matrix operator."""
        r_prefactor = -(hbar ** 2) / (2 * electron_mass_reduced * (self.delta_r ** 2))

        r_diagonal = np.zeros(self.mesh_points, dtype = np.complex128)
        r_offdiagonal = np.zeros(self.mesh_points - 1, dtype = np.complex128)
        for r_index in range(self.mesh_points):
            j = r_index % self.spec.r_points
            r_diagonal[r_index] = self.beta(j)
        if self.spec.hydrogen_zero_angular_momentum_correction:
            dr = self.delta_r / bohr_radius
            r_diagonal[0] += dr * (1 + dr) / 8  # modify beta_j for l = 0   (see notes)

        for r_index in range(self.mesh_points - 1):
            if (r_index + 1) % self.spec.r_points != 0:
                j = (r_index % self.spec.r_points)
                r_offdiagonal[r_index] = self.alpha(j)
        r_diagonal *= -2 * r_prefactor
        r_offdiagonal *= r_prefactor

        if include_effective_potential:
            effective_potential_mesh = ((hbar ** 2) / (2 * electron_mass_reduced)) * self.l_mesh * (self.l_mesh + 1) / (self.r_mesh ** 2)
            r_diagonal += self.flatten_mesh(effective_potential_mesh, 'r')

        r_kinetic = sparse.diags([r_offdiagonal, r_diagonal, r_offdiagonal], offsets = (-1, 0, 1))

        return r_kinetic

    @si.utils.memoize
    def get_internal_hamiltonian_matrix_operators(self):
        r_kinetic = self.get_kinetic_energy_matrix_operators().copy()

        potential_mesh = self.spec.internal_potential(r = self.r_mesh, test_charge = self.spec.test_charge)

        r_kinetic.data[1] += self.flatten_mesh(potential_mesh, 'r')

        return r_kinetic

    @si.utils.memoize
    def _get_interaction_hamiltonian_matrix_operators_without_field_LEN(self):
        l_prefactor = -self.spec.test_charge * self.flatten_mesh(self.r_mesh, 'l')[:-1]

        l_diagonal = np.zeros(self.mesh_points, dtype = np.complex128)
        l_offdiagonal = np.zeros(self.mesh_points - 1, dtype = np.complex128)
        for l_index in range(self.mesh_points - 1):
            if (l_index + 1) % self.spec.l_bound != 0:
                l = (l_index % self.spec.l_bound)
                l_offdiagonal[l_index] = c_l(l)
        l_offdiagonal *= l_prefactor

        return sparse.diags([l_offdiagonal, l_diagonal, l_offdiagonal], offsets = (-1, 0, 1))

    def _get_interaction_hamiltonian_matrix_operators_LEN(self):
        """Get the angular momentum interaction term calculated from the Lagrangian evolution equations in the length gauge."""
        return self._get_interaction_hamiltonian_matrix_operators_without_field_LEN() * self.spec.electric_potential.get_electric_field_amplitude(self.sim.time + (self.spec.time_step / 2))

    @si.utils.memoize
    def _get_interaction_hamiltonian_matrix_operators_without_field_VEL(self):
        h1_prefactor = 1j * hbar * (self.spec.test_charge / self.spec.test_mass) / self.flatten_mesh(self.r_mesh, 'l')[:-1]

        h1_offdiagonal = np.zeros(self.mesh_points - 1, dtype = np.complex128)
        for l_index in range(self.mesh_points - 1):
            if (l_index + 1) % self.spec.l_bound != 0:
                l = (l_index % self.spec.l_bound)
                h1_offdiagonal[l_index] = c_l(l) * (l + 1)
        h1_offdiagonal *= h1_prefactor

        h1 = sparse.diags((-h1_offdiagonal, h1_offdiagonal), offsets = (-1, 1))

        h2_prefactor = 1j * hbar * (self.spec.test_charge / self.spec.test_mass) / (2 * self.delta_r)

        alpha_vec = self.alpha(np.array(range(len(self.r) - 1), dtype = np.complex128))
        alpha_block = sparse.diags((-alpha_vec, alpha_vec), offsets = (-1, 1))

        c_vec = c_l(np.array(range(len(self.l) - 1), dtype = np.complex128))
        c_block = sparse.diags((c_vec, c_vec), offsets = (-1, 1))

        h2 = h2_prefactor * sparse.kron(c_block, alpha_block, format = 'dia')

        return h1, h2

    def _get_interaction_hamiltonian_matrix_operators_VEL(self):
        vector_potential_amp = self.spec.electric_potential.get_vector_potential_amplitude_numeric(self.sim.times_to_current)
        return (x * vector_potential_amp for x in self._get_interaction_hamiltonian_matrix_operators_without_field_VEL())

    def get_numeric_eigenstate_basis(self, max_energy, l_max):
        analytic_to_numeric = {}

        for l in range(l_max + 1):
            h = self._get_internal_hamiltonian_matrix_operator_single_l(l = l)

            estimated_spacing = twopi / self.r_max
            wavenumber_max = np.real(electron_wavenumber_from_energy(max_energy))
            number_of_eigenvectors = int(wavenumber_max / estimated_spacing)  # generate an initial guess based on roughly linear wavenumber steps between eigenvalues

            max_eigenvectors = h.shape[0] - 2  # can't generate more than this many eigenvectors using sparse linear algebra methods

            while True:
                if number_of_eigenvectors > max_eigenvectors:
                    number_of_eigenvectors = max_eigenvectors  # this will cause the loop to break after this attempt

                eigenvalues, eigenvectors = sparsealg.eigsh(h, k = number_of_eigenvectors, which = 'SA')

                if np.max(eigenvalues) > max_energy or number_of_eigenvectors == max_eigenvectors:
                    break

                number_of_eigenvectors = int(
                    number_of_eigenvectors * 1.1 * np.sqrt(np.abs(max_energy / np.max(eigenvalues))))  # based on approximate sqrt scaling of energy to wavenumber, with safety factor

            for eigenvalue, eigenvector in zip(eigenvalues, eigenvectors.T):
                eigenvector /= np.sqrt(self.inner_product_multiplier * np.sum(np.abs(eigenvector) ** 2))  # normalize
                eigenvector /= self.g_factor  # go to u from R

                if eigenvalue > max_energy:  # ignore eigenvalues that are too large
                    continue
                elif eigenvalue > 0:
                    analytic_state = states.HydrogenCoulombState(energy = eigenvalue, l = l)
                    bound = False
                else:
                    n_guess = round(np.sqrt(rydberg / np.abs(eigenvalue)))
                    if n_guess == 0:
                        n_guess = 1
                    analytic_state = states.HydrogenBoundState(n = n_guess, l = l)
                    bound = True

                numeric_state = states.NumericSphericalHarmonicState(eigenvector, l, 0, eigenvalue, analytic_state, bound = bound)

                analytic_to_numeric[analytic_state] = numeric_state

            logger.debug('Generated numerical eigenbasis for l = {}, energy <= {} eV'.format(l, uround(max_energy, 'eV', 3)))

        logger.debug('Generated numerical eigenbasis for l <= {}, energy <= {} eV. Found {} states.'.format(l_max, uround(max_energy, 'eV', 3), len(analytic_to_numeric)))

        return analytic_to_numeric

    def tg_mesh(self, use_abs_g = False):
        if use_abs_g:
            g = np.abs(self.g)
        else:
            g = self.g

        hamiltonian_r = self.get_kinetic_energy_matrix_operators()

        return self.wrap_vector(hamiltonian_r.dot(self.flatten_mesh(g, 'r')), 'r')

        # g_vector_r = self.flatten_mesh(g, 'r')
        # tg_vector_r = hamiltonian_r.dot(g_vector_r)
        # tg_mesh = self.wrap_vector(tg_vector_r, 'r')

        # return tg_mesh

    def hg_mesh(self, include_interaction = False):
        hamiltonian_r = self.get_internal_hamiltonian_matrix_operators()

        hg = self.wrap_vector(hamiltonian_r.dot(self.flatten_mesh(self.g, 'r')), 'r')

        if include_interaction:
            hamiltonian_l = self.get_interaction_hamiltonian_matrix_operators()
            wrapping_direction = 'l' if self.spec.evolution_gauge == 'LEN' else 'r'

            hg += self.wrap_vector(hamiltonian_l.dot(self.flatten_mesh(self.g, wrapping_direction)), wrapping_direction)

        return hg

    def energy_expectation_value(self, include_interaction = False):
        return np.real(self.inner_product(b = self.hg_mesh(include_interaction = include_interaction))) / self.norm()

    @si.utils.memoize
    def _get_probability_current_matrix_operators(self):
        raise NotImplementedError

        if self.spec.evolution_gauge == 'VEL':
            # add extra term -2 q A |psi|^2
            raise NotImplementedError('Velocity gauge probability current not yet implemented')

    def get_probability_current_vector_field(self):
        raise NotImplementedError

    @si.utils.memoize
    def _get_radial_probability_current_operator__spatial(self):
        r_prefactor = hbar / (2 * self.spec.test_mass * (self.delta_r ** 3))  # / extra 2 from taking Im later

        r_offdiagonal = np.zeros((self.spec.r_points * self.spec.theta_points) - 1, dtype = np.complex128)

        for r_index in range((self.spec.r_points * self.spec.theta_points) - 1):
            if (r_index + 1) % self.spec.r_points != 0:
                j = (r_index % self.spec.r_points) + 1
                r_offdiagonal[r_index] = self.gamma(j)

        r_offdiagonal *= r_prefactor

        r_current_operator = sparse.diags([-r_offdiagonal, r_offdiagonal], offsets = [-1, 1])

        return r_current_operator

    def get_radial_probability_current_density_mesh__spatial(self):
        r_current_operator = self._get_radial_probability_current_operator__spatial()

        g_spatial = self.space_g_calc
        g_spatial_shape = g_spatial.shape

        g_vector_r = g_spatial.flatten('F')
        gradient_vector_r = r_current_operator.dot(g_vector_r)
        gradient_mesh_r = np.reshape(gradient_vector_r, g_spatial_shape, 'F')
        current_mesh_r = np.imag(np.conj(g_spatial) * gradient_mesh_r)

        return current_mesh_r

    def _evolve_CN(self, time_step):
        if self.spec.evolution_gauge == "VEL":
            raise NotImplementedError

        tau = 1j * time_step / (2 * hbar)

        hamiltonian_r = tau * self.get_internal_hamiltonian_matrix_operators()
        hamiltonian_l = tau * self.get_interaction_hamiltonian_matrix_operators()

        hamiltonian_l_explicit = add_to_diagonal_sparse_matrix_diagonal(-hamiltonian_l, 1)
        hamiltonian_r_implicit = add_to_diagonal_sparse_matrix_diagonal(hamiltonian_r, 1)
        hamiltonian_r_explicit = add_to_diagonal_sparse_matrix_diagonal(-hamiltonian_r, 1)
        hamiltonian_l_implicit = add_to_diagonal_sparse_matrix_diagonal(hamiltonian_l, 1)

        operators = [
            DotOperator(hamiltonian_l_explicit, wrapping_direction = 'l'),
            TDMAOperator(hamiltonian_r_implicit, wrapping_direction = 'r'),
            DotOperator(hamiltonian_r_explicit, wrapping_direction = 'r'),
            TDMAOperator(hamiltonian_l_implicit, wrapping_direction = 'l'),
        ]

        self.g = apply_operators(self, self.g, *operators)

    def _make_split_operator_evolution_operators(self, interaction_hamiltonian_matrix_operators, tau):
        return getattr(self, f'_make_split_operator_evolution_operators_{self.spec.evolution_gauge}')(interaction_hamiltonian_matrix_operators, tau)

    def _make_split_operator_evolution_operators_LEN(self, interaction_hamiltonians_matrix_operators, tau):
        """Calculate split operator evolution matrices for the interaction term in the length gauge."""
        a = interaction_hamiltonians_matrix_operators.data[0][:-1] * tau

        a_even, a_odd = a[::2], a[1::2]

        even_diag = np.zeros(len(a) + 1, dtype = np.complex128)
        even_offdiag = np.zeros(len(a), dtype = np.complex128)
        odd_diag = np.zeros(len(a) + 1, dtype = np.complex128)
        odd_offdiag = np.zeros(len(a), dtype = np.complex128)

        if len(self.r) % 2 != 0 and len(self.l) % 2 != 0:
            even_diag[:-1] = np.cos(a_even).repeat(2)
            even_diag[-1] = 1

            even_offdiag[::2] = -1j * np.sin(a_even)

            odd_diag[0] = 1
            odd_diag[1:] = np.cos(a_odd).repeat(2)

            odd_offdiag[1::2] = -1j * np.sin(a_odd)
        else:
            even_diag[:] = np.cos(a_even).repeat(2)

            even_offdiag[::2] = -1j * np.sin(a_even)

            odd_diag[0] = odd_diag[-1] = 1
            odd_diag[1:-1] = np.cos(a_odd).repeat(2)

            odd_offdiag[1::2] = -1j * np.sin(a_odd)

        even = sparse.diags([even_offdiag, even_diag, even_offdiag], offsets = [-1, 0, 1])
        odd = sparse.diags([odd_offdiag, odd_diag, odd_offdiag], offsets = [-1, 0, 1])

        return (
            DotOperator(even, wrapping_direction = 'l'),
            DotOperator(odd, wrapping_direction = 'l'),
        )

    def _make_split_operators_VEL_h1(self, h1, tau):
        a = h1.data[-1][1:] * tau * (-1j)

        a_even, a_odd = a[::2], a[1::2]

        even_diag = np.zeros(len(a) + 1, dtype = np.complex128)
        even_offdiag = np.zeros(len(a), dtype = np.complex128)
        odd_diag = np.zeros(len(a) + 1, dtype = np.complex128)
        odd_offdiag = np.zeros(len(a), dtype = np.complex128)

        if len(self.r) % 2 != 0 and len(self.l) % 2 != 0:
            even_diag[:-1] = np.cos(a_even).repeat(2)
            even_diag[-1] = 1

            even_offdiag[::2] = np.sin(a_even)

            odd_diag[0] = 1
            odd_diag[1:] = np.cos(a_odd).repeat(2)

            odd_offdiag[1::2] = np.sin(a_odd)
        else:
            even_diag[:] = np.cos(a_even).repeat(2)

            even_offdiag[::2] = np.sin(a_even)

            odd_diag[0] = odd_diag[-1] = 1
            odd_diag[1:-1] = np.cos(a_odd).repeat(2)

            odd_offdiag[1::2] = np.sin(a_odd)

        even = sparse.diags([-even_offdiag, even_diag, even_offdiag], offsets = [-1, 0, 1])
        odd = sparse.diags([-odd_offdiag, odd_diag, odd_offdiag], offsets = [-1, 0, 1])

        return (
            DotOperator(even, wrapping_direction = 'l'),
            DotOperator(odd, wrapping_direction = 'l'),
        )

    def _make_split_operators_VEL_h2(self, h2, tau):
        len_r = len(self.r)

        a = h2.data[-1][len_r + 1:] * tau * (-1j)

        alpha_slices_even_l = []
        alpha_slices_odd_l = []
        for l in self.l:  # want last l but not last r, since unwrapped in r
            a_slice = a[l * len_r: ((l + 1) * len_r) - 1]
            if l % 2 == 0:
                alpha_slices_even_l.append(a_slice)
            else:
                alpha_slices_odd_l.append(a_slice)

        even_even_diag = []
        even_even_offdiag = []
        even_odd_diag = []
        even_odd_offdiag = []
        for alpha_slice in alpha_slices_even_l:  # FOR EACH l
            even_slice = alpha_slice[::2]
            odd_slice = alpha_slice[1::2]

            if len(even_slice) > 0:
                even_sines = np.zeros(len_r, dtype = np.complex128)
                if len_r % 2 == 0:
                    new_even_even_diag = np.tile(np.cos(even_slice).repeat(2), 2)
                    even_sines[::2] = np.sin(even_slice)
                else:
                    tile = np.ones(len_r, dtype = np.complex128)
                    tile[:-1] = np.cos(even_slice).repeat(2)
                    tile[-1] = 1
                    new_even_even_diag = np.tile(tile, 2)
                    even_sines[:-1:2] = np.sin(even_slice)
                    even_sines[-1] = 0

                even_even_diag.append(new_even_even_diag)
                even_even_offdiag.append(even_sines)
                even_even_offdiag.append(-even_sines)
            else:
                even_even_diag.append(np.ones(len_r))
                even_even_offdiag.append(np.zeros(len_r))

            if len(odd_slice) > 0:
                new_even_odd_diag = np.ones(len_r, dtype = np.complex128)

                if len_r % 2 == 0:
                    new_even_odd_diag[1:-1] = np.cos(odd_slice).repeat(2)
                else:
                    new_even_odd_diag[1::] = np.cos(odd_slice).repeat(2)

                new_even_odd_diag = np.tile(new_even_odd_diag, 2)

                even_odd_diag.append(new_even_odd_diag)

                odd_sines = np.zeros(len_r, dtype = np.complex128)
                odd_sines[1:-1:2] = np.sin(odd_slice)
                even_odd_offdiag.append(odd_sines)
                even_odd_offdiag.append(-odd_sines)
        if self.l[-1] % 2 == 0:
            even_odd_diag.append(np.ones(len_r))
            even_odd_offdiag.append(np.zeros(len_r))

        even_even_diag = np.hstack(even_even_diag)
        even_even_offdiag = np.hstack(even_even_offdiag)[:-1]  # last element is bogus

        even_odd_diag = np.hstack(even_odd_diag)
        even_odd_offdiag = np.hstack(even_odd_offdiag)[:-1]  # last element is bogus

        odd_even_diag = [np.ones(len_r)]
        odd_even_offdiag = [np.zeros(len_r)]
        odd_odd_diag = [np.ones(len_r)]
        odd_odd_offdiag = [np.zeros(len_r)]

        for alpha_slice in alpha_slices_odd_l:
            even_slice = alpha_slice[::2]
            odd_slice = alpha_slice[1::2]

            if len(even_slice) > 0:
                even_sines = np.zeros(len_r, dtype = np.complex128)
                if len_r % 2 == 0:
                    new_odd_even_diag = np.tile(np.cos(even_slice).repeat(2), 2)
                    even_sines[::2] = np.sin(even_slice)
                else:
                    tile = np.ones(len_r, dtype = np.complex128)
                    tile[:-1] = np.cos(even_slice).repeat(2)
                    # tile[-1] = 1
                    new_odd_even_diag = np.tile(tile, 2)
                    even_sines[:-1:2] = np.sin(even_slice)
                    # even_sines[-1] = 0

                odd_even_diag.append(new_odd_even_diag)
                odd_even_offdiag.append(even_sines)
                odd_even_offdiag.append(-even_sines)
            else:
                odd_even_diag.append(np.ones(len_r))
                odd_even_offdiag.append(np.zeros(len_r))

            if len(odd_slice) > 0:
                new_odd_odd_diag = np.ones(len_r, dtype = np.complex128)

                if len_r % 2 == 0:
                    new_odd_odd_diag[1:-1] = np.cos(odd_slice).repeat(2)
                else:
                    new_odd_odd_diag[1::] = np.cos(odd_slice).repeat(2)

                new_odd_odd_diag = np.tile(new_odd_odd_diag, 2)

                odd_odd_diag.append(new_odd_odd_diag)

                odd_sines = np.zeros(len_r, dtype = np.complex128)
                odd_sines[1:-1:2] = np.sin(odd_slice)
                odd_odd_offdiag.append(odd_sines)
                odd_odd_offdiag.append(-odd_sines)
        if self.l[-1] % 2 != 0:
            odd_odd_diag.append(np.ones(len_r))
            odd_odd_offdiag.append(np.zeros(len_r))

        odd_even_diag = np.hstack(odd_even_diag)
        odd_even_offdiag = np.hstack(odd_even_offdiag)[:-1]  # last element is bogus

        odd_odd_diag = np.hstack(odd_odd_diag)
        odd_odd_offdiag = np.hstack(odd_odd_offdiag)[:-1]  # last element is bogus

        even_even_matrix = sparse.diags((-even_even_offdiag, even_even_diag, even_even_offdiag), offsets = (-1, 0, 1))
        even_odd_matrix = sparse.diags((-even_odd_offdiag, even_odd_diag, even_odd_offdiag), offsets = (-1, 0, 1))
        odd_even_matrix = sparse.diags((-odd_even_offdiag, odd_even_diag, odd_even_offdiag), offsets = (-1, 0, 1))
        odd_odd_matrix = sparse.diags((-odd_odd_offdiag, odd_odd_diag, odd_odd_offdiag), offsets = (-1, 0, 1))

        operators = (
            SimilarityOperator(even_even_matrix, wrapping_direction = 'r', parity = 'even'),
            SimilarityOperator(even_odd_matrix, wrapping_direction = 'r', parity = 'even'),  # parity is based off FIRST splitting
            SimilarityOperator(odd_even_matrix, wrapping_direction = 'r', parity = 'odd'),
            SimilarityOperator(odd_odd_matrix, wrapping_direction = 'r', parity = 'odd'),
        )

        return operators

    def _make_split_operator_evolution_operators_VEL(self, interaction_hamiltonians_matrix_operators, tau):
        """Calculate split operator evolution matrices for the interaction term in the velocity gauge."""
        h1, h2 = interaction_hamiltonians_matrix_operators

        h1_operators = self._make_split_operators_VEL_h1(h1, tau)
        h2_operators = self._make_split_operators_VEL_h2(h2, tau)

        return [*h1_operators, *h2_operators]

    def _evolve_SO(self, time_step):
        """Evolve the mesh forward in time by using a split-operator algorithm with length-gauge evolution operators."""
        tau = time_step / (2 * hbar)

        hamiltonian_r = (1j * tau) * self.get_internal_hamiltonian_matrix_operators()

        hamiltonian_r_explicit = add_to_diagonal_sparse_matrix_diagonal(-hamiltonian_r, 1)
        hamiltonian_r_implicit = add_to_diagonal_sparse_matrix_diagonal(hamiltonian_r, 1)

        split_operators = self._make_split_operator_evolution_operators(self.get_interaction_hamiltonian_matrix_operators(), tau)

        operators = (
            *split_operators,
            DotOperator(hamiltonian_r_explicit, wrapping_direction = 'r'),
            TDMAOperator(hamiltonian_r_implicit, wrapping_direction = 'r'),
            *reversed(split_operators),
        )

        self.g = apply_operators(self, self.g, *operators)

    def _apply_length_gauge_transformation(self, vamp, g):
        bessel_mesh = special.spherical_jn(self.l_mesh, self.spec.test_charge * vamp * self.r_mesh / hbar)

        g_transformed = np.zeros(np.shape(g), dtype = np.complex128)
        for l_result in self.l:
            for l_outer in self.l:  # l'
                prefactor = np.sqrt(4 * pi * ((2 * l_outer) + 1)) * ((1j) ** (l_outer % 4)) * bessel_mesh[l_outer, :]
                for l_inner in self.l:  # l
                    print(l_result, l_outer, l_inner)
                    g_transformed[l_result, :] += g[l_inner, :] * prefactor * triple_y_integral(l_outer, 0, l_result, 0, l_inner, 0)

        return g_transformed

    def gauge_transformation(self, *, g = None, leaving_gauge = None):
        if g is None:
            g = self.g
        if leaving_gauge is None:
            leaving_gauge = self.spec.evolution_gauge

        vamp = self.spec.electric_potential.get_vector_potential_amplitude_numeric_cumulative(self.sim.times_to_current)
        integral = integ.simps(y = vamp ** 2,
                               x = self.sim.times_to_current)

        dipole_to_velocity = np.exp(1j * integral * (self.spec.test_charge ** 2) / (2 * self.spec.test_mass * hbar))

        if leaving_gauge == 'LEN':
            return self._apply_length_gauge_transformation(-vamp[-1], dipole_to_velocity * g)
        elif leaving_gauge == 'VEL':
            return dipole_to_velocity * self._apply_length_gauge_transformation(vamp[-1], g)

    @si.utils.memoize
    def get_mesh_slicer(self, distance_from_center = None):
        """Returns a slice object that slices a mesh to the given distance of the center."""
        if distance_from_center is None:
            mesh_slicer = (slice(None, None, 1), slice(None, None, 1))
        else:
            r_lim_points = int(distance_from_center / self.delta_r)
            mesh_slicer = (slice(None, None, 1), slice(0, int(r_lim_points + 1), 1))

        return mesh_slicer

    @si.utils.memoize
    def get_mesh_slicer_spatial(self, distance_from_center = None):
        """Returns a slice object that slices a mesh to the given distance of the center."""
        if distance_from_center is None:
            mesh_slicer = slice(None, None, 1)
        else:
            r_lim_points = int(distance_from_center / self.delta_r)
            mesh_slicer = slice(0, int(r_lim_points + 1), 1)

        return mesh_slicer

    @property
    @si.utils.memoize
    def theta_plot(self):
        return np.linspace(0, twopi, self.theta_points)

    @property
    @si.utils.memoize
    def theta_calc(self):
        return np.linspace(0, pi, self.theta_points)

    @property
    @si.utils.memoize
    def theta_plot_mesh(self):
        return np.meshgrid(self.r, self.theta_plot, indexing = 'ij')[1]

    @property
    @si.utils.memoize
    def r_theta_mesh(self):
        return np.meshgrid(self.r, self.theta_plot, indexing = 'ij')[0]

    @property
    @si.utils.memoize
    def theta_calc_mesh(self):
        return np.meshgrid(self.r, self.theta_calc, indexing = 'ij')[1]

    @property
    @si.utils.memoize
    def r_theta_calc_mesh(self):
        return np.meshgrid(self.r, self.theta_calc, indexing = 'ij')[0]

    @property
    @si.utils.memoize
    def _sph_harm_l_theta_plot_mesh(self):
        l_mesh, theta_mesh = np.meshgrid(self.l, self.theta_plot, indexing = 'ij')
        return special.sph_harm(0, l_mesh, 0, theta_mesh)

    @property
    @si.utils.memoize
    def _sph_harm_l_theta_calc_mesh(self):
        l_mesh, theta_mesh = np.meshgrid(self.l, self.theta_calc, indexing = 'ij')
        return special.sph_harm(0, l_mesh, 0, theta_mesh)

    def reconstruct_spatial_mesh__plot(self, mesh):
        """Reconstruct the spatial (r, theta) representation of a mesh from the (l, r) representation."""
        # l: l, angular momentum index
        # r: r, radial position index
        # t: theta, polar angle index
        return np.einsum('lr,lt->rt', mesh, self._sph_harm_l_theta_plot_mesh)

    def reconstruct_spatial_mesh__calc(self, mesh):
        """Reconstruct the spatial (r, theta) representation of a mesh from the (l, r) representation."""
        # l: l, angular momentum index
        # r: r, radial position index
        # t: theta, polar angle index
        return np.einsum('lr,lt->rt', mesh, self._sph_harm_l_theta_calc_mesh)

    @property
    @si.utils.watcher(lambda s: s.sim.time)
    def space_g(self):
        return self.reconstruct_spatial_mesh__plot(self.g)

    @property
    @si.utils.watcher(lambda s: s.sim.time)
    def space_g_calc(self):
        return self.reconstruct_spatial_mesh__calc(self.g)

    # @property
    # def space_psi(self):
    #     return self.space_g / self.g_factor

    @property
    def space_psi(self):
        return self.space_g / self.r_theta_mesh

    @property
    def g2(self):
        return np.abs(self.space_g) ** 2

    @property
    def psi2(self):
        return np.abs(self.space_psi) ** 2

    def attach_mesh_to_axis(self, axis, mesh,
                            distance_unit = 'bohr_radius',
                            colormap = plt.get_cmap('inferno'),
                            norm = si.vis.AbsoluteRenormalize(),
                            shading = 'flat',
                            plot_limit = None,
                            slicer = 'get_mesh_slicer_spatial',
                            **kwargs):
        unit_value, _ = get_unit_value_and_latex_from_unit(distance_unit)

        _slice = getattr(self, slicer)(plot_limit)

        color_mesh = axis.pcolormesh(self.theta_plot_mesh[_slice],
                                     self.r_theta_mesh[_slice] / unit_value,
                                     mesh[_slice],
                                     shading = shading,
                                     cmap = colormap,
                                     norm = norm,
                                     **kwargs)

        return color_mesh

    # def attach_probability_current_to_axis(self, axis, plot_limit = None, distance_unit = 'bohr_radius'):
    #     raise NotImplementedError

    def plot_mesh(self, mesh,
                  name = '',
                  title = None,
                  distance_unit = 'bohr_radius',
                  colormap = COLORMAP_WAVEFUNCTION,
                  norm = si.vis.AbsoluteRenormalize(),
                  shading = 'flat',
                  plot_limit = None,
                  slicer = 'get_mesh_slicer_spatial',
                  aspect_ratio = 1,
                  show_colorbar = True,
                  show_title = True,
                  show_axes = True,
                  title_size = 20,
                  tick_label_size = 10,
                  grid_kwargs = None,
                  # overlay_probability_current = False, probability_current_time_step = 0,
                  **kwargs):
        if grid_kwargs is None:
            grid_kwargs = {}

        with si.vis.FigureManager(name = f'{self.spec.name}__{name}', aspect_ratio = aspect_ratio, **kwargs) as figman:
            fig = figman.fig

            fig.set_tight_layout(True)
            axis = plt.subplot(111, projection = 'polar')
            axis.set_theta_zero_location('N')
            axis.set_theta_direction('clockwise')

            unit_value, unit_latex = get_unit_value_and_latex_from_unit(distance_unit)

            color_mesh = self.attach_mesh_to_axis(
                axis, mesh,
                distance_unit = distance_unit,
                colormap = colormap,
                norm = norm,
                shading = shading,
                plot_limit = plot_limit,
                slicer = slicer
            )
            # if overlay_probability_current:
            #     quiv = self.attach_probability_current_to_axis(axis, plot_limit = plot_limit, distance_unit = distance_unit)

            if title is not None and title != '' and show_axes and show_title:
                title = axis.set_title(title, fontsize = title_size)
                title.set_x(.03)  # move title to the upper left corner
                title.set_y(.97)

            # make a colorbar
            if show_colorbar and show_axes:
                cbar_axis = fig.add_axes([1.01, .1, .04, .8])  # add a new axis for the cbar so that the old axis can stay square
                cbar = plt.colorbar(mappable = color_mesh, cax = cbar_axis)
                cbar.ax.tick_params(labelsize = tick_label_size)

            axis.grid(True, color = si.vis.CMAP_TO_OPPOSITE[colormap.name], **{**si.vis.COLORMESH_GRID_KWARGS, **grid_kwargs})  # change grid color to make it show up against the colormesh
            angle_labels = ['{}\u00b0'.format(s) for s in (0, 30, 60, 90, 120, 150, 180, 150, 120, 90, 60, 30)]  # \u00b0 is unicode degree symbol
            axis.set_thetagrids(np.arange(0, 359, 30), frac = 1.09, labels = angle_labels)

            axis.tick_params(axis = 'both', which = 'major', labelsize = tick_label_size)  # increase size of tick labels
            axis.tick_params(axis = 'y', which = 'major', colors = si.vis.CMAP_TO_OPPOSITE[colormap.name], pad = 3)  # make r ticks a color that shows up against the colormesh
            # axis.tick_params(axis = 'both', which = 'both', length = 0)

            axis.set_rlabel_position(80)

            max_yticks = 5
            yloc = plt.MaxNLocator(max_yticks, symmetric = False, prune = 'both')
            axis.yaxis.set_major_locator(yloc)

            fig.canvas.draw()  # must draw early to modify the axis text

            tick_labels = axis.get_yticklabels()
            for t in tick_labels:
                t.set_text(t.get_text() + fr'${unit_latex}$')
            axis.set_yticklabels(tick_labels)

            if plot_limit is not None and plot_limit < self.r_max:
                axis.set_rmax((plot_limit - (self.delta_r / 2)) / unit_value)
            else:
                axis.set_rmax((self.r_max - (self.delta_r / 2)) / unit_value)

            if not show_axes:
                axis.axis('off')

    def attach_g_to_axis(self, axis,
                         colormap = plt.get_cmap('richardson'),
                         norm = None,
                         **kwargs):
        if norm is None:
            norm = si.vis.RichardsonNormalization(np.max(np.abs(self.space_g) / DEFAULT_RICHARDSON_MAGNITUDE_DIVISOR))

        return self.attach_mesh_to_axis(axis, self.space_g,
                                        colormap = colormap,
                                        norm = norm,
                                        **kwargs)

    def plot_g(self, name_postfix = '',
               colormap = plt.get_cmap('richardson'),
               norm = None,
               **kwargs):
        title = r'$g$'
        name = 'g' + name_postfix

        if norm is None:
            norm = si.vis.RichardsonNormalization(np.max(np.abs(self.space_g) / DEFAULT_RICHARDSON_MAGNITUDE_DIVISOR))

        self.plot_mesh(self.space_g, name = name, title = title,
                       colormap = colormap,
                       norm = norm,
                       show_colorbar = False,
                       **kwargs)

    def attach_psi_to_axis(self, axis,
                           colormap = plt.get_cmap('richardson'),
                           norm = None,
                           **kwargs):
        if norm is None:
            norm = si.vis.RichardsonNormalization(np.max(np.abs(self.space_psi) / DEFAULT_RICHARDSON_MAGNITUDE_DIVISOR))

        return self.attach_mesh_to_axis(axis, self.space_psi,
                                        colormap = colormap,
                                        norm = norm,
                                        **kwargs)

    def plot_psi(self, name_postfix = '',
                 colormap = plt.get_cmap('richardson'),
                 norm = None,
                 **kwargs):
        title = r'$\Psi$'
        name = 'psi' + name_postfix

        if norm is None:
            norm = si.vis.RichardsonNormalization(np.max(np.abs(self.space_psi) / DEFAULT_RICHARDSON_MAGNITUDE_DIVISOR))

        self.plot_mesh(self.space_psi, name = name, title = title,
                       colormap = colormap,
                       norm = norm,
                       show_colorbar = False,
                       **kwargs)

    def update_g_mesh(self, colormesh, **kwargs):
        self.update_mesh(colormesh, self.space_g, **kwargs)

    def update_psi_mesh(self, colormesh, **kwargs):
        self.update_mesh(colormesh, self.space_psi, **kwargs)

    def attach_mesh_repr_to_axis(self, axis, mesh,
                                 distance_unit = 'bohr_radius',
                                 colormap = plt.get_cmap('inferno'),
                                 norm = si.vis.AbsoluteRenormalize(),
                                 shading = 'flat',
                                 plot_limit = None,
                                 slicer = 'get_mesh_slicer',
                                 **kwargs):
        unit_value, _ = get_unit_value_and_latex_from_unit(distance_unit)

        _slice = getattr(self, slicer)(plot_limit)

        color_mesh = axis.pcolormesh(self.l_mesh[_slice],
                                     self.r_mesh[_slice] / unit_value,
                                     self.g[_slice],
                                     shading = shading,
                                     cmap = colormap,
                                     norm = norm,
                                     **kwargs)

        return color_mesh

    def plot_mesh_repr(self, mesh,
                       name = '',
                       title = None,
                       distance_unit = 'bohr_radius',
                       colormap = COLORMAP_WAVEFUNCTION,
                       norm = si.vis.AbsoluteRenormalize(),
                       shading = 'flat',
                       plot_limit = None,
                       slicer = 'get_mesh_slicer',
                       aspect_ratio = si.vis.GOLDEN_RATIO,
                       show_colorbar = True,
                       show_title = True,
                       show_axes = True,
                       title_y_adjust = 1.1,
                       title_size = 12,
                       axis_label_size = 12,
                       tick_label_size = 10,
                       grid_kwargs = None,
                       **kwargs):
        if grid_kwargs is None:
            grid_kwargs = {}
        with si.vis.FigureManager(name = f'{self.spec.name}__{name}', aspect_ratio = aspect_ratio, **kwargs) as figman:
            fig = figman.fig

            fig.set_tight_layout(True)
            axis = plt.subplot(111)

            unit_value, unit_latex = get_unit_value_and_latex_from_unit(distance_unit)

            color_mesh = self.attach_mesh_repr_to_axis(
                axis, mesh,
                distance_unit = distance_unit,
                colormap = colormap,
                norm = norm,
                shading = shading,
                plot_limit = plot_limit,
                slicer = slicer
            )

            axis.set_xlabel(r'$\ell$', fontsize = axis_label_size)
            axis.set_ylabel(fr'$r$ (${unit_latex}$)', fontsize = axis_label_size)
            if title is not None and title != '' and show_axes and show_title:
                title = axis.set_title(title, fontsize = title_size)
                title.set_y(title_y_adjust)  # move title up a bit

            # make a colorbar
            if show_colorbar and show_axes:
                cbar = fig.colorbar(mappable = color_mesh, ax = axis)
                cbar.ax.tick_params(labelsize = tick_label_size)

            axis.grid(True, color = si.vis.CMAP_TO_OPPOSITE[colormap.name], **{**si.vis.COLORMESH_GRID_KWARGS, **grid_kwargs})  # change grid color to make it show up against the colormesh

            axis.tick_params(labelright = True, labeltop = True)  # ticks on all sides
            axis.tick_params(axis = 'both', which = 'major', labelsize = tick_label_size)  # increase size of tick labels
            # axis.tick_params(axis = 'both', which = 'both', length = 0)

            y_ticks = axis.yaxis.get_major_ticks()
            y_ticks[0].label1.set_visible(False)
            y_ticks[0].label2.set_visible(False)
            y_ticks[-1].label1.set_visible(False)
            y_ticks[-1].label2.set_visible(False)

            axis.axis('tight')

            if not show_axes:
                axis.axis('off')

    def plot_g_repr(self, name_postfix = '',
                    colormap = plt.get_cmap('richardson'),
                    norm = None,
                    title = None,
                    **kwargs):
        if title is None:
            title = r'$g$'
        name = 'g_repr' + name_postfix

        if norm is None:
            norm = si.vis.RichardsonNormalization(np.max(np.abs(self.g) / DEFAULT_RICHARDSON_MAGNITUDE_DIVISOR))

        self.plot_mesh_repr(self.g, name = name, title = title,
                            colormap = colormap,
                            norm = norm,
                            show_colorbar = False,
                            **kwargs)

    def attach_g_repr_to_axis(self,
                              axis,
                              colormap = plt.get_cmap('richardson'),
                              norm = None,
                              **kwargs
                              ):

        if norm is None:
            norm = si.vis.RichardsonNormalization(np.max(np.abs(self.g) / DEFAULT_RICHARDSON_MAGNITUDE_DIVISOR))

        return self.attach_mesh_repr_to_axis(axis, self.g,
                                             colormap = colormap,
                                             norm = norm,
                                             **kwargs)

    def plot_electron_momentum_spectrum(self, r_type = 'wavenumber', r_scale = 'per_nm',
                                        r_lower_lim = twopi * .01 * per_nm, r_upper_lim = twopi * 10 * per_nm, r_points = 100,
                                        theta_points = 360,
                                        g = None,
                                        **kwargs):
        """

        :param r_type:
        :param r_scale:
        :param r_lower_lim:
        :param r_upper_lim:
        :param r_points:
        :param theta_points:
        :param g:
        :param kwargs:
        :return:
        """
        if r_type not in ('wavenumber', 'energy', 'momentum'):
            raise ValueError("Invalid argument to plot_electron_spectrum: r_type must be either 'wavenumber', 'energy', or 'momentum'")

        thetas = np.linspace(0, twopi, theta_points)
        r = np.linspace(r_lower_lim, r_upper_lim, r_points)

        if r_type == 'wavenumber':
            wavenumbers = r
        elif r_type == 'energy':
            wavenumbers = electron_wavenumber_from_energy(r)
        elif r_type == 'momentum':
            wavenumbers = r / hbar

        if g is None:
            g = self.g

        theta_mesh, wavenumber_mesh, inner_product_mesh = self.inner_product_with_plane_waves(thetas, wavenumbers, g = g)

        if r_type == 'wavenumber':
            r_mesh = wavenumber_mesh
        elif r_type == 'energy':
            r_mesh = electron_energy_from_wavenumber(wavenumber_mesh)
        elif r_type == 'momentum':
            r_mesh = wavenumber_mesh * hbar

        return self.plot_electron_momentum_spectrum_from_meshes(theta_mesh, r_mesh, inner_product_mesh,
                                                                r_type, r_scale,
                                                                **kwargs)

    def plot_electron_momentum_spectrum_from_meshes(self, theta_mesh, r_mesh, inner_product_mesh,
                                                    r_type, r_scale,
                                                    log = False,
                                                    **kwargs):
        """
        Generate a polar plot of the wavefunction decomposed into plane waves.

        The radial dimension can be displayed in wavenumbers, energy, or momentum. The angle is the angle of the plane wave in the z-x plane (because m=0, the decomposition is symmetric in the x-y plane).

        :param r_type: type of unit for the radial axis ('wavenumber', 'energy', or 'momentum')
        :param r_scale: unit specification for the radial dimension
        :param r_lower_lim: lower limit for the radial dimension
        :param r_upper_lim: upper limit for the radial dimension
        :param r_points: number of points for the radial dimension
        :param theta_points: number of points for the angular dimension
        :param log: True to displayed logged data, False otherwise (default: False)
        :param kwargs: kwargs are passed to compy.utils.FigureManager
        :return: the FigureManager generated during plot creation
        """
        if r_type not in ('wavenumber', 'energy', 'momentum'):
            raise ValueError("Invalid argument to plot_electron_spectrum: r_type must be either 'wavenumber', 'energy', or 'momentum'")

        r_unit_value, r_unit_name = get_unit_value_and_latex_from_unit(r_scale)

        plot_kwargs = {**dict(aspect_ratio = 1), **kwargs}

        r_mesh = np.real(r_mesh)
        overlap_mesh = np.abs(inner_product_mesh) ** 2

        with si.vis.FigureManager(self.sim.name + '__electron_spectrum', **plot_kwargs) as figman:
            fig = figman.fig

            fig.set_tight_layout(True)

            axis = plt.subplot(111, projection = 'polar')
            axis.set_theta_zero_location('N')
            axis.set_theta_direction('clockwise')

            figman.name += '__{}'.format(r_type)

            norm = None
            if log:
                norm = matplotlib.colors.LogNorm(vmin = np.nanmin(overlap_mesh), vmax = np.nanmax(overlap_mesh))
                figman.name += '__log'

            color_mesh = axis.pcolormesh(theta_mesh,
                                         r_mesh / r_unit_value,
                                         overlap_mesh,
                                         shading = 'flat',
                                         norm = norm,
                                         cmap = 'viridis')

            # make a colorbar
            cbar_axis = fig.add_axes([1.01, .1, .04, .8])  # add a new axis for the cbar so that the old axis can stay square
            cbar = plt.colorbar(mappable = color_mesh, cax = cbar_axis)
            cbar.ax.tick_params(labelsize = 10)

            axis.grid(True, color = si.vis.COLOR_OPPOSITE_VIRIDIS, **si.vis.COLORMESH_GRID_KWARGS)  # change grid color to make it show up against the colormesh
            angle_labels = ['{}\u00b0'.format(s) for s in (0, 30, 60, 90, 120, 150, 180, 150, 120, 90, 60, 30)]  # \u00b0 is unicode degree symbol
            axis.set_thetagrids(np.arange(0, 359, 30), frac = 1.075, labels = angle_labels)

            axis.tick_params(axis = 'both', which = 'major', labelsize = 8)  # increase size of tick labels
            axis.tick_params(axis = 'y', which = 'major', colors = si.vis.COLOR_OPPOSITE_VIRIDIS, pad = 3)  # make r ticks a color that shows up against the colormesh
            axis.tick_params(axis = 'both', which = 'both', length = 0)

            axis.set_rlabel_position(80)

            max_yticks = 5
            yloc = plt.MaxNLocator(max_yticks, symmetric = False, prune = 'both')
            axis.yaxis.set_major_locator(yloc)

            fig.canvas.draw()  # must draw early to modify the axis text

            tick_labels = axis.get_yticklabels()
            for t in tick_labels:
                t.set_text(t.get_text() + r'${}$'.format(r_unit_name))
            axis.set_yticklabels(tick_labels)

            axis.set_rmax(np.nanmax(r_mesh) / r_unit_value)

        return figman


class SphericalHarmonicSnapshot(Snapshot):
    def __init__(self, simulation, time_index,
                 plane_wave_overlap__max_wavenumber = 50 * per_nm, plane_wave_overlap__wavenumber_points = 500, plane_wave_overlap__theta_points = 200, ):
        super().__init__(simulation, time_index)

        self.plane_wave_overlap__max_wavenumber = plane_wave_overlap__max_wavenumber
        self.plane_wave_overlap__wavenumber_points = plane_wave_overlap__wavenumber_points
        self.plane_wave_overlap__theta_points = plane_wave_overlap__theta_points

    def take_snapshot(self):
        super().take_snapshot()

        for free_only in (True, False):
            self.collect_inner_product_with_plane_waves(free_only = free_only)

    def collect_inner_product_with_plane_waves(self, free_only = False):
        thetas = np.linspace(0, twopi, self.plane_wave_overlap__theta_points)
        wavenumbers = np.delete(np.linspace(0, self.plane_wave_overlap__max_wavenumber, self.plane_wave_overlap__wavenumber_points + 1), 0)

        if free_only:
            key = 'inner_product_with_plane_waves__free_only'
            g = self.sim.mesh.get_g_with_states_removed(self.sim.bound_states)
        else:
            key = 'inner_product_with_plane_waves'
            g = None

        self.data[key] = self.sim.mesh.inner_product_with_plane_waves(thetas, wavenumbers, g = g)
