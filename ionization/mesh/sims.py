import logging
from typing import Optional, Iterable, Callable, Collection

import collections
import itertools
import datetime
import sys
from pathlib import Path
from copy import deepcopy
import abc

import numpy as np
import numpy.fft as nfft
from tqdm import tqdm

import simulacra as si
import simulacra.units as u

from .. import potentials, states, core, utils, exceptions
from . import (
    meshes,
    anim,
    snapshots,
    data,
    evolution_methods,
    mesh_operators,
    sim_plotters,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MeshSimulation(si.Simulation):
    """
    A class for a TDSE simulation that uses a mesh (i.e., a :class:`QuantumMesh`).

    Attributes
    ----------
    data
        A :class:`Data` that provides access to time-indexed data.
    """

    def __init__(self, spec: "MeshSpecification"):
        super().__init__(spec)

        self.latest_checkpoint_time = datetime.datetime.utcnow()

        self.times = self.get_times()

        if self.spec.electric_potential_dc_correction:
            old_pot = self.spec.electric_potential
            self.spec.electric_potential = potentials.DC_correct_electric_potential(
                self.spec.electric_potential, self.times
            )

            logger.warning(
                f"DC-corrected electric potential {old_pot} --> {self.spec.electric_potential} for {self}"
            )

        if self.spec.electric_potential_fluence_correction:
            old_pot = self.spec.electric_potential
            self.spec.electric_potential = potentials.FluenceCorrector(
                electric_potential=self.spec.electric_potential,
                times=self.times,
                target_fluence=list(self.spec.electric_potential)[
                    0
                ].fluence,  # the analytic fluence of the embedded pulse, whether it's been dc-corrected or not
            )

            logger.warning(
                f"Fluence-corrected electric potential {old_pot} --> {self.spec.electric_potential} for {self}"
            )

        self.time_index = 0
        self.data_time_index = 0
        self.time_steps = len(self.times)

        self.mesh = self.spec.mesh_type(self)

        # simulation data storage
        time_indices = np.array(range(0, self.time_steps))
        self.data_mask = np.equal(time_indices, 0) + np.equal(
            time_indices, self.time_steps - 1
        )
        if self.spec.store_data_every >= 1:
            self.data_mask += np.equal(time_indices % self.spec.store_data_every, 0)
        self.data_times = self.times[self.data_mask]
        self.data_indices = time_indices[self.data_mask]
        self.data_time_steps = len(self.data_times)
        self.spacetime_points = (
            self.mesh.g.shape[0] * self.mesh.g.shape[1] * self.time_steps
        )

        self.data = data.Data(self)
        self.datastores_by_type = {
            ds.__class__: deepcopy(ds) for ds in self.spec.datastores
        }
        for ds in self.datastores_by_type.values():
            ds.init(self)

        # populate the snapshot times from the two ways of entering snapshot times in the spec (by index or by time)
        self.snapshot_times = set()

        for time in self.spec.snapshot_times:
            time_index, time_target, _ = si.utils.find_nearest_entry(self.times, time)
            self.snapshot_times.add(time_target)

        for index in self.spec.snapshot_indices:
            self.snapshot_times.add(self.times[index])

        self.snapshots = dict()

        self.warnings = collections.defaultdict(list)

        self.plot = self.spec.simulation_plotter_type(self)

    def get_blank_data(self, dtype=np.float64) -> np.array:
        """
        Return an array of NaNs appropriate for storing time-indexed data in a Datastore.

        Parameters
        ----------
        dtype
            A numpy ``dtype``, which will be the ``dtype`` of the resulting blank data array.

        Returns
        -------
        :class:`np.ndarray`
            A blank array for time-indexed data.
        """
        # this is the best method according to
        # https://stackoverflow.com/questions/1704823/initializing-numpy-matrix-to-something-other-than-zero-or-one
        a = np.empty(self.data_time_steps, dtype=dtype)
        a.fill(np.NaN)

        return a

    def info(self) -> si.Info:
        info = super().info()

        mem_mesh = self.mesh.g.nbytes if self.mesh is not None else 0

        mem_matrix_operators = 6 * mem_mesh
        mem_numeric_eigenstates = sum(
            state.g.nbytes
            for state in self.spec.test_states
            if state.numeric and state.g is not None
        )

        mem_misc = sum(
            x.nbytes
            for x in (self.times, self.data_times, self.data_mask, self.data_indices)
        )

        mem_total = sum(
            (mem_mesh, mem_matrix_operators, mem_numeric_eigenstates, mem_misc)
        )

        info_mem = si.Info(
            header=f"Memory Usage (approx.): {si.utils.bytes_to_str(mem_total)}"
        )
        info_mem.add_field("g", si.utils.bytes_to_str(mem_mesh))
        info_mem.add_field(
            "Matrix Operators", si.utils.bytes_to_str(mem_matrix_operators)
        )
        if (
            hasattr(self.spec, "use_numeric_eigenstates")
            and self.spec.use_numeric_eigenstates
        ):
            info_mem.add_field(
                "Numeric Eigenstates", si.utils.bytes_to_str(mem_numeric_eigenstates)
            )
        info_mem.add_fields(
            (ds_type.__name__, si.utils.bytes_to_str(sys.getsizeof(ds)))
            for ds_type, ds in self.datastores_by_type.items()
        )
        info_mem.add_field("Miscellaneous", si.utils.bytes_to_str(mem_misc))

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
        return self.times[: self.time_index + 1]

    def get_times(self) -> np.array:
        if not callable(self.spec.time_step):
            total_time = self.spec.time_final - self.spec.time_initial
            times = np.linspace(
                self.spec.time_initial,
                self.spec.time_final,
                int(total_time / self.spec.time_step) + 1,
            )
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

    def store_data(self):
        """Update the time-indexed data arrays with the current values."""
        for ds_type, ds in self.datastores_by_type.items():
            ds.store()
            logger.debug(f"{self} stored data for {ds_type.__name__}")

    def check(self):
        norm = self.data.norm[self.data_time_index]
        if norm > 1.001 * self.data.norm[0]:
            logger.warning(
                f"Wavefunction norm ({norm}) has exceeded initial norm ({self.data.norm[0]}) by more than .1% for {self}"
            )
        try:
            if norm > 1.001 * self.data.norm[self.data_time_index - 1]:
                logger.warning(
                    f"Wavefunction norm ({norm}) at time_index = {self.data_time_index} has exceeded norm from previous time step ({self.data.norm[self.data_time_index - 1]}) by more than .1% for {self}"
                )
        except IndexError:
            pass

    def take_snapshot(self):
        snapshot = self.spec.snapshot_type(
            self, self.time_index, **self.spec.snapshot_kwargs
        )

        snapshot.take_snapshot()

        self.snapshots[self.time_index] = snapshot

        logger.info(
            f"Stored {snapshot.__class__.__name__} for {self} at time index {self.time_index} (t = {self.time / u.asec:.3f} as)"
        )

    def run(
        self,
        progress_bar: bool = False,
        callback: Callable[["MeshSimulation"], None] = None,
        checkpoint_callback: Callable[[Path], None] = None,
    ):
        """
        Run the simulation by repeatedly evolving the mesh by the time step.
        During this process, :class:`Datastore` will be used to collect data about the wavefunction.

        Parameters
        ----------
        progress_bar
            If ``True``, a progress bar will be displayed.
        callback
            If given, will be called with the :class:`MeshSimulation` as its argument after every time step.
        """
        if callback is None:
            callback = lambda s: None
        if checkpoint_callback is None:
            checkpoint_callback = lambda p: None

        logger.info(
            f"Performing time evolution on {self}, starting from time index {self.time_index}"
        )
        try:
            for animator in self.spec.animators:
                animator.initialize(self)

            if progress_bar:
                pbar = tqdm(
                    total=self.time_steps - self.time_index - 1, ascii=True, ncols=80
                )

            while True:
                is_data_time = self.time in self.data_times

                if is_data_time:
                    self.store_data()
                    self.check()

                if self.time in self.snapshot_times:
                    self.take_snapshot()

                for animator in self.spec.animators:
                    if (
                        self.time_index == 0
                        or self.time_index == self.time_steps
                        or self.time_index % animator.decimation == 0
                    ):
                        animator.send_frame_to_ffmpeg()

                callback(self)

                if (
                    is_data_time
                ):  # having to repeat this is clunky, but I need the data for the animators to work and I can't change the data index until the animators are done
                    self.data_time_index += 1

                if self.time_index == self.time_steps - 1:
                    break

                self.time_index += 1

                self.mesh.evolve(
                    self.times[self.time_index] - self.times[self.time_index - 1]
                )  # evolve the mesh forward to the next time step

                logger.debug(
                    f"{self} evolved to time index {self.time_index} / {self.time_steps - 1} ({self.percent_completed:.2f}%)"
                )

                if self.spec.checkpoints:
                    now = datetime.datetime.utcnow()
                    if (now - self.latest_checkpoint_time) > self.spec.checkpoint_every:
                        self.do_checkpoint(now, checkpoint_callback)

                if progress_bar:
                    pbar.update(1)

            if progress_bar:
                pbar.close()

            logger.info(f"Finished performing time evolution on {self}")
        except Exception as e:
            raise e
        finally:
            # make sure the animators get cleaned up if there's some kind of error during time evolution
            for animator in self.spec.animators:
                animator.cleanup()

            self.spec.animators = ()

    def do_checkpoint(self, now, callback: Callable[[Path], None]):
        self.status = si.Status.PAUSED
        path = self.save(target_dir=self.spec.checkpoint_dir, save_mesh=True)
        callback(path)
        self.latest_checkpoint_time = now
        logger.info(
            f"{self} checkpointed at time index {self.time_index} / {self.time_steps - 1} ({self.percent_completed:.2f}%)"
        )
        self.status = si.Status.RUNNING

    @property
    def percent_completed(self):
        return round(100 * self.time_index / (self.time_steps - 1), 2)

    @property
    def bound_states(self) -> Iterable[states.QuantumState]:
        yield from (s for s in self.spec.test_states if s.bound)

    @property
    def free_states(self) -> Iterable[states.QuantumState]:
        yield from (s for s in self.spec.test_states if not s.bound)

    def dipole_moment_vs_frequency(
        self, first_time: Optional[float] = None, last_time: Optional[float] = None
    ):
        logger.critical(
            "ALERT: dipole_momentum_vs_frequency does not account for non-uniform time step!"
        )

        if first_time is None:
            first_time_index, first_time = 0, self.times[0]
        else:
            first_time_index, first_time, _ = si.utils.find_nearest_entry(
                self.times, first_time
            )
        if last_time is None:
            last_time_index, last_time = (
                self.time_steps - 1,
                self.times[self.time_steps - 1],
            )
        else:
            last_time_index, last_time, _ = si.utils.find_nearest_entry(
                self.times, last_time
            )
        points = last_time_index - first_time_index
        frequency = nfft.fftshift(nfft.fftfreq(points, self.spec.time_step))
        dipole_moment = nfft.fftshift(
            nfft.fft(
                self.data.electric_dipole_moment_expectation_value[
                    first_time_index:last_time_index
                ],
                norm="ortho",
            )
        )

        return frequency, dipole_moment

    def save(self, target_dir: Optional[str] = None, save_mesh: bool = True):
        """
        Atomically pickle the Simulation to ``{target_dir}/{sim.name}.sim``.

        Parameters
        ----------
        target_dir
            Directory to save the Simulation to
        save_mesh
            If ``True`` (default), save the mesh. If ``False``, don't.

        Returns
        -------

        """
        if len(self.spec.animators) > 0:
            raise exceptions.IonizationException(
                "Cannot pickle simulation containing animators"
            )

        if not save_mesh:
            # remove numeric eigenstate information
            for state in self.spec.test_states:
                state.g = None

            mesh = self.mesh
            self.mesh = None

        out = super().save(target_dir=target_dir)

        if not save_mesh:
            self.mesh = mesh

        return out


class MeshSpecification(si.Specification, abc.ABC):
    """An abstract :class:`simulacra.Specification` for a :class:`simulacra.Simulation` with a :class:`QuantumMesh`."""

    simulation_type = MeshSimulation
    mesh_type = meshes.QuantumMesh
    simulation_plotter_type = sim_plotters.MeshSimulationPlotter

    def __init__(
        self,
        name: str,
        test_mass: float = u.electron_mass_reduced,
        test_charge: float = u.electron_charge,
        initial_state: states.QuantumState = states.HydrogenBoundState(1, 0),
        test_states: Iterable[states.QuantumState] = tuple(),
        internal_potential: potentials.PotentialEnergy = potentials.CoulombPotential(
            charge=u.proton_charge
        ),
        electric_potential: potentials.ElectricPotential = potentials.NoElectricPotential(),
        electric_potential_dc_correction: bool = False,
        electric_potential_fluence_correction: bool = False,
        mask: potentials.Mask = potentials.NoMask(),
        operators: mesh_operators.MeshOperators = None,
        evolution_method: evolution_methods.EvolutionMethod = None,
        time_initial: float = 0 * u.asec,
        time_final: float = 200 * u.asec,
        time_step: float = 1 * u.asec,
        checkpoints: bool = False,
        checkpoint_every: datetime.timedelta = datetime.timedelta(hours=1),
        checkpoint_dir: Optional[str] = None,
        animators: Iterable[anim.WavefunctionSimulationAnimator] = tuple(),
        store_data_every: int = 1,
        snapshot_times=(),
        snapshot_indices=(),
        snapshot_type=None,
        snapshot_kwargs: Optional[dict] = None,
        datastores: Optional[Collection[data.Datastore]] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        name
            The name of the specification/simulation.
        test_mass
            The mass of the test particle.
        test_charge
            The charge of the test particle.
        initial_state
            The initial :class:`QuantumState` of the test particle.
        test_states
            The :class:`QuantumState` that will be tracked during the simulation.
        internal_potential
            The static, internal potentials.
        electric_potential
            The possibly-time varying external electric field.
        electric_potential_dc_correction
            If ``True``, perform DC correction on the ``electric_potential``.
        mask
            A :class:`ionizatpotentials.Mask` to apply to the wavefunction after every time step.
        operators
            A :class:`MeshOperators` to generate discretized mesh operators from.
        evolution_method
            A :class:`EvolutionMethod` to provide a time evolution algorithm.
        time_initial
            The time to begin the simulation at.
        time_final
            The time to end the simulation at.
        time_step
            The amount of time to evolve by on each evolution step.
        checkpoints
            If ``True``, the simulation will save checkpoints to ``checkpoint_dir`` every ``checkpoint_every``.
        checkpoint_every
            The time between checkpoints.
        checkpoint_dir
            The directory to save checkpoints to.
        animators
            Any :class:`WavefunctionSimulationAnimator` to run during the simulation.
        store_data_every
            Data will be stored every ``store_data_every`` time steps.
            The special value ``store_data_every = -1`` causes data to be stored only on the first and last time steps.
        snapshot_times
        snapshot_indices
        snapshot_type
        snapshot_kwargs
        datastores
            A list of prototype datastores to use during the simulation.
            During :class:`MeshSimulation` initialization they are cloned and the clones are then attached to the simulation.
        kwargs
            Any additional keyword arguments are passed to the :class:`simulacra.Specification` constructor.
        """
        super().__init__(name, **kwargs)

        self.test_mass = test_mass
        self.test_charge = test_charge
        self.initial_state = initial_state
        self.test_states = sorted(test_states)
        if len(self.test_states) == 0:
            self.test_states = [self.initial_state]

        self.internal_potential = internal_potential
        self.electric_potential = electric_potential
        self.electric_potential_dc_correction = electric_potential_dc_correction
        self.electric_potential_fluence_correction = (
            electric_potential_fluence_correction
        )
        self.mask = mask

        self.operators = operators
        self.evolution_method = evolution_method

        self.time_initial = time_initial
        self.time_final = time_final
        self.time_step = time_step

        self.checkpoints = checkpoints
        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir = checkpoint_dir

        self.animators = deepcopy(tuple(animators))

        self.store_data_every = int(store_data_every)

        self.snapshot_times = set(snapshot_times)
        self.snapshot_indices = set(snapshot_indices)
        if snapshot_type is None:
            snapshot_type = snapshots.Snapshot
        self.snapshot_type = snapshot_type
        if snapshot_kwargs is None:
            snapshot_kwargs = dict()
        self.snapshot_kwargs = snapshot_kwargs

        if datastores is None:
            datastores = [ds_type() for ds_type in data.DEFAULT_DATASTORE_TYPES]
        self.datastores = datastores
        self.datastore_types = tuple(
            sorted(
                set(ds.__class__ for ds in self.datastores),
                key=lambda ds: ds.__class__.__name__,
            )
        )

        if len(self.datastores) != len(self.datastore_types):
            raise exceptions.DuplicateDatastores("Cannot duplicate datastores")

    def info(self) -> si.Info:
        info = super().info()

        info_checkpoint = si.Info(header="Checkpointing")
        if self.checkpoints:
            if self.checkpoint_dir is not None:
                working_in = self.checkpoint_dir
            else:
                working_in = "cwd"
            info_checkpoint.header += (
                f": every {self.checkpoint_every} time steps, working in {working_in}"
            )
        else:
            info_checkpoint.header += ": disabled"

        info.add_info(info_checkpoint)

        info_animation = si.Info(header="Animation")
        if len(self.animators) > 0:
            for animator in self.animators:
                info_animation.add_info(animator.info())
        else:
            info_animation.header += ": none"
        info.add_info(info_animation)

        info_evolution = si.Info(header="Time Evolution")
        info_evolution.add_field("Initial State", str(self.initial_state))
        info_evolution.add_field(
            "Initial Time", utils.fmt_quantity(self.time_initial, utils.TIME_UNITS)
        )
        info_evolution.add_field(
            "Final Time", utils.fmt_quantity(self.time_final, utils.TIME_UNITS)
        )
        if not callable(self.time_step):
            info_evolution.add_field(
                "Time Step", utils.fmt_quantity(self.time_step, utils.TIME_UNITS)
            )
        else:
            info_evolution.add_field("Time Step", f"determined by {self.time_step}")
        info.add_info(info_evolution)

        info_algorithm = si.Info(header="Operators and Evolution Algorithm")
        info_algorithm.add_info(self.operators.info())
        info_algorithm.add_info(self.evolution_method.info())
        info.add_info(info_algorithm)

        info_potentials = si.Info(header="Potentials and Masks")
        info_potentials.add_field(
            "DC Correct Electric Field",
            "yes" if self.electric_potential_dc_correction else "no",
        )
        for x in itertools.chain(
            self.internal_potential, self.electric_potential, self.mask
        ):
            info_potentials.add_info(x.info())
        info.add_info(info_potentials)

        info_analysis = si.Info(header="Analysis")

        info_test_particle = si.Info(header="Test Particle")
        info_test_particle.add_field(
            "Charge", utils.fmt_quantity(self.test_charge, utils.CHARGE_UNITS)
        )
        info_test_particle.add_field(
            "Mass", utils.fmt_quantity(self.test_mass, utils.MASS_UNITS)
        )
        info_analysis.add_info(info_test_particle)

        if len(self.test_states) > 10:
            info_analysis.add_field(
                f"Test States (first 5 of {len(self.test_states)})",
                ", ".join(str(s) for s in sorted(self.test_states)[:5]),
            )
        else:
            info_analysis.add_field(
                "Test States", ", ".join(str(s) for s in sorted(self.test_states))
            )
        info_analysis.add_field(
            "Datastore Types", ", ".join(ds.__name__ for ds in self.datastore_types)
        )
        info_analysis.add_field("Data Storage Decimation", self.store_data_every)
        info_analysis.add_field(
            "Snapshot Indices",
            ", ".join(sorted(self.snapshot_indices))
            if len(self.snapshot_indices) > 0
            else "none",
        )
        info_analysis.add_field(
            "Snapshot Times",
            (f"{st / u.asec:.3f} as" for st in self.snapshot_times)
            if len(self.snapshot_times) > 0
            else "none",
        )
        info.add_info(info_analysis)

        return info


class LineSpecification(MeshSpecification):
    """A concrete :class:`MeshSpecification` for a :class:`MeshSimulation` with a :class:`LineMesh`."""

    mesh_type = meshes.LineMesh

    def __init__(
        self,
        name: str,
        internal_potential: potentials.PotentialEnergy = potentials.HarmonicOscillator(
            1 * u.N / u.m
        ),
        initial_state: states.QuantumState = states.QHOState(1 * u.N / u.m),
        z_bound: float = 10 * u.nm,
        z_points: int = 2 ** 9,
        use_numeric_eigenstates: bool = False,
        number_of_numeric_eigenstates: int = 100,
        analytic_eigenstate_type: Optional[states.QuantumState] = None,
        operators: mesh_operators.MeshOperators = mesh_operators.LineLengthGaugeOperators(),
        evolution_method: evolution_methods.EvolutionMethod = evolution_methods.AlternatingDirectionImplicit(),
        **kwargs,
    ):
        """
        Parameters
        ----------
        z_bound
            The symmetric bounds of the simulation (i.e., the simulation region goes from ``-z`` to ``+z``.
        z_points
            The number of coordinate points for the z-dimension.
        use_numeric_eigenstates
            If ``True``, the ``test_states`` will be replaced by numeric eigenstates generated from the field-free evolution operators.
        number_of_numeric_eigenstates
            The number of numeric eigenstates to generate.
        analytic_eigenstate_type
            The type of analytic eigenstate to use.
            This can be ``None`` only if ``use_numeric_eigenstates`` is ``False``.
        kwargs
            Any additional keyword arguments are passed to the :class:`MeshSpecification` constructor.
        """
        super().__init__(
            name,
            internal_potential=internal_potential,
            initial_state=initial_state,
            operators=operators,
            evolution_method=evolution_method,
            **kwargs,
        )

        self.z_bound = z_bound
        self.z_points = int(z_points)

        self.analytic_eigenstate_type = analytic_eigenstate_type
        self.use_numeric_eigenstates = use_numeric_eigenstates
        self.number_of_numeric_eigenstates = number_of_numeric_eigenstates

    def info(self) -> si.Info:
        info = super().info()

        info_mesh = si.Info(header=f"Mesh: {self.mesh_type.__name__}")
        info_mesh.add_field(
            "Z Boundary", utils.fmt_quantity(self.z_bound, utils.LENGTH_UNITS)
        )
        info_mesh.add_field("Z Points", self.z_points)
        info_mesh.add_field(
            "Z Mesh Spacing",
            utils.fmt_quantity(2 * self.z_bound / self.z_points, utils.LENGTH_UNITS),
        )

        info.add_info(info_mesh)

        info_eigenstates = si.Info(
            header=f"Numeric Eigenstates: {self.use_numeric_eigenstates}"
        )
        if self.use_numeric_eigenstates:
            info_eigenstates.add_field(
                "Number of Numeric Eigenstates", self.number_of_numeric_eigenstates
            )

        info.add_info(info_eigenstates)

        return info


class RectangleSpecification(MeshSpecification):
    """A concrete :class:`MeshSpecification` for a :class:`MeshSimulation` with a :class:`RectangleMesh`."""

    mesh_type = meshes.RectangleMesh

    def __init__(
        self,
        name: str,
        internal_potential=potentials.NoPotentialEnergy(),
        initial_state=states.TwoDGaussianWavepacket(),
        z_bound: float = 10 * u.nm,
        z_points: int = 2 ** 9,
        x_bound: float = 10 * u.nm,
        x_points: int = 2 ** 9,
        operators: mesh_operators.MeshOperators = mesh_operators.RectangleLengthGaugeOperators(),
        evolution_method: evolution_methods.EvolutionMethod = evolution_methods.AlternatingDirectionImplicit(),
        **kwargs,
    ):
        """

        Parameters
        ----------
        z_bound
            The symmetric bounds of the simulation (i.e., the simulation region goes from ``-z`` to ``+z``.
        z_points
            The number of coordinate points for the z-dimension.
        use_numeric_eigenstates
            If ``True``, the ``test_states`` will be replaced by numeric eigenstates generated from the field-free evolution operators.
        number_of_numeric_eigenstates
            The number of numeric eigenstates to generate.
        analytic_eigenstate_type
            The type of analytic eigenstate to use.
            This can be ``None`` only if ``use_numeric_eigenstates`` is ``False``.
        kwargs
            Any additional keyword arguments are passed to the :class:`MeshSpecification` constructor.
        """
        super().__init__(
            name,
            internal_potential=internal_potential,
            initial_state=initial_state,
            operators=operators,
            evolution_method=evolution_method,
            **kwargs,
        )

        self.z_bound = z_bound
        self.z_points = int(z_points)
        self.x_bound = x_bound
        self.x_points = int(x_points)

    def info(self) -> si.Info:
        info = super().info()

        info_mesh = si.Info(header=f"Mesh: {self.mesh_type.__name__}")
        info_mesh.add_field(
            "Z Boundary", utils.fmt_quantity(self.z_bound, utils.LENGTH_UNITS)
        )
        info_mesh.add_field("Z Points", self.z_points)
        info_mesh.add_field(
            "Z Mesh Spacing",
            utils.fmt_quantity(2 * self.z_bound / self.z_points, utils.LENGTH_UNITS),
        )
        info_mesh.add_field(
            "X Boundary", utils.fmt_quantity(self.x_bound, utils.LENGTH_UNITS)
        )
        info_mesh.add_field("X Points", self.x_points)
        info_mesh.add_field(
            "X Mesh Spacing",
            utils.fmt_quantity(2 * self.x_bound / self.z_points, utils.LENGTH_UNITS),
        )

        info.add_info(info_mesh)

        return info


class CylindricalSliceSpecification(MeshSpecification):
    """A concrete :class:`MeshSpecification` for a :class:`MeshSimulation` with a :class:`CylindricalSliceMesh`."""

    mesh_type = meshes.CylindricalSliceMesh

    def __init__(
        self,
        name: str,
        z_bound: float = 20 * u.bohr_radius,
        rho_bound: float = 20 * u.bohr_radius,
        z_points: int = 2 ** 9,
        rho_points: int = 2 ** 8,
        operators=mesh_operators.CylindricalSliceLengthGaugeOperators(),
        evolution_method=evolution_methods.AlternatingDirectionImplicit(),
        **kwargs,
    ):
        """
        Parameters
        ----------
        z_bound
            The symmetric bounds of the simulation in :math:`z` (i.e., the simulation region goes from :math:`-z` to :math:`+z`.
        rho_bound
            The bound of the simulation in :math:`\\rho` (i.e., the simulation region from :math:`\\sim 0` to :math:`\\sim \\rho`).
        z_points
            The number of coordinate points for the z-dimension.
        rho_points
            The number of coordinate points for the rho-dimension.
        kwargs
            Any additional keyword arguments are passed to the :class:`MeshSpecification` constructor.
        """
        super().__init__(
            name, operators=operators, evolution_method=evolution_method, **kwargs
        )

        self.z_bound = z_bound
        self.rho_bound = rho_bound
        self.z_points = int(z_points)
        self.rho_points = int(rho_points)

    def info(self) -> si.Info:
        info = super().info()

        info_mesh = si.Info(header=f"Mesh: {self.mesh_type.__name__}")
        info_mesh.add_field(
            "Z Boundary", utils.fmt_quantity(self.z_bound, utils.LENGTH_UNITS)
        )
        info_mesh.add_field("Z Points", self.z_points)
        info_mesh.add_field(
            "Z Mesh Spacing",
            utils.fmt_quantity(2 * self.z_bound / self.z_points, utils.LENGTH_UNITS),
        )
        info_mesh.add_field(
            "Rho Boundary", utils.fmt_quantity(self.rho_bound, utils.LENGTH_UNITS)
        )
        info_mesh.add_field("Rho Points", self.rho_points)
        info_mesh.add_field(
            "Rho Mesh Spacing",
            utils.fmt_quantity(self.rho_bound / self.rho_points, utils.LENGTH_UNITS),
        )
        info_mesh.add_field("Total Mesh Points", int(self.z_points * self.rho_points))

        info.add_info(info_mesh)

        return info


class SphericalSliceSpecification(MeshSpecification):
    """A concrete :class:`MeshSpecification` for a :class:`MeshSimulation` with a :class:`SphericalSliceMesh`."""

    mesh_type = meshes.SphericalSliceMesh

    def __init__(
        self,
        name: str,
        r_bound: float = 20 * u.bohr_radius,
        r_points: int = 2 ** 10,
        theta_points: int = 2 ** 10,
        operators: mesh_operators.MeshOperators = mesh_operators.SphericalSliceLengthGaugeOperators(),
        evolution_method: evolution_methods.EvolutionMethod = evolution_methods.AlternatingDirectionImplicit(),
        **kwargs,
    ):
        """
        Parameters
        ----------
        r_bound
            The outer radius of the simulation region.
        r_points
            The number of points to use for the :math:`r` coordinate.
        theta_points
            The number of points to use for the :math:`\\theta` coordinate (:math:`\\theta` always goes from :math:`0` to :math:`\\pi`).
        kwargs
            Any additional keyword arguments are passed to the :class:`MeshSpecification` constructor.
        """
        super().__init__(
            name, operators=operators, evolution_method=evolution_method, **kwargs
        )

        self.r_bound = r_bound

        self.r_points = int(r_points)
        self.theta_points = int(theta_points)

    def info(self) -> si.Info:
        info = super().info()

        info_mesh = si.Info(header=f"Mesh: {self.mesh_type.__name__}")
        info_mesh.add_field(
            "R Boundary", utils.fmt_quantity(self.r_bound, utils.LENGTH_UNITS)
        )
        info_mesh.add_field("R Points", self.r_points)
        info_mesh.add_field(
            "R Mesh Spacing",
            utils.fmt_quantity(self.r_bound / self.r_points, utils.LENGTH_UNITS),
        )
        info_mesh.add_field("Theta Points", self.theta_points)
        info_mesh.add_field(
            "Theta Mesh Spacing",
            utils.fmt_quantity(u.pi / self.theta_points, utils.ANGLE_UNITS),
        )
        info_mesh.add_field(
            "Maximum Adjacent-Point Spacing",
            utils.fmt_quantity(
                u.pi * self.r_bound / self.theta_points, utils.LENGTH_UNITS
            ),
        )
        info_mesh.add_field("Total Mesh Points", int(self.r_points * self.theta_points))

        info.add_info(info_mesh)

        return info


class SphericalHarmonicSimulation(MeshSimulation):
    """
    An extended :class:`MeshSimulation` that takes advantage of particular
    features of :class:`SphericalHarmonicMesh`.
    """

    def check(self):
        super().check()

        g_for_largest_l = self.mesh.g[-1]
        norm_in_largest_l = self.mesh.state_overlap(g_for_largest_l, g_for_largest_l)

        if norm_in_largest_l > 1e-6:
            msg = f"Wavefunction norm in largest angular momentum state is large at time index {self.time_index} (norm at bound = {norm_in_largest_l}, fraction of norm = {norm_in_largest_l / self.data.norm[self.data_time_index]}), consider increasing l bound"
            logger.warning(msg)
            self.warnings["norm_in_largest_l"].append(
                core.WarningRecord(self.time_index, msg)
            )


class SphericalHarmonicSpecification(MeshSpecification):
    """
    A concrete :class:`MeshSpecification` for a :class:`MeshSimulation` with a
    :class:`SphericalHarmonicMesh`.

    Unlike other :class:`MeshSpecification`,
    :class:`SphericalHarmonicSpecification` produces a
    :class:`SphericalHarmonicSimulation` instead of a
    :class:`MeshSimulation`.
    """

    simulation_type = SphericalHarmonicSimulation
    mesh_type = meshes.SphericalHarmonicMesh
    simulation_plotter_type = sim_plotters.SphericalHarmonicSimulationPlotter

    def __init__(
        self,
        name: str,
        r_bound: float = 100 * u.bohr_radius,
        r_points: int = 1000,
        l_bound: int = 300,
        theta_points: int = 180,
        operators: mesh_operators.MeshOperators = mesh_operators.SphericalHarmonicLengthGaugeOperators(),
        evolution_method: evolution_methods.EvolutionMethod = evolution_methods.SplitInteractionOperator(),
        use_numeric_eigenstates: bool = True,
        numeric_eigenstate_max_energy: float = 20 * u.eV,
        numeric_eigenstate_max_angular_momentum: int = 5,
        **kwargs,
    ):
        """
        Parameters
        ----------
        r_bound
            The outer radius of the simulation region.
        r_points
            The number of points to use for the :math:`r` coordinate.
        l_bound
            The maximum orbital angular momentum to track.
        theta_points
            The number of points to use for the :math:`\\theta` coordinate (:math:`\\theta` always goes from :math:`0` to :math:`\\pi`).
            This is only used for display purposes, and has no impact on evolution calculations.
        use_numeric_eigenstates
            If ``True``, the ``test_states`` will be replaced by numeric eigenstates generated from the field-free evolution operators.
        numeric_eigenstate_max_energy
            The maximum energy to keep from the generated numeric eigenstates.
        numeric_eigenstate_max_angular_momentum
            The maximum angular momentum to keep from the generated numeric eigenstates.
        kwargs
            Any additional keyword arguments are passed to the :class:`MeshSpecification` constructor.
        """
        super().__init__(
            name, operators=operators, evolution_method=evolution_method, **kwargs
        )

        self.r_bound = r_bound
        self.r_points = int(r_points)
        self.l_bound = l_bound
        self.theta_points = theta_points
        self.spherical_harmonics = tuple(
            si.math.SphericalHarmonic(l, 0) for l in range(self.l_bound)
        )

        self.use_numeric_eigenstates = use_numeric_eigenstates
        self.numeric_eigenstate_max_angular_momentum = min(
            self.l_bound - 1, numeric_eigenstate_max_angular_momentum
        )
        self.numeric_eigenstate_max_energy = numeric_eigenstate_max_energy

    def info(self) -> si.Info:
        info = super().info()

        info_mesh = si.Info(header=f"Mesh: {self.mesh_type.__name__}")
        info_mesh.add_field(
            "R Boundary", utils.fmt_quantity(self.r_bound, utils.LENGTH_UNITS)
        )
        info_mesh.add_field("R Points", self.r_points)
        info_mesh.add_field(
            "R Mesh Spacing",
            utils.fmt_quantity(self.r_bound / self.r_points, utils.LENGTH_UNITS),
        )
        info_mesh.add_field("L Bound", self.l_bound)
        info_mesh.add_field("Total Mesh Points", self.r_points * self.l_bound)

        info.add_info(info_mesh)

        info_eigenstates = si.Info(
            header=f"Numeric Eigenstates: {self.use_numeric_eigenstates}"
        )
        if self.use_numeric_eigenstates:
            info_eigenstates.add_field(
                "Max Energy",
                utils.fmt_quantity(
                    self.numeric_eigenstate_max_energy, utils.ENERGY_UNITS
                ),
            )
            info_eigenstates.add_field(
                "Max Angular Momentum", self.numeric_eigenstate_max_angular_momentum
            )

        info.add_info(info_eigenstates)

        return info
