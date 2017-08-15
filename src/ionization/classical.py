import collections
import functools as ft
import itertools as it
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

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Force(si.Summand):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.summation_class = Forces


class Forces(si.Sum):
    container_name = 'forces'


class NoForce(si.Summand):
    def __call__(self, t, **kwargs):
        return 0


class CoulombForce(Force):
    def __init__(self, position, charge = proton_charge):
        super().__init__()

        self.position = np.array(position)
        self.charge = charge

    def __call__(self, t, *, position, test_charge, **kwargs):
        r_vec = position - self.position
        denom = np.sum(r_vec ** 2, axis = 1) ** 1.5
        return coulomb_constant * self.charge * test_charge * r_vec / denom


class ClassicalSimulation(si.Simulation):
    def __init__(self, spec):
        super().__init__(spec)

        total_time = self.spec.time_final - self.spec.time_initial
        self.times = np.linspace(self.spec.time_initial, self.spec.time_final, int(total_time / self.spec.time_step) + 1)
        self.time_steps = len(self.times)
        self.time_index = 0

        # indexing on these is [time_index, particle_number, position/velocity components]
        self.positions = np.empty((len(self.times), *np.shape(self.spec.initial_positions)), dtype = np.float64) * np.NaN
        self.velocities = np.empty((len(self.times), *np.shape(self.spec.initial_velocities)), dtype = np.float64) * np.NaN

        self.positions[0] = self.spec.initial_positions
        self.velocities[0] = self.spec.initial_velocities

        print(self.positions[0], self.velocities[0])

    def time(self):
        return self.times[self.time_index]

    @si.utils.memoize
    def eval_force(self, time_index):
        return self.spec.force(
            self.time,
            position = self.positions[time_index],
            velocity = self.velocities[time_index],
            test_mass = self.spec.test_mass,
            test_charge = self.spec.test_charge,
        )

    def evolve(self):
        getattr(self, f'evolve_{self.spec.evolution_method}')()

    def evolve_FE(self):
        acc = self.eval_force(self.time_index - 1) / self.spec.test_mass
        self.velocities[self.time_index] = self.velocities[self.time_index - 1] + (acc * self.spec.time_step)
        self.positions[self.time_index] = self.positions[self.time_index - 1] + (self.velocities[self.time_index - 1] * self.spec.time_step)

    def evolve_VV(self):
        acc_start = self.eval_force(self.time_index - 1) / self.spec.test_mass
        self.positions[self.time_index] = self.positions[self.time_index - 1] + (self.velocities[self.time_index - 1] * self.spec.time_step) + (.5 * acc_start * (self.spec.time_step ** 2))
        acc_end = self.eval_force(self.time_index) / self.spec.test_mass
        self.velocities[self.time_index] = self.velocities[self.time_index - 1] + (.5 * (acc_start + acc_end) * self.spec.time_step)

    def run_simulation(self, progress_bar = False, callback = None):
        logger.info(f'Performing time evolution on {self}, starting from time index {self.time_index}')
        try:
            self.status = si.Status.RUNNING

            for animator in self.spec.animators:
                animator.initialize(self)

            if progress_bar:
                pbar = tqdm(total = self.time_steps - 1)

            while True:
                if callback is not None:
                    callback(self)

                if self.time_index == self.time_steps - 1:
                    break

                self.time_index += 1

                self.evolve()

                logger.debug('{} {} ({}) evolved to time index {} / {} ({}%)'.format(self.__class__.__name__, self.name, self.file_name, self.time_index, self.time_steps - 1,
                                                                                     np.around(100 * (self.time_index + 1) / self.time_steps, 2)))

                if self.spec.checkpoints:
                    if (self.time_index + 1) % self.spec.checkpoint_every == 0:
                        self.save(target_dir = self.spec.checkpoint_dir)
                        self.status = si.Status.RUNNING
                        logger.info('Checkpointed {} {} ({}) at time step {} / {}'.format(self.__class__.__name__, self.name, self.file_name, self.time_index + 1, self.time_steps))

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

    def plot_particle_paths_2d(self, particle_slice = None, **kwargs):
        if particle_slice is None:
            particle_slice = slice(None)

        si.vis.xy_plot(
            f'{self.name}__particle_paths',
            self.positions[:, particle_slice, 0],
            self.positions[:, particle_slice, 1],
            x_lower_limit = -2 * bohr_radius, x_upper_limit = 2 * bohr_radius,
            y_lower_limit = -2 * bohr_radius, y_upper_limit = 2 * bohr_radius,
            x_unit = 'bohr_radius', y_unit = 'bohr_radius',
            square_axis = True,
            **kwargs,
        )

    def info(self):
        info = super().info()

        mem_pos_and_vel = self.positions.nbytes + self.velocities.nbytes

        mem_total = sum((
            mem_pos_and_vel,
        ))
        info_mem = si.Info(header = f'Memory Usage (approx.): {si.utils.bytes_to_str(mem_total)}')
        info_mem.add_field('Position and Velocity', si.utils.bytes_to_str(mem_pos_and_vel))

        info.add_info(info_mem)

        return info


class ClassicalSpecification(si.Specification):
    simulation_type = ClassicalSimulation

    def __init__(self, name,
                 initial_positions,
                 initial_velocities,
                 test_mass = electron_mass_reduced,
                 test_charge = electron_charge,
                 force = NoForce(),
                 time_initial = 0,
                 time_final = 1000 * asec,
                 time_step = 1 * asec,
                 evolution_method = 'FE',
                 checkpoints = False, checkpoint_every = 20, checkpoint_dir = None,
                 animators = tuple(),
                 **kwargs):
        super().__init__(name, **kwargs)

        self.initial_positions = np.array(initial_positions)
        self.initial_velocities = np.array(initial_velocities)

        self.test_mass = test_mass
        self.test_charge = test_charge

        self.force = force

        self.time_initial = time_initial
        self.time_final = time_final
        self.time_step = time_step

        self.evolution_method = evolution_method

        self.checkpoints = checkpoints
        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir = checkpoint_dir

        self.animators = deepcopy(tuple(animators))

    def info(self):
        info = super().info()

        info_evolution = si.Info(header = 'Time Evolution')
        info_evolution.add_field('Initial Time', f'{uround(self.time_initial, asec, 3)} as | {uround(self.time_initial, fsec, 3)} fs | {uround(self.time_initial, atomic_time, 3)} a.u.')
        info_evolution.add_field('Final Time', f'{uround(self.time_final, asec, 3)} as | {uround(self.time_final, fsec, 3)} fs | {uround(self.time_final, atomic_time, 3)} a.u.')
        info_evolution.add_field('Time Step', f'{uround(self.time_step, asec, 3)} as | {uround(self.time_step, atomic_time, 3)} a.u.')
        info.add_info(info_evolution)

        info_algorithm = si.Info(header = 'Evolution Algorithm')
        info_algorithm.add_field('Evolution Method', self.evolution_method)

        info.add_info(info_algorithm)

        forces = si.Info(header = 'Forces')
        for x in self.force:
            forces.add_info(x.info())

        info.add_info(forces)

        info_analysis = si.Info(header = 'Analysis')
        info_analysis.add_field('Test Charge', f'{uround(self.test_charge, proton_charge, 3)} e')
        info_analysis.add_field('Test Mass', f'{uround(self.test_mass, electron_mass, 3)} m_e | {uround(self.test_mass, electron_mass_reduced, 3)} mu_e')

        info.add_info(info_analysis)

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

        return info
