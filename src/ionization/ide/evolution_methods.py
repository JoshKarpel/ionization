import logging
import enum
from abc import ABC, abstractmethod

import numpy as np

import simulacra as si
import simulacra.units as u

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TimeStepType(enum.Enum):
    FIXED = 'fixed'
    ADAPTIVE = 'adaptive'


class EvolutionMethod(ABC):
    """
    An abstract class for evolution methods for IDE simulations.

    Any subclass must have a class attribute called `time_step_type` that is assigned to a member of :class:`TimeStepType`.
    Which one obviously depends on what kind of time step the method uses.
    """

    @abstractmethod
    def evolve(self, sim, b, times, time_step):
        raise NotImplementedError

    def info(self):
        info = si.Info(header = f'Evolution Method: {self.__class__.__name__}')

        info.add_field('Time Step Type', self.time_step_type)

        return info


class ForwardEulerMethod(EvolutionMethod):
    """
    The forward Euler method.
    This method is first order in time, and not particularly stable.
    See `Wikipedia<https://en.wikipedia.org/wiki/Euler_method>`_ for examples.
    """

    time_step_type = TimeStepType.FIXED

    def evolve(self, sim, b, times, time_step):
        times = np.array(times)
        time_curr = times[-1]

        fs_curr = sim.f(times)
        f_curr = fs_curr[-1]

        kernel = sim.evaluate_kernel(time_curr, times)

        integral = sim.integrate(
            y = fs_curr * kernel * np.array(b),
            x = times
        )
        derivative = sim.spec.integral_prefactor * f_curr * integral

        b_next = b[-1] + (time_step * derivative)
        t_next = time_curr + time_step

        return b_next, t_next


class BackwardEulerMethod(EvolutionMethod):
    """
    The backward Euler method.
    This method is first order in time, but is much more stable than forward Euler.
    See `Wikipedia<https://en.wikipedia.org/wiki/Backward_Euler_method>`_ for examples.
    """

    time_step_type = TimeStepType.FIXED

    def evolve(self, sim, b, times, time_step):
        times_next = np.array(times + [times[-1] + time_step])
        times = np.array(times)

        time_curr = times[-1]

        fs_next = sim.f(times_next)
        fs_curr = fs_next[:-1]
        f_next = fs_next[-1]

        kernel = sim.evaluate_kernel(time_curr + time_step, times)

        integral = sim.integrate(
            y = fs_curr * kernel * np.array(b),
            x = times
        )
        derivative = sim.spec.integral_prefactor * f_next * integral

        b_next = (b[-1] + (time_step * derivative)) / (1 - sim.spec.integral_prefactor * sim.evaluate_kernel(0, 0) * ((time_step * f_next) ** 2))
        t_next = time_curr + time_step

        return b_next, t_next


class TrapezoidMethod(EvolutionMethod):
    """
    The trapezoid method, which is the combination of the forward and backward Euler methods.
    This method is second order in time and shares the stability properties of the backward Euler method.
    """

    time_step_type = TimeStepType.FIXED

    def evolve(self, sim, b, times, time_step):
        times_next = np.array(times + [times[-1] + time_step])
        times = np.array(times)
        time_curr = times[-1]

        fs_next = sim.f(times_next)
        fs_curr = fs_next[:-1]
        f_curr = fs_next[-2]
        f_next = fs_next[-1]

        fs_curr_times_b = fs_curr * np.array(b)

        kernel_1 = sim.evaluate_kernel(time_curr + time_step, times)
        integral_forward = sim.integrate(
            y = fs_curr_times_b * kernel_1,
            x = times
        )
        derivative_forward = sim.spec.integral_prefactor * f_next * integral_forward

        kernel_2 = sim.evaluate_kernel(time_curr, times)
        integral_backward = sim.integrate(
            y = fs_curr_times_b * kernel_2,
            x = times
        )
        derivative_backward = sim.spec.integral_prefactor * f_curr * integral_backward

        b_next = (b[-1] + (time_step * (derivative_forward + derivative_backward) / 2))
        b_next /= (1 - (.5 * sim.spec.integral_prefactor * sim.evaluate_kernel(0, 0) * ((time_step * f_next) ** 2)))

        t_next = time_curr + time_step

        return b_next, t_next


class RungeKuttaFourMethod(EvolutionMethod):
    """
    The fourth-order Runge-Kutta method.
    This method is fourth-order in time.
    See `Wikipedia<https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#The_Runge–Kutta_method>` for discussion.
    """

    time_step_type = TimeStepType.FIXED

    def evolve(self, sim, b, times, time_step):
        b_curr = b[-1]

        time_curr = times[-1]
        time_half = time_curr + (time_step / 2)
        time_next = time_curr + time_step

        times_half = np.array(times + [time_half])
        times_next = np.array(times + [time_next])
        times = np.array(times)

        fs_half = sim.f(times_half)
        f_half = fs_half[-1]

        fs_next = sim.f(times_next)
        f_next = fs_next[-1]
        f_curr = fs_next[-2]
        fs_curr = fs_next[:-1]

        kernel_curr = sim.evaluate_kernel(time_curr, times)
        kernel_half = sim.evaluate_kernel(time_half, times_half)
        kernel_next = sim.evaluate_kernel(time_next, times_next)

        fs_curr_times_b = fs_curr * np.array(b)

        # integrate through the current time step
        integrand_for_k1 = fs_curr_times_b * kernel_curr
        integral_for_k1 = sim.integrate(y = integrand_for_k1, x = times)
        k1 = sim.spec.integral_prefactor * f_curr * integral_for_k1
        b_midpoint_for_k2 = b_curr + (time_step * k1 / 2)  # time_step / 2 here because we moved forward to midpoint

        integrand_for_k2 = np.append(fs_curr_times_b, f_half * b_midpoint_for_k2) * kernel_half
        integral_for_k2 = sim.integrate(y = integrand_for_k2, x = times_half)
        k2 = sim.spec.integral_prefactor * f_half * integral_for_k2  # time_step / 2 because it's half of an interval that we're integrating over
        b_midpoint_for_k3 = b_curr + (time_step * k2 / 2)  # estimate midpoint based on estimate of slope at midpoint

        integrand_for_k3 = np.append(fs_curr_times_b, f_half * b_midpoint_for_k3) * kernel_half
        integral_for_k3 = sim.integrate(y = integrand_for_k3, x = times_half)
        k3 = sim.spec.integral_prefactor * f_half * integral_for_k3
        b_end_for_k4 = b_curr + (time_step * k3)

        integrand_for_k4 = np.append(fs_curr_times_b, f_next * b_end_for_k4) * kernel_next
        integral_for_k4 = sim.integrate(y = integrand_for_k4, x = times_next)
        k4 = sim.spec.integral_prefactor * f_next * integral_for_k4

        b_next = b_curr + (time_step * (k1 + (2 * k2) + (2 * k3) + k4) / 6)
        t_next = time_next

        return b_next, t_next


class AdaptiveRungeKuttaFourMethod(RungeKuttaFourMethod):
    """
    An adaptive fourth-order Runge-Kutta method.
    """

    time_step_type = TimeStepType.ADAPTIVE

    def __init__(self,
                 time_step_min = .01 * u.asec,
                 time_step_max = 1 * u.asec,
                 epsilon = 1e-6,
                 error_on = 'db/dt',
                 safety_factor = .98):
        """
        Parameters
        ----------
        time_step_min : :class:`float`
            The minimum time step that can be used by the adaptive algorithm.
        time_step_max : :class:`float`
            The maximum time step that can be used by the adaptive algorithm.
        epsilon : :class:`float`
            The acceptable fractional error in the quantity specified by `error_on`.
        error_on : {``'b'``, ``'db/dt'``}
            Which quantity to control the fractional error in.
        safety_factor : :class:`float`
            The safety factor that new time steps are multiplicatively fudged by.
        """
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.epsilon = epsilon
        self.error_on = error_on
        self.safety_factor = safety_factor

    def evolve(self, sim, b, times, time_step):
        b_full_step_estimate, _ = super().evolve(sim, b, times, time_step)

        time_curr = times[-1]
        time_half = time_curr + (time_step / 2)

        a_half_step_estimate, _ = super().evolve(sim, b, times, time_step / 2)
        a_double_step_estimate, _ = super().evolve(sim, b + [a_half_step_estimate], times + [time_half], time_step / 2)

        delta_1 = a_double_step_estimate - b_full_step_estimate  # estimate truncation error from difference in estimates of a

        if sim.spec.error_on == 'b':
            delta_0 = self.epsilon * b[-1]
        elif sim.spec.error_on == 'db/dt':
            delta_0 = self.epsilon * (b[-1] - a_double_step_estimate)

        ratio = np.abs(delta_0 / delta_1)

        old_step = sim.time_step  # for log message
        if ratio >= 1 or np.isinf(ratio) or np.isnan(ratio) or time_step == self.time_step_min:  # step was ok
            if delta_1 != 0:  # don't adjust time step if truncation error is zero
                sim.time_step = self.safety_factor * time_step * (ratio ** (1 / 5))

            # ensure new time step is inside min and max allowable time steps
            sim.time_step = min(self.time_step_max, sim.time_step)
            sim.time_step = max(self.time_step_min, sim.time_step)

            logger.debug(f'Accepted ARK4 step to {u.uround(times[-1] + time_step, u.asec, 6)} as. Changed time step to {u.uround(sim.time_step, u.asec, 6)} as from {u.uround(old_step, u.asec, 6)} as')

            return a_double_step_estimate + (delta_1 / 15), time_curr + time_step
        else:  # reject step
            sim.time_step = self.safety_factor * time_step * (ratio ** (1 / 4))  # set new time step
            sim.time_step = min(self.time_step_max, sim.time_step)
            sim.time_step = max(self.time_step_min, sim.time_step)

            logger.debug(f'Rejected ARK4 step. Changed time step to {u.uround(sim.time_step, u.asec, 6)} as from {u.uround(old_step, u.asec, 6)} as')
            return self.evolve(sim, b, times, sim.time_step)  # retry with new time step

    def info(self):
        info = super().info()

        info.add_field('Error Control On', self.error_on)
        info.add_field('Epsilon', self.epsilon)
        info.add_field('Safety Factor', self.safety_factor)
        info.add_field('Minimum Time Step', f'{u.uround(self.time_step_min, u.asec, 3)} as')
        info.add_field('Maximum Time Step', f'{u.uround(self.time_step_max, u.asec, 3)} as')

        return info
