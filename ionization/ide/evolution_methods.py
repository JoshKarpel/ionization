import logging
import enum
from abc import ABC, abstractmethod

import numpy as np

import simulacra as si
import simulacra.units as u

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TimeStepType(enum.Enum):
    FIXED = "fixed"
    ADAPTIVE = "adaptive"


class EvolutionMethod(ABC):
    """
    An abstract class for evolution methods for IDE simulations.

    Any subclass must have a class attribute called `time_step_type` that is assigned to a member of :class:`TimeStepType`.
    Which one obviously depends on what kind of time step the method uses.

    Subclasses should implement the abstract ``evolve`` method.
    """

    time_step_type = None

    @abstractmethod
    def evolve(self, sim, b, times, time_step) -> complex:
        """
        Given the history of the bound state amplitude ``b``, construct ``b`` at the last of the ``times`` plus the ``time_step``.

        Parameters
        ----------
        sim
            The :class:`IDESimulation`.
        b
            The history of the bound state amplitude, :math:`b(t)`.
        times
            The times at which ``b`` is given, :math:`t`.
        time_step
            The amount of time to evolve forward by.

        Returns
        -------
        next_b
            The next value of ``b``, at ``times[-1] + time_step``.
        """
        raise NotImplementedError

    def info(self) -> si.Info:
        info = si.Info(header=f"Evolution Method: {self.__class__.__name__}")

        info.add_field("Time Step Type", self.time_step_type.value.title())

        return info


class ForwardEulerMethod(EvolutionMethod):
    """
    The forward Euler method.
    This method is first order in time, and not particularly stable.
    See `Forward Euler method <https://en.wikipedia.org/wiki/Euler_method>`_ for examples.
    """

    time_step_type = TimeStepType.FIXED

    def evolve(self, sim, b, times, time_step):
        times = np.array(times)
        time_curr = times[-1]

        fs_curr = sim.f(times)
        f_curr = fs_curr[-1]

        derivative = (
            sim.spec.integral_prefactor
            * f_curr
            * sim.integrate(
                y=fs_curr * sim.evaluate_kernel(time_curr, times) * np.array(b), x=times
            )
        )

        b_next = b[-1] + (time_step * derivative)
        t_next = time_curr + time_step

        return b_next, t_next


class BackwardEulerMethod(EvolutionMethod):
    """
    The backward Euler method.
    This method is first order in time, but is much more stable than forward Euler.
    See `Backward Euler method <https://en.wikipedia.org/wiki/Backward_Euler_method>`_ for examples.
    """

    time_step_type = TimeStepType.FIXED

    def evolve(self, sim, b, times, time_step):
        times_next = np.array(times + [times[-1] + time_step])
        times = np.array(times)

        time_curr = times[-1]

        fs_next = sim.f(times_next)
        fs_curr = fs_next[:-1]
        f_next = fs_next[-1]

        derivative = (
            sim.spec.integral_prefactor
            * f_next
            * sim.integrate(
                y=fs_curr
                * sim.evaluate_kernel(time_curr + time_step, times)
                * np.array(b),
                x=times,
            )
        )

        b_next = (b[-1] + (time_step * derivative)) / (
            1
            - sim.spec.integral_prefactor
            * sim.evaluate_kernel(time_curr, np.array([time_curr]))
            * ((time_step * f_next) ** 2)
        )
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

        derivative_forward = (
            sim.spec.integral_prefactor
            * f_next
            * sim.integrate(
                y=fs_curr_times_b * sim.evaluate_kernel(time_curr + time_step, times),
                x=times,
            )
        )

        derivative_backward = (
            sim.spec.integral_prefactor
            * f_curr
            * sim.integrate(
                y=fs_curr_times_b * sim.evaluate_kernel(time_curr, times), x=times
            )
        )

        b_next = b[-1] + (time_step * (derivative_forward + derivative_backward) / 2)
        b_next /= 1 - (
            0.5
            * sim.spec.integral_prefactor
            * sim.evaluate_kernel(time_curr, np.array([time_curr]))
            * ((time_step * f_next) ** 2)
        )

        t_next = time_curr + time_step

        return b_next, t_next


class RungeKuttaFourMethod(EvolutionMethod):
    """
    The fourth-order Runge-Kutta method.
    This method is fourth-order in time.
    See `Wikipedia <https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#The_Runge–Kutta_method>`_ for discussion.
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
        f_curr = fs_next[-2]
        fs_curr = fs_next[:-1]
        f_next = fs_next[-1]

        kernel_curr = sim.evaluate_kernel(time_curr, times)
        kernel_half = sim.evaluate_kernel(time_half, times_half)
        kernel_next = sim.evaluate_kernel(time_next, times_next)

        fs_curr_times_b = fs_curr * np.array(b)

        # STEP 1
        k1 = (sim.spec.integral_prefactor * f_curr) * sim.integrate(
            y=fs_curr_times_b * kernel_curr, x=times
        )
        b_midpoint_for_k2 = b_curr + ((time_step / 2) * k1)

        # STEP 2
        k2 = (
            sim.spec.integral_prefactor
            * f_half
            * sim.integrate(
                y=np.append(fs_curr_times_b, f_half * b_midpoint_for_k2) * kernel_half,
                x=times_half,
            )
        )
        b_midpoint_for_k3 = b_curr + ((time_step / 2) * k2)

        # STEP 3
        k3 = (
            sim.spec.integral_prefactor
            * f_half
            * sim.integrate(
                y=np.append(fs_curr_times_b, f_half * b_midpoint_for_k3) * kernel_half,
                x=times_half,
            )
        )
        b_end_for_k4 = b_curr + (time_step * k3)

        # STEP 4

        k4 = (
            sim.spec.integral_prefactor
            * f_next
            * sim.integrate(
                y=np.append(fs_curr_times_b, f_next * b_end_for_k4) * kernel_next,
                x=times_next,
            )
        )

        # FINAL ESTIMATE
        b_next = b_curr + ((time_step / 6) * (k1 + (2 * (k2 + k3)) + k4))
        t_next = time_next

        return b_next, t_next


class AdaptiveRungeKuttaFourMethod(RungeKuttaFourMethod):
    """
    An adaptive fourth-order Runge-Kutta algorithm.
    """

    time_step_type = TimeStepType.ADAPTIVE

    def __init__(
        self,
        time_step_min: float = 0.1 * u.asec,
        time_step_max: float = 1 * u.asec,
        epsilon: float = 1e-6,
        error_on: str = "db/dt",
        safety_factor: float = 0.98,
    ):
        """
        Parameters
        ----------
        time_step_min
            The minimum time step that can be used by the adaptive algorithm.
        time_step_max
            The maximum time step that can be used by the adaptive algorithm.
        epsilon
            The acceptable fractional error in the quantity specified by `error_on`.
        error_on : {``'b'``, ``'db/dt'``}
            Which quantity to control the fractional error in.
        safety_factor
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
        a_double_step_estimate, _ = super().evolve(
            sim, b + [a_half_step_estimate], times + [time_half], time_step / 2
        )

        delta_1 = (
            a_double_step_estimate - b_full_step_estimate
        )  # estimate truncation error from difference in estimates of a

        if sim.spec.error_on == "b":
            delta_0 = self.epsilon * b[-1]
        elif sim.spec.error_on == "db/dt":
            delta_0 = self.epsilon * (b[-1] - a_double_step_estimate)

        ratio = np.abs(delta_0 / delta_1)

        old_step = sim.time_step  # for log message
        if (
            ratio >= 1
            or np.isinf(ratio)
            or np.isnan(ratio)
            or time_step == self.time_step_min
        ):  # step was ok
            if delta_1 != 0:  # don't adjust time step if truncation error is zero
                sim.time_step = self.safety_factor * time_step * (ratio ** (1 / 5))

            # ensure new time step is inside min and max allowable time steps
            sim.time_step = min(self.time_step_max, sim.time_step)
            sim.time_step = max(self.time_step_min, sim.time_step)

            logger.debug(
                f"Accepted ARK4 step to {times[-1] + time_step / u.asec:.6f)} as. Changed time step to {sim.time_step / u.asec:6f} as from {old_step / u.asec:6f} as"
            )

            return a_double_step_estimate + (delta_1 / 15), time_curr + time_step
        else:  # reject step
            sim.time_step = (
                self.safety_factor * time_step * (ratio ** (1 / 4))
            )  # set new time step
            sim.time_step = min(self.time_step_max, sim.time_step)
            sim.time_step = max(self.time_step_min, sim.time_step)

            logger.debug(
                f"Rejected ARK4 step. Changed time step to {sim.time_step / u.asec:6f} as from {old_step / u.asec:6f} as"
            )
            return self.evolve(sim, b, times, sim.time_step)  # retry with new time step

    def info(self) -> si.Info:
        info = super().info()

        info.add_field("Error Control On", self.error_on)
        info.add_field("Epsilon", self.epsilon)
        info.add_field("Safety Factor", self.safety_factor)
        info.add_field("Minimum Time Step", f"{self.time_step_min / u.asec:3f} as")
        info.add_field("Maximum Time Step", f"{self.time_step_max / u.asec:3f} as")

        return info
