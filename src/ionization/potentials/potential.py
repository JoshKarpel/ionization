import logging

import simulacra as si

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PotentialEnergy(si.summables.Summand):
    """A class representing some kind of potential energy. Can be summed to form a PotentialEnergySum."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.summation_class = PotentialEnergySum


class PotentialEnergySum(si.summables.Sum, PotentialEnergy):
    """A class representing a combination of potential energies."""

    container_name = 'potentials'

    def get_electric_field_amplitude(self, t):
        return sum(x.get_electric_field_amplitude(t) for x in self._container)

    def get_electric_field_integral_numeric(self, t):
        return sum(x.get_electric_field_integral_numeric(t) for x in self._container)

    def get_vector_potential_amplitude(self, t):
        return sum(x.get_vector_potential_amplitude(t) for x in self._container)

    def get_vector_potential_amplitude_numeric(self, times, rule = 'simps'):
        return sum(x.get_vector_potential_amplitude_numeric(times, rule = rule) for x in self._container)

    def get_electric_field_integral_numeric_cumulative(self, times):
        """Return the integral of the electric field amplitude from the start of times for each interval in times."""
        return sum(x.get_electric_field_integral_numeric_cumulative(times) for x in self._container)

    def get_vector_potential_amplitude_numeric_cumulative(self, times):
        return sum(x.get_vector_potential_amplitude_numeric_cumulative(times) for x in self._container)

    def get_fluence_numeric(self, times, rule = 'simps'):
        return sum(x.get_fluence_numeric(times, rule = rule) for x in self._container)


class NoPotentialEnergy(PotentialEnergy):
    """A class representing no potential energy from any source."""

    def __call__(self, *args, **kwargs):
        """Return 0 for any arguments."""
        return 0
