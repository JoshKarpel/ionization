Tunneling Model Simulations
===========================

.. currentmodule:: ionization.tunneling

Specifications
--------------

.. autoclass:: TunnelingSpecification

    .. automethod:: info

Simulations
-----------

.. autoclass:: TunnelingSimulation

    .. automethod:: run

    .. automethod:: info

Tunneling Models
----------------

Each :class:`TunnelingModel` subclass should implement a ``_tunneling_rate`` method which calculates the tunneling rate of :math:`b` based on the instantaneous electric field amplitude.
Call the `tunneling_rate` (no prefixing ``_``) to calculate the tunneling rate taking into account cutoffs and other features from the base :class:`TunnelingModel`.

.. autoclass:: TunnelingModel

    .. automethod:: tunneling_rate

    .. automethod:: _tunneling_rate

.. autoclass:: NoTunneling

.. autoclass:: LandauRate

.. autoclass:: KeldyshRate

.. autoclass:: PosthumusRate

.. autoclass:: MulserRate

.. autoclass:: ADKRate
