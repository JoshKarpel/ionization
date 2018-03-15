Quantum States
==============

``ionization`` implements a somewhat-sophisticated system of modelling quantum states abstractly.
Each :class:`QuantumState` carries with it a lot of information about the state's properties.

States can be added together, and they can be multiplied by real and complex numbers.
However, we do not support product states or continuous superpositions, so all of our states represent discrete superpositions of the state of a single particle.
