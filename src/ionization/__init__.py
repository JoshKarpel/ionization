import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.NullHandler())

# set up platform-independent runtime cython compilation and imports
import numpy as _np
import pyximport

pyx_dir = os.path.join(os.path.dirname(__file__), '.pyxbld')
pyximport.install(
    setup_args = {"include_dirs": _np.get_include()},
    build_dir = pyx_dir,
    language_level = 3,
)

from . import mesh, ide, tunneling, potentials, states, exceptions
from .core import Gauge
