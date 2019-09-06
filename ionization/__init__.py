import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.NullHandler())

import os as _os

# see https://github.com/ContinuumIO/anaconda-issues/issues/905
_os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

from .version import __version__, version, version_info

# set up platform-independent runtime cython compilation and imports
import numpy as _np
import pyximport as _pyximport

_pyx_dir = os.path.join(os.path.dirname(__file__), ".pyxbld")
_pyximport.install(
    setup_args={"include_dirs": _np.get_include()}, build_dir=_pyx_dir, language_level=3
)

from . import mesh, ide, tunneling, potentials, states, exceptions, analysis, vis
from .core import Gauge
