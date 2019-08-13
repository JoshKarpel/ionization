import functools
from pathlib import Path
import random
import sys

import htmap

import numpy as np

import simulacra as si
import simulacra.units as u

import modulation
from modulation.resonators import microspheres
from modulation.raman import AUTO_CUTOFF

# CLI

from halo import Halo
from spinners import Spinners

CLI_CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

SPINNERS = list(name for name in Spinners.__members__ if "dots" in name)


def make_spinner(*args, **kwargs):
    return Halo(*args, spinner=random.choice(SPINNERS), stream=sys.stderr, **kwargs)


# SHARED QUESTIONS


def ask_for_tag():
    tag = si.cluster.ask_for_input("Map Tag?", default=None)
    if tag is None:
        raise ValueError("tag cannot be None")
    return tag
