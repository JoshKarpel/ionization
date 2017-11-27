import pytest
import hypothesis as hyp
import hypothesis.strategies as st

import numpy as np

import ionization as ion
import ionization.integrodiff as ide
from simulacra.units import *


def test_hydrogen_bessel_kernel_factory_returns_singleton():
    first_call = ide._hydrogen_kernel_LEN_factory()
    second_call = ide._hydrogen_kernel_LEN_factory()

    assert first_call is second_call

