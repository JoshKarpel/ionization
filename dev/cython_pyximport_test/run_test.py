import os
import numpy as np
import pyximport

pyx_dir = os.path.join(os.path.dirname(__file__), '.pyxbld')
pyximport.install(
    setup_args = {"include_dirs": np.get_include()},
    build_dir = pyx_dir,
    language_level = 3
)

from cython_code import sum

x = sum(np.ones(10))
print(x)
assert x == 10
