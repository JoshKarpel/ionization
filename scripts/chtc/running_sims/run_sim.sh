#!/usr/bin/env bash

set -e

tar -xzf python.tar.gz
tar -xzf simulacra.tar.gz
tar -xzf ionization.tar.gz

export PATH=$(pwd)/python/bin:$PATH
export LDFLAGS="-L$(pwd)/python/lib $LDFLAGS"
export C_INCLUDE_PATH="$(pwd)/python/include:$C_INCLUDE_PATH"
export LD_LIBRARY_PATH="$(pwd)/python/lib:$LD_LIBRARY_PATH"

python run_sim.py $1
