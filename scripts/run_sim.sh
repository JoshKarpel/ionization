#!/bin/bash

printenv
echo ""

# untar my Python installation and simulation package
tar -xzf python.tar.gz
tar -xzf simulacra.tar.gz
tar -xzf ionization.tar.gz

# make sure the script will use my Python installation
export PATH=$(pwd)/python/bin:$PATH
echo "set python path"

export LDFLAGS="-L$(pwd)/python/lib $LDFLAGS"
echo "set LDFLAGS"

export C_INCLUDE_PATH="$(pwd)/python/include:$C_INCLUDE_PATH"
echo "set C_INCLUDE_PATH"

export LD_LIBRARY_PATH="$(pwd)/python/lib:$LD_LIBRARY_PATH"
echo "set LD_LIBRARY_PATH"

echo ""

printenv


# run the python script
python run_sim.py $1
