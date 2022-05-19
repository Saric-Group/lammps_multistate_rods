# The *LAMMPS multistate rods* library

This is a Python library for running hybrid MD-MC simulations in LAMMPS of rigid, rod-like structures that can change between multiple (internal) states during a simulation. It was developed for (very) coarse-grained modeling of proteins with multiple functionally different conformations (but hopefully can find other applications).

The molecular dynamics (MD) part of the simulations is the moving around of the rods/particles/atoms due to forces/interactions, and the Monte Carlo (MC) part is the changing of the internal states of the rods and keeping (approximately) constant concentrations of rods of a given state (if desired).

In essence the library is (just) a collection of convenient classes and methods that provide for:
1. fast and easy building of such (coarse-grained, rod-like, multi-state) models,
2. for simplified making of LAMMPS (Python) simulation scripts to be run with such models, and
3. for (limited, but extendable) post-processing of simulation results.

## Overview and contents

This library is written in Python 2.7 and uses LAMMPS's Python library interface (`PyLammps`) to communicate with an instance of LAMMPS. It is fully compatible with MPI, meaning applications that use it (like the `simulate_nvt.py` in the *examples* directory) can be run with `mpiexec`.

The library is contained within the **lammps_multistate_rods** directory and is consisted of 2 main classes: `Rod_model` and `Simulation`, along with some other potentially useful tools (methods etc.) in the **tools** subdirectory.

A rod model is described in a configuration file, which is essentially a simplified Python text file with some additional requirements and special keywords. A `Rod_model` object takes in information from a rod model configuration file (via a `Rod_params` object that parses the file) and stores that information, along with some others it calculates/derives from those specified in the configuration file.

A `Simulation` object takes a `Rod_model` instance and a `PyLammps` instance, along with some other simulation-specific information (e.g. the temperature), and provides various convenience methods, parameters, LAMMPS constructs (e.g. groups, variables) etc. to easily and correctly setup and use LAMMPS for simulating the rods defined by the model. The main methods, besides the ones to setup the "box" (styles, interactions etc.) and create the rods, are those that set up fixes that define the behaviour of rods in the simulation, and those are:
1. `set_rod_dynamics` - sets a *rigid/small* fix of a given ensemble (NVE for example) on the group that contains all the rod atoms; this is because the idea is that the rods being simulated are rigid and small molecules
2. `set_state_transitions` - sets a *change/state* fix (developed by myself and not currently available in the standard LAMMPS version) that defines MC change-of-state moves for the rods of the simulation, as defined by the rod model configuration file (allowed transitions and energy penalties); the groups and fixes are set up in such a way that at each time there is a group containing all atoms of all rods in the same state, and only those atoms (i.e. dynamic groups for each state a rod can be in)
3. `set_state_concentration` - sets a *gcmc* fix for a given state that keeps the concentration of rods in *that* state approximately constant (using MC moves)

The **examples** directory contains examples of model configuration files (.cfg) and a console application / LAMMPS Python script that showcases the use of the library, along with an example run configuration file for that application.

The **tools** directory contains a console application that calculates and makes interactive figures of overall interactions between rods and particles defined in a model configuration file (.cfg).

The **docs** directory contains documents describing library features and how to use them (e.g. how to define a model).

## Installation / setup

In order to use this library one must simply download it and enable Python to find it. This can be done in a number of ways, some of which are:
1. add the path of the *lammps_multistate_rods* directory (the one that contains the class module files) to the `PYTHONPATH` environment variable (this has to be done every time for a new environment, which can be automated);
2. simply copy/put the *lammps_multistate_rods* directory in the local search path of your Python, for example in `~/.local/lib/python2.7/site-packages` (this has to be done every time you want a new version, which can also be automated, but is dirty);
3. make a symbolic link to the *lammps_multistate_rods* directory in the local search path of you Python, for example `ln -s <download_dir>/lammps_multistate_rods/ ~/.local/lib/python2.7/site-packages/lammps_multistate_rods`

In a related repository of mine (https://github.com/Saric-Group/amyloid-simulations) this library is used and there is a *setup_scripts* directory that might be very useful to check out. It contains scripts that setup everything (virtualenv and LAMMPS) and get the most current library. In that project I use the first of the methods listed above, that can be checked in any of the *job templates* in the same-named directory.

### Requirements

The requirements for using the library are:
1. **Python 2.7** with *numpy* and *pyquaternion*:
    ```
    pip install numpy pyquaternion
    ```
    * *scipy* and *matplotlib* are necessary for some tools, but not for the library itself (I think)
    * also *mpi4py* is necessary for the example `simulate_nvt.py` application (but that dependency can easily be removed by hand)

2. The *develop* branch of my LAMMPS fork built with the *molecule*, *rigid*, *mc*, *python* and *extra-pair* (for the *cosine/squared* pair potential) modules:
    ```
    git clone -b develop https://github.com/erozic/lammps.git <LAMMPS_dir>  
    cd <LAMMPS_dir>/src
    make no-all # also can't harm to clean, purge and package-update beforehand...
    make yes-molecule yes-rigid yes-mc yes-python yes-extra-pair # and any additoinal ones...
    make -j4 <target> mode=shared LMP_INC="-DLAMMPS_EXCEPTIONS" # <target> is the make configuration, e.g. serial, mpi or any other in <LAMMPS_dir>/src/MAKE/...
    make install-python
    ```
    * using my fork of LAMMPS is necessary (for now) because of the *change/state* fix which is not yet part of standard LAMMPS (although I soon plan it to be, and I keep this *develop* branch updated with the official LAMMPS *develop* branch). Also, it might contain some other non-essential features and will contain all future updates relevant for the optimal functioning of this library.

## Old version

The *stable-no-fix-change-state* branch contains the old version of the library that didn't use the *change/state* fix, but instead changed the states of rods and kept track of them itself (in the Python code) through the LAMMPS library.

Major drawbacks of that version and approach are:
1. slow and limited in options/modifiablity/compatibility with non-intended uses
2. can't run on multiple processors with MPI, only serial on a single machine (making it even slower)
3. couldn't change the total number of rods during simulation, i.e. couldn't keep constant concentration of any species

On the other hand, it uses only the "official" LAMMPS features, no need to download and build from my fork of the LAMMPS project. I'm not keeping it very updated though...
