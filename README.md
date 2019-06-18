# The _LAMMPS multistate rods_ library

This is a Python library for running LAMMPS simulations of rigid rod-like structures that can exist in multiple (internal) states that can change during simulation. Rods in different states have the same geometry but (can) differ in interactions with other rods and other particles/structures in the system.

The simulations are hybrid MD-MC simulations where all of the moving of both rods and all other particles and structures is done through standrad LAMMPS commands (MD) and the changes in the internal states of the rods are done through the library as MC moves in between LAMMPS runs (usually in batches).

## Overview

This library is written in Python 2.7 and uses LAMMPS's Python library interface to communicate with an instance of LAMMPS.

The library is contained within the **lammps_multistate_rods** directory and is consisted of the 3 main classes: `Rod_model`, `Rod` and `Simulation`, and some potentially useful tools.

It also comes with examples of usage in the **examples** directory which contains examples of model configuration files (.cfg) and a console application that uses the library along with an example run configuration file for that application.

The **tools** directory contains a console application that calculates and makes interactive figures of overall interactions between rods and particles defined in a model configuration file (.cfg), and the **docs** directory contains some documentation on the possibilities the user has in modelling a rod model and using this library (TODO).

## Usage

In order to use the library one must first define the rod model configuration file. This is the input to a new `Rod_model` object which basically parses the config file, infers and/or gives default values to what is not explicitly written and keeps all that data for use by the user and a `Simulation` object. The details on the options and possibilities in defining a model are given in the *Modelling a model* pdf file in the **docs** directory (or will be in due time).

Once a configuration file is constructed a Python program has to be written which will use the LAMMPS Python library interface (e.g. `import lammps`) to start LAMMPS (make an instance) and send commands to it. This is exactly the same as writting a LAMMPS input script and all of LAMMPS stands at the user's disposal. This library can then be imported into this program and an instance of `Rod_model` made from the confguration file. Using this a `Simulation` object can be instantiated and used in conjunction with the standard LAMMPS library to achieve the primary function of this library. How exactly to do this is documented in the Python modules of the library as well as shown in the example application in the **examples** directory.

## Installation / setup

In order to use this library one must simply download it and enable Python to find it. This can be done in a number of ways, some of which are:
1. add the path of the *lammps_multistate_rods* directory (the one that contains the class module files) to the `PYTHONPATH` environment variable (this has to be done every time for a new environment, which can be automated);
2. simply copy/put the *lammps_multistate_rods* directory in the local search path of your Python, for example in `~/.local/lib/python2.7/site-packages` (this has to be done every time you want a new version, which can also be automated, but is dirty);
3. make a symbolic link to the *lammps_multistate_rods* directory in the local search path of you Python, for example `ln -s <download_dir>/lammps_multistate_rods/ ~/.local/lib/python2.7/site-packages/lammps_multistate_rods`

In a related repository of mine (https://github.com/Saric-Group/amyloid-simulations) this library is used and there is a *setup_scripts* directory that might be very useful to check out. It contains scripts that setup everything (virtualenv and LAMMPS) and get the most current library. In that project I use the first of the methods listed above, that can be checked in any of the *job templates* in the same-named directory.

### Requirements
The requirements for using the library are:
1. Python 2.7 with numpy and pyquaternion:
    ```
    pip install numpy pyquaternion
    ```
    * scipy and matplotlib are necessary for some tools, but not for the library itself (I think)  

2. LAMMPS built with the rigid, molecule and python modules (I also suggest mc, opt, misc and user-misc, just in case):
    ```
    cd <LAMMPS_dir>/src
    make no-all # can't harm to clean, purge and package-update beforehand...
    make yes-rigid yes-molecule yes-python
    make -j4 serial mode=shlib LMP_INC="-DLAMMPS_EXCEPTIONS"
    make install-python #this is redundant, but again, can't harm
    ```
    * It would be best to use the **develop** branch of my fork of the LAMMPS project:  
        ```  
        git clone -b develop https://github.com/erozic/lammps.git <LAMMPS_dir>  
        ```  
        Currently it contains the `cosine/squared` pair style that is not yet in the standard LAMMPS distribution, along with the `lammps_get_pe` library function (that will never be included in standard LAMMPS) and possibly some other non-essential things. All in all, the library should work with the official LAMMPS (8 Feb 2019 last I checked), but not optimally and with reduced capabilities and I don't guarantee it and I strongly suggest using the develop branch of my LAMMPS fork.  
