# encoding: utf-8
'''
Created on 22 Mar 2018

@author: Eugen Rožić
'''

from model import set_model_params, generate_model # these can be used without creating a Lammps instance, to generate the .mol files

from model import init, setup_simulation, set_dynamics

from model import Rod, master_group_name, group_name, nrods, state_count, get_random_rod

from model import count_var

from model import try_conformation_change, conformation_Monte_Carlo


