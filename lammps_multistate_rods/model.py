# encoding: utf-8
'''
TODO

Created on 17 Jul 2018

@author: Eugen Rožić
'''

import os, re

vx = 'vx' #constant (for labeling vol_exclusion interaction)

class Model(object):
    '''
    This class describes a model for a lammps_multistate_rods simulation, i.e. it holds all
    the information about the parameters of the rods and the simulation that are parsed from
    a config file.
    This class is meant to be instantiated and used by instances of the Simulation and Rods
    classes of this library. An instance should be available to the user through an instance
    of the simulation class.
    '''

    def __init__(self, config_file_path):
        # ROD PROPERTIES (available to set in the config file)
        rod_states = None #example: ('base_state', 'beta_state')
        num_states = None #dependent on "rod_states"
        state_structures = None #dependent on "rod_states"
            #example of elements:
            # state_structures[0] = '1111112|333333' # '1' is inert body type, '3' is inert side-patch type
            # state_structures[1] = '1111111|444444' # '2' is active body type, '4' is active side-patch type
        rod_radius = 1.0 #default
        body_bead_overlap = 1.0*rod_radius #default
        int_radius = 0.5*rod_radius #default
        int_bead_overlap = 0.0 #default
        int_bulge_out = 0.0 #default
        rod_mass = 1.0 #default
        # INTERACTION PROPERTIES (available to set in the config file)
        int_types = None #example: ('morse', 2.5/rod_radius)
        vol_exclusion = None #example: 10 (strength of bead/particle repulsion)
        global_range = 1.5*rod_radius #default (separation between bead surfaces at which interaction =0)
        eps = {} # (interaction strengths between bead types)
            #example of elements:
            # eps[(1,1)] = eps[(1,2)] = eps[(1,3)] = eps[(1,4)] = vx
            # eps[(2,3)] = eps[(3,3)] = eps[(3,4)] = vx
            # eps[(2,2)] = 5.5 # base-base interaction
            # eps[(2,4)] = 5.75 # base-beta interaction
            # eps[(4,4)] = 6.0 # beta-beta interaction
        trans_penalty = {} # (transition penalties between states)
            #example of elements:
            # trans_penalty[(0,1)] = 15.0 (base to beta state transition)
            
        with open(config_file_path,'r') as config_file:
            for line in config_file:
                line = line.strip()
                if (line.startswith('#') or line == ''):
                    continue
                parts = line.split('=')
                assign = parts[0].strip()
                expr = parts[1].strip()
                if assign == 'rod_states':
                    rod_states = eval(expr)
                    num_states = len(rod_states)
                    state_structures = ['']*num_states
                elif assign in ('rod_radius', 'body_bead_overlap', 'int_radius', 'int_bead_overlap', 'int_bulge_out',
                                'rod_mass', 'int_types', 'vol_exclusion', 'global_range'):
                    exec(line)
                elif re.compile(r'state_structures\[\d+\]').match(assign) != None:
                    exec(line)
                elif re.compile(r'eps\[\(\d+,\d+\)\]').match(assign) != None:
                    exec(line)
                elif re.compile(r'trans_penalty\[\(\d+,\d+\)\]').match(assign) != None:
                    exec(line)
                else:
                    raise Exception('ERROR: Unknown config parameter encountered (' + line + ')')
        
        self.rod_states = rod_states
        self.num_states = num_states
        self.state_structures = state_structures
        self.rod_radius = rod_radius
        self.rod_length = None
        self.body_beads = None #dependent on "state_structures"
        self.body_bead_types = None #dependent on "state_structures"
        self.body_bead_overlap = body_bead_overlap
        self.int_radius = int_radius
        self.int_sites = None #dependent on "state_structures"
        self.int_bead_types = None #dependent on "state_structures"
        self.int_bead_overlap = int_bead_overlap
        self.int_bulge_out = int_bulge_out
        self.total_beads = None #dependent on "state_structures"
        self.active_bead_types = None #dependent on "state_structures" & "eps"
        self.max_bead_type = None #dependent on "state_structures"
        self.rod_mass = rod_mass
        self.int_types = int_types
        self.vol_exclusion = vol_exclusion
        self.global_range = global_range
        self.global_cutoff = global_range + 2*rod_radius
        self.eps = eps
        self.trans_penalty = trans_penalty
        self.transitions = None #dependent on "trans_penalty";
            # a list by state_ID of lists of (state_ID, penalty) pairs for all allowed transitions
    
        self._set_dependent_params()
    
    def _set_dependent_params(self):
        '''
        Sets the global variables of the instance that are used in the rest of the module
        but are derived from the essential information given by a user, specifically:
            - information contained in the string specification of the states (+ a consistency check)
            - the anti-symmetric partners in the "trans_penalty" matrix (trans_penalty[(n,m)] = -trans_penalty[(m,n)])
            - the "transitions" list from the "trans_penalty" dictionary
        '''
        self.body_bead_types = set()
        self.int_bead_types = set()
        self.body_beads = len(self.state_structures[0].split('|')[0])
        self.int_sites = len(self.state_structures[0].split('|')[1])
        for state_struct in self.state_structures: #check all have same "form"
            temp1, temp2 = state_struct.split('|')
            if len(temp1) != self.body_beads:
                raise Exception('All states must have the same number of body beads!')
            if len(temp2) != self.int_sites:
                raise Exception('All states must have the same number of interaction sites!')
            for body_bead_type in temp1:
                self.body_bead_types.add(int(body_bead_type))
            for int_bead_type in temp2:
                self.int_bead_types.add(int(int_bead_type))
        self.total_beads = self.body_beads + self.int_sites
        self.body_bead_types = list(self.body_bead_types)
        self.int_bead_types = list(self.int_bead_types)
        self.max_bead_type = max((max(self.body_bead_types), max(self.int_bead_types)))
        self.active_bead_types = set()
        for bead_types, epsilon in self.eps.iteritems():
            if epsilon != vx:
                self.active_bead_types.update(bead_types)
        self.active_bead_types = list(self.active_bead_types)
        
        self.rod_length = self.body_beads*(2*self.rod_radius - self.body_bead_overlap) + self.body_bead_overlap 
    
        #self.int_bead_overlap = 2 - ((self.body_beads - 2)*(2 - self.body_bead_overlap)*self.rod_radius)/((self.int_sites - 1)*self.int_radius)
    
        antisym_completion = {}
        self.transitions = [[] for _ in range(self.num_states)]
        for (from_state, to_state), value in self.trans_penalty.iteritems():
            antisym_completion[(to_state, from_state)] = -value
            self.transitions[from_state].append((to_state, value))
            self.transitions[to_state].append((from_state, -value))
        self.trans_penalty.update(antisym_completion)

    def generate_mol_files(self, model_output_dir):
        '''
        Generates the <model_output_dir>/<state>.mol files, where <state> represents
        one possible state of a rod (an element of self.rod_states).
        These are important to define rods as molecules in LAMMPS.
        '''
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)
        for state in range(self.num_states):
            with open(os.path.join(model_output_dir, self.rod_states[state]+'.mol'), "w") as mol_file:
                
                mol_file.write("(AUTO-GENERATED file by the lammps_multistate_rods library, any changes will be OVERWRITTEN)\n\n")
                
                mol_file.write("{:d} atoms\n\n".format(self.total_beads))
                mol_file.write("{:d} bonds\n\n".format(self.total_beads))
                
                mol_file.write("Coords\n\n")
                for i in range(self.body_beads):
                    x = 0.0 - ((self.body_beads - 2*i - 1) / 2.)*(2*self.rod_radius - self.body_bead_overlap)
                    mol_file.write("{:2d} {:6.3f}  0.000  0.000\n".format(i+1, x))
                for i in range(self.int_sites):
                    x = 0.0 - ((self.int_sites - 2*i - 1) / 2.)*(2*self.int_radius - self.int_bead_overlap)
                    z = self.rod_radius - self.int_radius + self.int_bulge_out
                    mol_file.write("{:2d} {:6.3f}  0.000 {:6.3f}\n".format(self.body_beads+i+1, x, z))
                
                mol_file.write("\nTypes\n\n")
                for i in range(self.body_beads):
                    mol_file.write("{:2d} {:s}\n".format(i+1, self.state_structures[state][i]))
                for i in range(self.body_beads, self.total_beads):
                    mol_file.write("{:2d} {:s}\n".format(i+1, self.state_structures[state][i+1]))
                
                mol_file.write("\nBonds\n\n") # cyclic bonds...
                n = 1
                for i in range(1, self.body_beads): # ... through body beads ...
                    mol_file.write("{:2d} 1 {:2d} {:2d}\n".format(n, i, i+1))
                    n += 1
                mol_file.write("{:2d} 1 {:2d} {:2d}\n".format(n, self.body_beads, self.total_beads)) # last body - last int
                n += 1
                for i in range(self.total_beads, self.body_beads+1, -1): # ... though int sites ...
                    mol_file.write("{:2d} 1 {:2d} {:2d}\n".format(n, i, i-1))
                    n += 1
                mol_file.write("{:2d} 1 {:2d} {:2d}\n\n".format(n, self.body_beads+1, 1)) # first site - first int
    