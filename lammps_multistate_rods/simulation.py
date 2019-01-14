# encoding: utf-8
'''
This module holds just that same-name class, refer to its description.

Created on 22 Mar 2018

@author: Eugen Rožić
'''
import os
from math import exp, sqrt, pi
import random

from lammps import PyLammps
import rod, model

class Simulation(object):
    '''
    This class is a wrapper for a single simulation of multi-state rods in LAMMPS. It holds
    all the necessary information and its state should reflect the true state of the
    LAMMPS simulation.
    All changes to the particles created by this library (the rods) should go through methods
    of this class, otherwise inconsistent states will probably be reached.
    '''
    
    rods_group = "rods" # name of the LAMMPS group which holds all particles of all the rods
    active_beads_group = "active_rod_beads" # name of the LAMMPS group which holds the active beads of all rods
    cluster_compute = "rod_cluster" # name of the LAMMPS compute that gives cluster labels to active beads
    
    def __init__(self, py_lmp, model, seed, output_dir, log_path=None, clusters=3.0):
        '''
        Generates the model files and initiates the LAMMPS log.
        
        py_lmp : pointer to a PyLammps object
        
        model : pointer to a Model object
        
        seed : a seed to be used for all random number generators
        
        output_dir : where all files that are created (.mol, LAMMPS log, dumps etc.) will be put
        
        log_path : LAMMPS will be set to write to this log file and the methods of this library
        will log useful information and hide a lot of useless ones (like the voluminous output of
        "state_change_MC"). If not given everything will be logged (danger of huge log
        files) to a file specified before (or the "log.lammps" default if not specified)
        
        clusters = <number> : the distance between bead centers that qualifies two beads (and
        consequently whole rods) to be in the same cluster. If it is > 0.0 a LAMMPS group will
        be defined that will contain all active beads (those that have non-vx interactions; this
        slows down the simulation) and a compute that gives all active beads a label corresponding
        to the label for a cluster of rods (which can be dumped with "c_rod_cluster") will be made
        available to the user through the "cluster_compute" class variable 
        '''       
        if not isinstance(py_lmp, PyLammps):
            raise Exception("py_lmp has to be an instance of lammps.PyLammps!")
        self.py_lmp = py_lmp
        
        self.model = model
        self.output_dir = output_dir
        self.model.generate_mol_files(output_dir)
    
        self.log_path = log_path
        if log_path != None:
            py_lmp.log('"'+log_path+'"')
            
        # simulation properties (most of which to be set in "setup" and "create_rods")
        self.seed = seed
        self.clusters = clusters
        self.type_offset = None
        self._all_atom_types = None
        self._active_bead_types = None
        self._state_types = None
        self._rods = []
        self._nrods = 0
        self._rod_counters = [0]*model.num_states
    
    def _set_pair_coeff(self, type_1, type_2, (eps, int_type_key), sigma):
        
        int_type = self.model.int_types[int_type_key]
        
        if int_type[0] == 'lj/cut':
            self.py_lmp.pair_coeff(type_1, type_2, int_type[0], eps,
                                   sigma/pow(2,1./6), sigma+int_type[1])
        elif int_type[0] == 'cosine/squared':
            self.py_lmp.pair_coeff(type_1, type_2, int_type[0], eps,
                                   sigma, sigma+int_type[1],
                                   int_type[2] if len(int_type)==3 else "")
        elif int_type[0] == 'nm/cut':
            self.py_lmp.pair_coeff(type_1, type_2, int_type[0], eps,
                                   sigma, int_type[1], int_type[2], sigma+int_type[3])
        elif int_type[0] == 'morse':
            self.py_lmp.pair_coeff(type_1, type_2, int_type[0], eps,
                                   int_type[1], sigma, sigma+int_type[2])
        elif int_type[0] == 'gauss/cut':
            H = -eps*sqrt(2*pi)*int_type[1]
            self.py_lmp.pair_coeff(type_1, type_2, int_type[0], H,
                                   sigma, int_type[1], sigma+int_type[2])
        else:
            raise Exception('Unknown/invalid int_type parameter: '+ str(int_type))
            
    def setup(self, region_ID, atom_style=None, type_offset=0, extra_pair_styles=[], overlay=False,
              bond_offset=0, extra_bond_styles=[], **kwargs):
        '''
        This method sets-up all the styles (atom, pair, bond), the simulation box and all the
        data need to simulate the rods (mass, coeffs, etc.).
        
        region_ID : the region ID to use in the "create_box" command
        
        atom_style : a string given verbatim to the LAMMPS "atom_style" command; if not given
        "atom_style molecular" is used
        
        type_offset : the number of particle types that will be used for non-rod particles
        
        extra_pair_styles : an iterable consisted of pair style names and parameters needed to
        define them in LAMMPS, e.g. ("lj/cut", 3.0, "lj/long/dipole/long", "cut", "long", 5.0, ...)
        WARNING: don't use the same style as given in the config file!
        NOTE: the styles from the config file are automatically set with "shift yes"
        
        overlay : if True the "hybrid/overlay" pair_style will be used, instead of the default "hybrid"
        
        bond_offset : the number of bond types that will be used for non-rod particles
        
        extra_bond_styles : an iterable consisted of bond style names and parameters needed to
        define them in LAMMPS, e.g. ("harmonic", "fene", ...)
        NOTE: style "zero" is already defined by default
        
        kwargs : any LAMMPS "create_box" command keyword is allowed here and all of it will be given
        to the said command as "key value"
        '''
        # set instance variables
        self.type_offset = type_offset
        self._active_bead_types = ' '.join(str(t + self.type_offset)
                                           for t in self.model.active_bead_types)
        self._state_types = [ [ elem + self.type_offset 
                                for patch in state_struct for elem in patch]
                              for state_struct in self.model.state_structures]
        self.bond_offset = bond_offset
        
        # set LAMMPS styles (atom, pair, bond)
        if atom_style is None:
            atom_style = "molecular"
        self.py_lmp.atom_style(atom_style)
        
        pair_styles_cmd = ['hybrid/overlay' if overlay else 'hybrid']
        pair_styles = set([int_type[0] for int_type in self.model.int_types.values()])
        for pair_style in pair_styles:
            pair_styles_cmd.append('{:s} {:f}'.format(pair_style, self.model.global_cutoff))
        pair_styles_cmd.append(' '.join(map(str, extra_pair_styles)))
        self.py_lmp.pair_style(' '.join(pair_styles_cmd))
        for pair_style in pair_styles:
            self.py_lmp.pair_modify('pair', pair_style, 'shift yes')
        
        self.py_lmp.bond_style('hybrid', 'zero', ' '.join(map(str, extra_bond_styles)))
        
        # create region_ID (with all the parameters)
        try:
            kwargs['extra/bond/per/atom'] = int(kwargs['extra/bond/per/atom']) + 2
        except KeyError:
            kwargs['extra/bond/per/atom'] = 2
        try:
            kwargs['extra/special/per/atom'] = int(kwargs['extra/special/per/atom']) + 6
        except KeyError:
            kwargs['extra/special/per/atom'] = 6
        create_box_args = []
        for key, value in kwargs.iteritems():
            create_box_args.append('{} {}'.format(key, value))
        self.py_lmp.create_box(type_offset + self.model.max_bead_type, region_ID, "bond/types", 1 + bond_offset,
                                ' '.join(create_box_args))
        
        # load molecules from model files
        for state_name in self.model.rod_states:
            self.py_lmp.molecule(state_name, '"'+os.path.join(self.output_dir, state_name+'.mol')+'"')
            
        rod_type_range = "{:d}*{:d}".format(self.type_offset + 1, self.type_offset + self.model.max_bead_type)
        
        # set masses (interaction sites are massless, only body beads contribute to mass)
        self.py_lmp.mass(rod_type_range, self.model.rod_mass*10**-10)
        for bead_type in self.model.body_bead_types:
            self.py_lmp.mass(bead_type + self.type_offset, self.model.rod_mass/self.model.body_beads)
            
        # set interactions (initially to 0 between all pairs of types)
        self._set_pair_coeff(rod_type_range, rod_type_range, (0.0, model.vx), 1.0)
        for bead_types, eps_val in self.model.eps.iteritems():
            sigma = 0
            for bead_type in bead_types:
                if bead_type in self.model.body_bead_types:
                    sigma += self.model.rod_radius
                else:
                    for k in range(self.model.num_patches):
                        if bead_type in self.model.patch_bead_types[k]:
                            sigma += self.model.patch_bead_radii[k]
                            break
            type_1 = bead_types[0] + self.type_offset
            type_2 = bead_types[1] + self.type_offset
            self._set_pair_coeff(type_1, type_2, eps_val, sigma)
        
        self.py_lmp.bond_coeff(self.bond_offset + 1, 'zero')
        
        #create groups & set cluster tracking
        self.py_lmp.group(Simulation.rods_group, "empty")
        if self.clusters > 0.0:
            self.py_lmp.group(Simulation.active_beads_group, "empty")
            self.py_lmp.compute(Simulation.cluster_compute, Simulation.active_beads_group, "aggregate/atom",
                                 self.clusters*self.model.rod_radius)

    def create_rods(self, state_ID=0, **kwargs):
        '''
        This method creates the rods (in the specified state) and associates them with
        appropriate LAMMPS groups.
    
        The method supports specifying different ways of creating the rods by passing
        one of the following optional parameters:
        
            box = None (DEFAULT) - creates them on a defined lattice
            region = <region_ID> - creates them on a defined lattice only in the specified region
            random = (N, seed, <region_ID>) - creates them on random locations in the specified region
            file = <file_path> - creates them on locations and with rotations specified in the file;
            the file has to have the following format:
                monomers: N
                <empty line>
                <x> <y> <z> <angle> <Rx> <Ry> <Rz>
                ...(N-1 more lines like above)...
            where the <angle> should be in radians, and the Rs are components of a unit vector about
            which to rotate, and whose origin is at the insertion point.
        '''
        particle_offset = self.py_lmp.lmp.get_natoms()
        
        if "region" in kwargs.keys():
            region_ID = kwargs['region']
            self.py_lmp.create_atoms(self.type_offset, "region", region_ID,
                                     "mol", self.model.rod_states[state_ID], self.seed)
        elif "random" in kwargs.keys():
            vals = kwargs['random']
            self.py_lmp.create_atoms(self.type_offset, "random", vals[0], vals[1], vals[2],
                                     "mol", self.model.rod_states[state_ID], self.seed)
        elif "file" in kwargs.keys():
            filename = kwargs['file']
            with open(filename, 'r') as rods_file:
                N = int(rods_file.readline().split()[1])
                rods_file.readline()
                for i in range(N):
                    vals = map(float, rods_file.readline().split())
                    self.py_lmp.create_atoms(self.type_offset, "single", vals[0], vals[1], vals[2],
                                             "mol", self.model.rod_states[state_ID], self.seed,
                                             "rotate", vals[3], vals[4], vals[5], vals[6],
                                             "units box")
        else:
            self.py_lmp.create_atoms(self.type_offset, "box",
                                     "mol", self.model.rod_states[state_ID], self.seed)
            
        self._all_atom_types = self.py_lmp.lmp.gather_atoms("type", 0, 1)
        
        # create & populate LAMMPS groups (and setup cluster tracking)
        self.py_lmp.group("temp_new_rods", "id >", particle_offset)
        self.py_lmp.group(Simulation.rods_group, "union", Simulation.rods_group, "temp_new_rods")
        self.py_lmp.group("temp_new_rods", "clear")
        if self.clusters > 0.0:
            self.py_lmp.group(Simulation.active_beads_group, "type", self._active_bead_types) # contains all active beads of all rods
        
        # create rods (in Python) + supporting stuff
        rods_before = self._nrods
        new_rods = int((self.py_lmp.lmp.get_natoms() - particle_offset) / self.model.total_beads)
        for i in range(new_rods):
            rod_start_index = particle_offset + i * self.model.total_beads
            rod_atom_indices = range(rod_start_index, rod_start_index + self.model.total_beads)
            self._rods.append(rod.Rod(self, rods_before + i + 1, rod_atom_indices, state_ID))
        self._nrods = len(self._rods) # = rods_before + new_rods
        self._rod_counters[state_ID] += new_rods
    
    def set_rod_dynamics(self, ensemble = "", **kwargs):
        '''
        Sets a "rigid/<ensemble>/small" integrator for all the rods (default is just "rigid/small")
        
        Any additional LAMMPS "keyword" options (e.g. langevin, temp, iso etc.) can be passed as
        named arguments in the following form:
            keyword = (value_1, value_2, ...)
        '''
        keyword_options = ""
        for key, values in kwargs.iteritems():
            keyword_options += key
            for value in values:
                keyword_options += " " + str(value)
            keyword_options += "\t"
        
        ensemble = ensemble.strip().lower()
        fix_name = "rigid/"+ensemble+"/small" if ensemble != "" else "rigid/small"
    
        self.py_lmp.fix("rod_dynamics", Simulation.rods_group, fix_name, "molecule",
                            "mol", self.model.rod_states[0], keyword_options) 
        self.py_lmp.neigh_modify("exclude", "molecule/intra", Simulation.rods_group)
    
#        if (gcmc != None):
#        py_lmp.fix("rod_supply", base_group, "gcmc", gcmc[0], gcmc[1], 0, 0, seed, temp, gcmc[2], 0.0,
#                    "mol", rod_states[0], "rigid rod_dynamics", "region", gcmc[3], "group", master_group_name)
#        
#        py_lmp.compute_modify("thermo_temp", "dynamic/dof yes")
#        py_lmp.fix_modify("rod_dynamics", "dynamic/dof yes") #only necessary for rigid/nvt

    #########################################################################################
    ### SIMULATION TOOLS ####################################################################

    def total_pe(self):
        '''
        Returns the total (non-normalised) potential energy of the system
        '''
        return self.py_lmp.lmp.extract_compute("thermo_pe", 0, 0)

    def rods_count(self):
        '''
        Returns the overall number of rods in the simulation.
        '''
        return self._nrods

    def state_count(self, state_id):
        '''
        Returns the number of rods in the state given by ID.
        '''
        return self._rod_counters[state_id]

    def get_random_rod(self):
        '''
        returns : a randomly picked rod as a lammps_multistate_rods.rod.Rod object
        '''
        return self._rods[random.randrange(self._nrods)]

    def _try_state_change(self, rod, U_before, T):
        '''
        Tries an MC state change on the given rod. The change
        is accepted with probability equal to:
            max{ exp(- delta_U - trans_penalty), 1}
        where:
            delta_U = U_after - U_before
    
        WARNING: this method leaves LAMMPS in an inconsistent state regarding
        thermodynamic variables (i.e. the state of atoms might not correspond
        with current "pe", because when a move is rejected the energy is not
        calculated again after restoring old rod state)
    
        returns : (1, U_after) or (0, U_before)
        '''
        old_state = rod.state
        candidate_states = self.model.transitions[old_state]
        new_state, penalty = candidate_states[random.randrange(0, len(candidate_states))] # certainty a try will be made
        rod.set_state(new_state)
    
        self.py_lmp.command('run 0 post no')
    
        U_after = self.total_pe()
        accept_prob = exp((U_before - U_after - penalty) / T)
    
        if (accept_prob > 1 or random.random() < accept_prob):
            self._rod_counters[old_state] -= 1
            self._rod_counters[new_state] += 1
            self._reset_active_beads_group()
            return (1, U_after)
        else:
            rod.set_state(old_state) # revert change back
            return (0, U_before)

    def state_change_MC(self, ntries):
        '''
        Tries to make "ntries" Monte Carlo state changes on randomly selected rods that are
        presumed to be equilibrated to the simulation temperature.
        
        returns : the number of accepted moves
        '''
        if self.log_path != None:
            self.py_lmp.log('none') # don't print all the "run 0" runs
    
        U_start = U_current = self.total_pe()
        T_current = self.py_lmp.lmp.extract_compute("thermo_temp", 0, 0)
        success = 0
        for _ in range(ntries):
            (acpt, U_current) = self._try_state_change(self.get_random_rod(), U_current, T_current)
            success += acpt
    
        if self.log_path != None:
            self.py_lmp.log(self.log_path, 'append')
            self.py_lmp.command('print "state_change_MC: {:d}/{:d} (delta_U = {:f})"'.format(
                                    success, ntries, U_start - U_current))
        return success
    
    def _reset_active_beads_group(self):
        '''
        Resets the group by first clearing it and then reassigning to it.
        '''
        if self.clusters > 0.0:
            self.py_lmp.group(Simulation.active_beads_group, "clear")
            self.py_lmp.group(Simulation.active_beads_group, "type", self._active_bead_types)
