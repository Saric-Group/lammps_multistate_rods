# encoding: utf-8
'''
This module holds just the same-name class, refer to its description.

Created on 22 Mar 2018

@author: Eugen Rožić
'''
import os
from math import exp, sqrt, pi
from random import Random

from lammps import PyLammps
from ctypes import c_int, c_double

from rod import Rod
from rod_model import vx

class Simulation(object):
    '''
    This class is a wrapper for a single simulation of multi-state rods in LAMMPS. It holds
    all the necessary information and its state should reflect the true state of the
    LAMMPS simulation.
    All changes to the particles created by this library (the rods) should go through methods
    of this class, otherwise inconsistent states will probably be reached.
    
    Using the "setup" method sets up the simulation, i.e. calls the "create_box" LAMMPS method.
    This should be done first.
    Afterwards the "create_rods" method should be called, multiple times if necessary, to create
    the initial configuration of the simulation.
    The "set_rod_dynamics" method sets the dynamics of the rods in LAMMPS, but this can also be
    done "by hand", outside of the library, without calling this method. It is just a convenience
    method.
    All the other methods, including the central one ("state_change_MC") can be called during the
    simulation as necessary, in between LAMMPS "run" commands.
    '''
    
    rods_group = "rods" # name of the LAMMPS group which holds all particles of all the rods
    rod_dyn_fix = "rod_dynamics"
    
    def __init__(self, py_lmp, model, seed, output_dir):
        '''
        Generates the model files and initiates the LAMMPS log.
        
        py_lmp : reference to a PyLammps object
        
        model : reference to a Model object
        
        seed : a seed to be used for all random number generators
        
        output_dir : where all files that are created (.mol, LAMMPS log, dumps etc.) will be put
        '''       
        if not isinstance(py_lmp, PyLammps):
            raise Exception("py_lmp has to be an instance of lammps.PyLammps!")
        self.py_lmp = py_lmp
        self.random = Random(seed)
        
        self.model = model
        self.output_dir = output_dir
        self.model.generate_mol_files(output_dir)
            
        # simulation properties (most of which to be set in "setup" and "create_rods")
        self.seed = seed
        self.type_offset = None
        self._state_types = None
        self.bond_offset = None
        self.pair_styles = None
        self._rods = []
        self._nrods = 0
        self._rod_counters = [0]*model.num_states
        
    def _set_pair_coeff(self, type_1, type_2, eps, int_type_key, sigma):
        
        if type_1 > type_2:
            temp = type_1
            type_1 = type_2
            type_2 = temp
        
        int_type_vals = self.model.int_types[int_type_key]
        if self._pair_style_type.startswith('hybrid'):
            int_type = int_type_vals[0]
        else:
            int_type = ''
        
        if int_type_vals[0] == 'lj/cut':
            self.py_lmp.pair_coeff(type_1, type_2, int_type, eps,
                                   sigma/pow(2,1./6), sigma+int_type_vals[1])
        elif int_type_vals[0] == 'cosine/squared':
            self.py_lmp.pair_coeff(type_1, type_2, int_type, eps,
                                   sigma, sigma+int_type_vals[1],
                                   int_type_vals[2] if len(int_type_vals)==3 else "")
        elif int_type_vals[0] == 'nm/cut':
            self.py_lmp.pair_coeff(type_1, type_2, int_type, eps,
                                   sigma, int_type_vals[1], int_type_vals[2], sigma+int_type_vals[3])
        elif int_type_vals[0] == 'morse':
            self.py_lmp.pair_coeff(type_1, type_2, int_type, eps,
                                   int_type_vals[1], sigma, sigma+int_type_vals[2])
        elif int_type_vals[0] == 'gauss/cut':
            H = -eps*sqrt(2*pi)*int_type_vals[1]
            self.py_lmp.pair_coeff(type_1, type_2, int_type, H,
                                   sigma, int_type_vals[1], sigma+int_type_vals[2])
        else:
            raise Exception('Unknown/invalid int_type parameter: '+ str(int_type_vals))
        
    def set_pair_coeff(self, type_1, type_2, eps, int_type_key):
        '''
        This method sets a single pair coefficient between two particle types.
        
        type_1, type_2 : rod bead types as given in the config file (not the actual, offseted LAMMPS ones)
        eps : interaction strength
        int_type_key : the identificator of the interaction type (key to "int_types" from the config file)
        '''
        
        sigma = self.model.bead_radii[type_1] + self.model.bead_radii[type_2]
        type_1 += self.type_offset
        type_2 += self.type_offset
        
        self._set_pair_coeff(type_1, type_2, eps, int_type_key, sigma)
        
    def set_config_interactions(self):
        '''
        This method sets all the pair coefficients as they are given in the config file (in the 
        "eps" dictionary).
        '''
        for (type_1, type_2), (eps, int_type) in self.model.eps.iteritems():
            self.set_pair_coeff(type_1, type_2, eps, int_type)
    
    def deactivate_state(self, state_ID, vx_eps=1.0):
        '''
        This method effectively turns the rods in this state into passive solid rods, with regard
        to other rods (the user has to take care of interactions with any other non-rod particles).
        
        This is achieved by removing all interactions between active bead types of this rod state
        and any other active rod beads, with the exception of body bead types whose interaction is
        set to volume-exclusion with other body bead types (of strength vx_eps) 
        '''
        active_filter = lambda t: t in self.model.active_bead_types
        for t1 in filter(active_filter, self.model.state_bead_types[state_ID]):
            for t2 in filter(active_filter, self.model.all_bead_types):
                if t1 in self.model.body_bead_types and t2 in self.model.body_bead_types:
                    self.set_pair_coeff(t1, t2, vx_eps, vx)
                elif (t1,t2) in self.model.eps or (t2,t1) in self.model.eps:
                    self.set_pair_coeff(t1, t2, 0.0, vx)
    
    def activate_state(self, state_ID):
        '''
        This method sets the interactions of the bead types of the specified state to the ones given in
        the configuration file (e.g. according to the contents of the "eps" matrix/dictionary).
        '''
        for (type_1, type_2), (eps, int_type) in self.model.eps.iteritems():
            if type_1 in self.model.state_bead_types[state_ID] or\
               type_2 in self.model.state_bead_types[state_ID]:
                self.set_pair_coeff(type_1, type_2, eps, int_type)
            
    def setup(self, region_ID, atom_style='molecular', type_offset=0,
              extra_pair_styles=[], overlay=False,
              bond_offset=0, extra_bond_styles=[],
              everything_else=[]):
        '''
        This method sets-up all the styles (atom, pair, bond), the simulation box and all the
        data needed to simulate the rods (mass, coeffs, etc.). It is essentially a proxy for the
        "box_create" LAMMPS command.
        
        region_ID : the region ID to use in the "create_box" command
        
        atom_style : a string given verbatim to the LAMMPS "atom_style" command
        
        type_offset : the number of particle types that will be used for non-rod particles
        
        extra_pair_styles : an iterable consisted of tuples of pair style names and parameters
        needed to define them in LAMMPS, e.g. (("lj/cut", 3.0), ("lj/long/dipole/long",
        "cut long 5.0"), ...)
        WARNING: if a styles already defined in the model .cfg file is given here, the default
        model parameters (e.g. global cutoff) will be disregarded for the ones given here,
        which might cause unexpected behaviour
        NOTE: the styles from the model .cfg file are automatically set with "shift yes"
        
        overlay : if True the "hybrid/overlay" pair_style will be used, instead of the default "hybrid"
        
        bond_offset : the number of bond types that will be used for non-rod particles
        
        extra_bond_styles : an iterable consisted of bond style names and parameters needed to
        define them in LAMMPS, e.g. ("harmonic", "fene", ...)
        NOTE: style "zero" is already defined by default
        
        everything_else : a list containing additional arguments that will be passed verbatim as a single
        space-separated string to the LAMMPS "box_create" command (e.g. "angle/types", "extra/???/per/atom" etc.)
        '''
        # set instance variables
        self.type_offset = type_offset
        self._state_types = [ (self.model.total_beads*c_int)(
            *[ elem + type_offset for patch in state_struct for elem in patch])
            for state_struct in self.model.state_structures]
        self.bond_offset = bond_offset
        
        # set LAMMPS styles (atom, pair, bond)
        self.py_lmp.atom_style(atom_style)
        
        pair_styles_cmd = [' '.join(map(str, extra_pair_style))
                           for extra_pair_style in extra_pair_styles]
        extra_pair_style_names = [extra_pair_style[0].strip()
                                  for extra_pair_style in extra_pair_styles]
        self.pair_styles = list(set([int_type[0].strip()
                                     for int_type in self.model.int_types.values()]))
        for pair_style in self.pair_styles:
            if pair_style in extra_pair_style_names:
                continue
            pair_styles_cmd.append('{:s} {:f}'.format(pair_style, self.model.global_cutoff))
        
        if len(pair_styles_cmd) == 1:
            self._pair_style_type = ''
        elif overlay:
            self._pair_style_type = 'hybrid/overlay'
        else:
            self._pair_style_type = 'hybrid'
        self.py_lmp.pair_style(self._pair_style_type, ' '.join(pair_styles_cmd))
        
        if self._pair_style_type.startswith('hybrid'):
            for pair_style in self.pair_styles:
                self.py_lmp.pair_modify('pair', pair_style, 'shift yes')
        else:
            self.py_lmp.pair_modify('shift yes')
        
            
        if len(extra_bond_styles) == 0:
            self._bond_style_type = ''
        else:
            self._bond_style_type = 'hybrid'
        
        self.py_lmp.bond_style(self._bond_style_type, 'zero', ' '.join(extra_bond_styles))
        
        # create region_ID (with all the parameters)
        create_box_args = ' '.join(map(str,everything_else)).split()
        try:
            ebpa_index = create_box_args.index('extra/bond/per/atom')
        except ValueError:
            create_box_args.extend(['extra/bond/per/atom', '2'])
        else:
            ebpa = int(create_box_args[ebpa_index+1])
            if ebpa < 2:
                create_box_args[ebpa_index+1] = '2'
        try:
            espa_index = create_box_args.index('extra/special/per/atom')
        except ValueError:
            create_box_args.extend(['extra/special/per/atom', '6'])
        else:
            espa = int(create_box_args[espa_index+1])
            if espa < 6:
                create_box_args[espa_index+1] = '6'
        
        self.py_lmp.create_box(type_offset + max(self.model.all_bead_types), region_ID,
                               "bond/types", 1 + bond_offset, ' '.join(create_box_args))
        
        # load molecules from model files
        for state_name in self.model.rod_states:
            self.py_lmp.molecule(state_name,
                                 '"'+os.path.join(self.output_dir, state_name+'.mol')+'"',
                                 "toff", type_offset, "boff", bond_offset)
            
        rod_type_range = "{:d}*{:d}".format(type_offset + 1,
                                            type_offset + max(self.model.all_bead_types))
        
        # set masses (interaction sites are massless, only body beads contribute to mass)
        self.py_lmp.mass(rod_type_range, self.model.rod_mass*10**-10)
        for bead_type in self.model.body_bead_types:
            self.py_lmp.mass(bead_type + type_offset,
                             self.model.rod_mass/self.model.num_beads[0])
            
        # set interactions initially to 0.0 between all pairs of types
        self._set_pair_coeff(rod_type_range, rod_type_range, 0.0, vx, self.model.global_cutoff)
        # this has to be done because some type pairs are not given in the config file,
        # because they just don't interact at all
        self.set_config_interactions()
        
        if self._bond_style_type == 'hybrid':
            self.py_lmp.bond_coeff(bond_offset + 1, 'zero')
        else:
            self.py_lmp.bond_coeff(bond_offset + 1)
        
        self.py_lmp.group(Simulation.rods_group, "empty")
        
    def get_min_rod_type(self):
        return self.type_offset+1
    
    def get_max_rod_type(self):
        return self.type_offset + max(self.model.all_bead_types)

    def create_rods(self, state_ID=0, **kwargs):
        '''
        This method creates the rods (in the specified state) and associates them with
        appropriate LAMMPS groups.
    
        The method supports specifying different ways of creating the rods by passing
        one of the following optional parameters:
        
            box = (...) (DEFAULT) - creates them on a defined lattice
            region = (<region_ID>, ...) - creates them on a defined lattice only in the specified region
            random = (N, seed, <region_ID>, ...) - creates them on random locations in the specified region
            file = (<file_path>, ...) - creates them on locations and with rotations specified in the file;
            the file has to have the following format:
                monomers: N
                <empty line>
                <x> <y> <z> <angle> <Rx> <Ry> <Rz>
                ...(N-1 more lines like above)...
            where the <angle> should be in radians, and the Rs are components of a unit vector about
            which to rotate, and whose origin is at the insertion point.
        
        The "..." stands for extra parameters that will be passed to "create_atoms" verbatim as a single
        space-separated string.
        '''
        all_atom_ids = self.py_lmp.lmp.gather_atoms_concat("id", 0, 1)
        if len(all_atom_ids) > 0:
            id_offset = max(all_atom_ids)
        else:
            id_offset = 0
        
        if "box" in kwargs.keys():
            params = kwargs['box']
            if not params:
                params = []
            self.py_lmp.create_atoms(0, "box",
                                     "mol", self.model.rod_states[state_ID], self.seed,
                                     ' '.join(map(str, params)))
        elif "region" in kwargs.keys():
            params = kwargs['region']
            self.py_lmp.create_atoms(0, "region", params[0],
                                     "mol", self.model.rod_states[state_ID], self.seed,
                                     ' '.join(map(str, params[1:])))
        elif "random" in kwargs.keys():
            params = kwargs['random']
            self.py_lmp.create_atoms(0, "random", params[0], params[1], params[2],
                                     "mol", self.model.rod_states[state_ID], self.seed,
                                     ' '.join(map(str, params[3:])))
        elif "file" in kwargs.keys():
            params = kwargs['file']
            with open(params[0], 'r') as rods_file:
                N = int(rods_file.readline().split()[1])
                rods_file.readline()
                for i in range(N):
                    vals = map(float, rods_file.readline().split())
                    self.py_lmp.create_atoms(0, "single", vals[0], vals[1], vals[2],
                                             "mol", self.model.rod_states[state_ID], self.seed,
                                             "rotate", vals[3], vals[4], vals[5], vals[6],
                                             "units box", ' '.join(map(str, params[1:])))
        elif len(kwargs) == 0: #default
            self.py_lmp.create_atoms(0, "box",
                                     "mol", self.model.rod_states[state_ID], self.seed)
        else:
            raise Exception('Unknown options ({:s}) passed to "create_rods"!'.format(kwargs))
        
        # create & populate LAMMPS groups (and setup cluster tracking)
        self.py_lmp.group("temp_new_rods", "id >", id_offset)
        self.py_lmp.group(Simulation.rods_group, "union", Simulation.rods_group, "temp_new_rods")
        self.py_lmp.group("temp_new_rods", "clear")
        
        # create rods (in Python) + supporting stuff
        rods_before = self._nrods
        new_rods = int((self.py_lmp.lmp.get_natoms() - id_offset) / self.model.total_beads)
        for i in range(new_rods):
            start_id = id_offset + i*self.model.total_beads + 1
            rod_bead_ids = range(start_id, start_id + self.model.total_beads)
            self._rods.append(Rod(self, rods_before + i + 1, rod_bead_ids, state_ID))
        self._nrods = len(self._rods) # = rods_before + new_rods
        self._rod_counters[state_ID] += new_rods
    
    def set_rod_dynamics(self, ensemble = "", everything_else=[]):
        '''
        Sets a "rigid/<ensemble>/small" integrator for all the rods (default is just "rigid/small")
        
        everything_else : a list containing additional arguments that will be passed verbatim as a single
        space-separated string to the LAMMPS "fix" command (e.g. langevin, temp, iso, ...)
        '''
        fix_opt_args = ' '.join(map(str,everything_else))
        
        ensemble = ensemble.strip().lower()
        fix_name = "rigid/"+ensemble+"/small" if ensemble != "" else "rigid/small"
    
        self.py_lmp.fix(Simulation.rod_dyn_fix, Simulation.rods_group, fix_name,
                        "molecule", fix_opt_args) 
        self.py_lmp.neigh_modify("exclude", "molecule/intra", Simulation.rods_group)
    
    def unset_rod_dynamics(self):
        '''
        Unsets (unfix) the integrator set by "set_rod_dynamics" (the fix ID is stored in
        Simulation.rod_dyn_fix for manual manipulation).
        '''
        self.py_lmp.unfix(Simulation.rod_dyn_fix)

    #####################################################################################
    ### SIMULATION TOOLS ################################################################

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
        return self._rods[self.random.randrange(self._nrods)]

    def _try_state_change(self, rod, U_before, T, neigh_flag):
        '''
        Tries an MC state change on the given rod. The change is accepted with
        Boltzmann factor probability at the given temperature (T). 
    
        WARNING: this method leaves LAMMPS in an inconsistent state regarding
        thermodynamic variables (i.e. the state of atoms might not correspond
        with current "pe", because when a move is rejected the energy is not
        calculated again after restoring old rod state)
    
        returns : (1, U_after) or (0, U_before)
        '''
        old_state = rod.state
        candidate_states = self.model.transitions[old_state]
        new_state, penalty = candidate_states[self.random.randrange(0, len(candidate_states))] # certainty a try will be made
        rod.set_state(new_state)
        
        try:
            self.py_lmp.lmp.lib.lammps_get_pe.restype = c_double
            U_after = self.py_lmp.lmp.lib.lammps_get_pe(self.py_lmp.lmp.lmp, neigh_flag)
        except:
            # use these if library has no "lammps_get_pe" method
            print 'WARNING: LAMMPS library has no "lammps_get_pe" method! Using'\
                  ' the (much) less efficient "run 0 post no"...'
            self.py_lmp.command('run 0 post no')
            U_after = self.py_lmp.lmp.extract_compute("thermo_pe", 0, 0)
        
        accept_prob = exp((U_before - U_after - penalty) / T)
        
        if (accept_prob > 1 or self.random.random() < accept_prob):
            self._rod_counters[old_state] -= 1
            self._rod_counters[new_state] += 1
            return (1, U_after)
        else:
            rod.set_state(old_state) # revert change back
            return (0, U_before)

    def state_change_MC(self, ntries, optimise=None):
        '''
        Tries to make "ntries" Monte Carlo state changes on randomly selected rods that are
        presumed to be equilibrated to the simulation temperature.
        
        optimise : if None (default) optimisation is used (no neighbour list
        rebuild) only if the model uses a single LAMMPS pair style, otherwise this
        optimisation can be forced (True) or ignored (False).
        NOTE: using optimisation while corresponding beads in different states have
        a different style of pair interaction MAY, and most probably WILL, lead to
        wrong energy calculations!
        
        returns : the number of accepted moves
        '''
        if optimise is True:
            neigh_flag = 0
        elif optimise is False:
            neigh_flag = 1
        elif optimise is None:
            if len(self.pair_styles) > 1:
                neigh_flag = 1
            else:
                neigh_flag = 0
        else:
            raise TypeError('optimise ({}) can only be None, True or False!'.format(optimise))
        
        U_start = U_current = self.py_lmp.lmp.extract_compute("thermo_pe", 0, 0)
        T_current = self.py_lmp.lmp.extract_compute("thermo_temp", 0, 0)
        success = 0
        for _ in range(ntries):
            rand_rod = self.get_random_rod()
            (acpt, U_current) = self._try_state_change(rand_rod, U_current, T_current,
                                                       neigh_flag)
            success += acpt
    
        self.py_lmp.command('print "state_change_MC: {:d}/{:d} (delta_U = {:f})"'.format(
                            success, ntries, U_start - U_current))
        return success
