# encoding: utf-8
'''
This module holds just the same-name class, refer to its description.

Created on 22 Mar 2018

@author: Eugen Rožić
'''
import os
from math import sqrt, pi

from lammps import PyLammps

from .rod_model import vx

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
    The "set" methods are for setting relevant fixes with relevant and necessary parameters/options
    for the simulation of rods, and other methods are various convenience methods that might come
    in handy. All of this, and much more, can be done "by hand" outside of the library methods. 
    '''
    
    rods_group = "rods" # name of LAMMPS group which holds all particles of all the rods
    rods_group_var = rods_group + "_count" # name of LAMMPS var which counts the rods_group
    
    rod_dyn_fix = "rod_dynamics"
    state_trans_fix = "state_transitions"
    
    def __init__(self, py_lmp, model, temperature, seed, output_dir):
        '''
        Generates the model files and initiates the LAMMPS log.
        
        py_lmp : reference to a PyLammps object
        
        model : reference to a Model object
        
        temperature : the temperature of the simulation (for Monte Carlo fixes)
        
        seed : a seed to be used for most random number generators
        
        output_dir : where all files that are created (.mol, LAMMPS log, dumps etc.) will be put
        '''       
        if not isinstance(py_lmp, PyLammps):
            raise Exception("py_lmp has to be an instance of lammps.PyLammps!")
        self.py_lmp = py_lmp
        
        self.mpi_enabled = False
        self.mpi_rank = 0
        if py_lmp.lmp.comm != None:
            self.mpi_enabled = True
            self.mpi_rank = py_lmp.lmp.comm.Get_rank()
        
        self.model = model
        self.output_dir = output_dir
        self.trans_file = os.path.join(output_dir, 'states.trans')
        # generate files only on the base/single processor
        if self.mpi_rank == 0:
            self.model.generate_mol_files(output_dir)
            model.generate_trans_file(self.trans_file)
            
        # simulation properties (most of which to be set in "setup" and "create_rods")
        self.temp = temperature
        self.seed = seed
        self.type_offset = None
        self.bond_offset = None
        self.pair_styles = None
        
        self.state_groups = [None] * model.num_states
        self.state_group_vars = [None] * model.num_states
        for i in range(model.num_states):
            self.state_groups[i] = "{}_{}".format(model.rod_states[i], Simulation.rods_group)
            self.state_group_vars[i] = "{}_count".format(self.state_groups[i])
        
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
    
    def deactivate_state(self, state_ID, vx_eps = 1.0):
        '''
        This method effectively turns the rods in this state into passive solid rods, with regard
        to other rods (the user has to take care of interactions with any other non-rod particles).
        
        This is achieved by removing all interactions between active bead types of this rod state
        and any other active rod beads, with the exception of body bead types whose interaction is
        set to volume-exclusion with other body bead types (of strength vx_eps).
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
            
    def setup(self, region_ID, atom_style = 'molecular', type_offset = 0, extra_pair_styles = [],
              overlay = False, bond_offset = 0, extra_bond_styles = [], opt = []):
        '''
        This method sets-up all the styles (atom, pair, bond), the simulation box and all the
        data needed to simulate the rods (mass, coeffs, etc.). It is essentially a proxy for the
        "box_create" LAMMPS command.
        It also sets up groups and variables: one for all rods (rods_group, rods_group_var) and one
        for each rod state (state_groups, state_group_vars). The variables count the number of rods
        in each corresponding group.
        
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
        
        opt : a list containing additional, non-required command parameters that will be passed verbatim
        as a single space-separated string to the LAMMPS "box_create" command (e.g. "angle/types",
        "extra/???/per/atom" etc.) 
        '''
        # set instance variables
        self.type_offset = type_offset
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
        create_box_args = ' '.join(map(str, opt)).split()
        max_bonds = self.model.num_patches + 1
        try:
            ebpa_index = create_box_args.index('extra/bond/per/atom')
        except ValueError:
            create_box_args.extend(['extra/bond/per/atom', str(max_bonds)])
        else:
            ebpa = int(create_box_args[ebpa_index + 1])
            if ebpa < max_bonds:
                create_box_args[ebpa_index + 1] = str(max_bonds)
        try:
            espa_index = create_box_args.index('extra/special/per/atom')
        except ValueError:
            create_box_args.extend(['extra/special/per/atom', str(3 * max_bonds)])
        else:
            espa = int(create_box_args[espa_index + 1])
            if espa < 3 * max_bonds:
                create_box_args[espa_index + 1] = str(3 * max_bonds)
        
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
        
        # create LAMMPS groups and variables for rods (general and each state)
        self.py_lmp.group(Simulation.rods_group, "empty")
        self.py_lmp.variable(Simulation.rods_group_var, "equal",
                             "count({})/{}".format(Simulation.rods_group,
                                                   self.model.total_beads))
        for i in range(self.model.num_states):
            self.py_lmp.group(self.state_groups[i], "empty")
            self.py_lmp.variable(self.state_group_vars[i], "equal",
                                 "count({})/{}".format(self.state_groups[i],
                                                       self.model.total_beads))        
        
    def get_min_rod_type(self):
        return self.type_offset+1
    
    def get_max_rod_type(self):
        return self.type_offset + max(self.model.all_bead_types)

    def create_rods(self, state_ID = 0, **kwargs):
        '''
        This method creates the rods (in the specified state) and associates them with
        appropriate LAMMPS groups.
    
        The method supports specifying different ways of creating the rods by passing
        one of the following optional parameters:
        
            box = (...) (DEFAULT) - creates them on a defined lattice
            region = (<region_ID>, ...) - creates them on a defined lattice only in the specified region
            random = (N, seed, <region_ID>, ...) - creates them on random locations in the specified region
            exact = (per-rod-list, ...) - creates them on locations and with rotations specified in a per-rod-list;
                each element of the list has to be a 7-tuple with the following contents:
                    (x, y, z, theta, Rx, Ry, Rz)
                where theta is the angle of rotation (in radians) around a unit vector given by (Rx, Ry, Rz),
                and (x,y,z) is the insertion point.
        
        The "..." stands for extra parameters that will be passed to "create_atoms" verbatim as a single
        space-separated string.
        '''
        all_atom_ids = self.py_lmp.lmp.gather_atoms_concat("id", 0, 1)
        if len(all_atom_ids) > 0:
            id_offset = max(all_atom_ids)
        else:
            id_offset = 0
    
        state_template = self.model.rod_states[state_ID]
        
        if len(kwargs) == 0: #default
            self.py_lmp.create_atoms(0, "box",
                                     "mol", state_template, self.seed)
        elif "box" in kwargs.keys():
            params = kwargs['box'] if kwargs['box'] else []
            self.py_lmp.create_atoms(0, "box",
                                     "mol", state_template, self.seed,
                                     ' '.join(map(str, params)))
        elif "region" in kwargs.keys():
            params = kwargs['region']
            if len(params) < 1:
                raise Exception('The "region" option has to come with at least 1 argument (the region ID)!')
            self.py_lmp.create_atoms(0, "region", params[0],
                                     "mol", state_template, self.seed,
                                     ' '.join(map(str, params[1:])))
        elif "random" in kwargs.keys():
            params = kwargs['random']
            if len(params) < 3:
                raise Exception('The "random" option has to come with at least 3 arguments (N, seed and a region ID)!')
            self.py_lmp.create_atoms(0, "random", params[0], params[1], params[2],
                                     "mol", state_template, self.seed,
                                     ' '.join(map(str, params[3:])))
        elif "exact" in kwargs.keys():
            params = kwargs['exact']
            if len(params) < 1:
                raise Exception('The "exact" option has to come with at least 1 argument (a per-rod-list)!')
            for vals in params[0]:
                self.py_lmp.create_atoms(0, "single", vals[0], vals[1], vals[2],
                                         "mol", state_template, self.seed,
                                         "rotate", vals[3], vals[4], vals[5], vals[6],
                                         "units box", ' '.join(map(str, params[1:])))
        else:
            raise Exception('Unsupported option(s) ({:s}) passed to "create_rods"!'.format(kwargs))
        
        # update LAMMPS groups for rods
        self.py_lmp.group("temp_new_rods", "id >", id_offset)
        self.py_lmp.group(Simulation.rods_group, "union", Simulation.rods_group, "temp_new_rods")
        self.py_lmp.group(self.state_groups[state_ID], "union", self.state_groups[state_ID], "temp_new_rods")
        self.py_lmp.group("temp_new_rods", "clear")
    
    def rods_count(self):
        '''
        Returns the number of rods in the simulation (via rods group atom count)
        '''
        return int(self.py_lmp.eval('v_' + Simulation.rods_group_var))

    def state_count(self, state_ID):
        '''
        Returns the number of rods of the given state in the simulation (via state group atom count)
        '''
        return int(self.py_lmp.eval('v_' + self.state_group_vars[state_ID]))
    
    def set_rod_dynamics(self, ensemble = "", opt = []):
        '''
        Sets a "rigid/<ensemble>/small" integrator for all the rods (default is just "rigid/small").
        It also does some comm (ghost) and neighbor (intramolecular exclusion) modifications needed
        for the simulation to function properly with rods as rigid small molecules.
        
        opt : a list containing additional, non-required fix parameters that will be passed verbatim
        as a single space-separated string to the LAMMPS "fix" command (e.g. langevin, temp, iso, ...)
        
        returns: the fix ID (for convenience; available at Simulation.rod_dyn_fix)
        '''
        fix_opt_args = ' '.join(map(str, opt))
        
        ensemble = ensemble.strip().lower()
        fix_type = "rigid/"+ensemble+"/small" if ensemble != "" else "rigid/small"
    
        output = self.py_lmp.fix(Simulation.rod_dyn_fix, Simulation.rods_group, fix_type,
                                 "molecule", fix_opt_args)
        # TODO
        # currently fix rigid/small supports only one mol template for later creation of
        # molecules (with e.g. gcmc) so I left it to be defined in "opt" by hand;
        # if multiple templates would be supported they could all be specified here by
        # adding something like:
        # " ".join(["mol "+state for state in self.model.rod_states]),
        
        try:
            for line in output:
                if "max distance" in line:
                    rigid_body_extent = float(line.split("=")[0])
                    break
            # will be ignored if < neighbor cutoff (max pair cutoff + skin) so OK...
            self.py_lmp.comm_modify("cutoff", (1+10**-8)*rigid_body_extent)
            # the (1+10^-8) factor is needed because rigid/small outputs "maxextent" to the 8th decimal place
        except:
            if self.mpi_rank == 0:
                print("WARNING: LAMMPS output to screen probably suppressed; needs to be enabled "\
                "in order for 'lammps_multistate_rods' library to function properly")
        
        self.py_lmp.neigh_modify("exclude", "molecule/intra", Simulation.rods_group)
        
        return Simulation.rod_dyn_fix
    
    def unset_rod_dynamics(self):
        '''
        Unsets (unfix) the integrator set by the "set_rod_dynamics" method (the fix ID is stored in
        Simulation.rod_dyn_fix for manual manipulation).
        '''
        self.py_lmp.unfix(Simulation.rod_dyn_fix)
        
    def set_state_transitions(self, every, attempts, opt = []):
        '''
        Sets the "change/state" fix that does Monte Carlo changing of states of rods.
        
        every : how often will the fix be called (every that many steps)
        
        attempts : how many MC attempts will be made per call of the fix
        
        temperature : the ideal temperature of the simulation (needed for energy penalties which
        are assumed to be in kT units)
        
        opt : a list containing additional, non-required fix parameters that will be passed verbatim
        as a single space-separated string to the LAMMPS "fix" command (e.g. antisym, full_energy, pe, ...)
        
        returns: the fix ID (for convenience; available at Simulation.state_trans_fix)
        '''
        fix_opt_args = ' '.join(map(str, opt))
    
        self.py_lmp.fix(Simulation.state_trans_fix, Simulation.rods_group, "change/state",
                        every, attempts, self.seed, self.temp,
                        "mols", " ".join([state for state in self.model.rod_states]),
                        "trans_pens", self.trans_file,
                        "groups", " ".join([group for group in self.state_groups]),
                        fix_opt_args)
        
        return Simulation.state_trans_fix
    
    def unset_state_transitions(self):
        '''
        Unsets (unfix) the "change/state" fix set by the "set_state_transitions" method (the fix ID
        is stored in Simulation.state_trans_fix for manual manipulation).
        '''
        self.py_lmp.unfix(Simulation.state_trans_fix)
    
    def set_state_concentration(self, state_ID, concentration, every, attempts, opt = []):
        '''
        Sets the "gcmc" fix for the given state to keep concentration approx constant (no
        MC moves, only insertion/deletion of rods in the given state).
        It does so using the "pressure" keyword and the relation: P = c*kB*T (ignoring fugacity,
        which can be taken into account "by hand")
        It also modifies the "thermo_temp" compute for changing degrees of freedom (dynamic/dof);
        this has to be done manually for all other temperature computes over rod atoms, if they
        exist (for example the internal temperature compute of an "nvt" fix, if used).
        
        state_ID : the ID of the state to fix the concentration of
        
        concentration : desired concentration (in the units that were set at LAMMPS creation)
        
        every : how often will the fix be called (every that many steps)
        
        attempts : how many MC attempts will be made per call of the fix
        
        opt : a list containing additional, non-required fix parameters that will be passed verbatim
        as a single space-separated string to the LAMMPS "fix" command (e.g. region, full_energy, 
        overlap_cutoff, group, ...)
        
        returns: the fix ID (for convenience; available at self.gcmcs[state_ID])
        '''
        fix_name = "gcmc_{}".format(self.model.rod_states[state_ID])
        
        if not hasattr(self, 'gcmcs'):
            self.gcmcs = [None] * self.model.num_states
        
        self.gcmcs[state_ID] = fix_name
        
        fix_opt_args = ' '.join(map(str, opt))
        
        kB = self.py_lmp.lmp.extract_global("boltz")
        pressure = concentration * kB * self.temp
        self.py_lmp.fix(fix_name, self.state_groups[state_ID], "gcmc",
                        every, attempts, 0, 0, self.seed, self.temp, 0, 0, 
                        "mol", self.model.rod_states[state_ID],
                        "rigid", Simulation.rod_dyn_fix,
                        "pressure", pressure,
                        "group", Simulation.rods_group,
                        fix_opt_args)
        
        self.py_lmp.compute_modify("thermo_temp", "dynamic/dof", "yes")
        #TODO if rod dynamics is nvt or npt (has own thermo compute) then a fix_modify
        # is necessary also...
        
        return fix_name
    
    def unset_state_concentration(self, state_ID):
        '''
        Unsets (unfix) the "gcmc" fix set by the "set_state_concentration" method (the fix ID
        is stored in self.gcmcs[state_ID] for manual manipulation).
        '''
        self.py_lmp.unfix(self.gcmcs[state_ID])
        # will fail if previously not "set" because gcmcs won't exist - but that is OK, it should fail
