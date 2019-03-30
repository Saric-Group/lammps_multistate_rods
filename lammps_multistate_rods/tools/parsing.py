# encoding: utf-8
'''
This module contains methods for parsing of LAMMPS dump files generated by using this
library.

Created on 11 Jan 2019

@author: Eugen Rožić
'''

import re
from lammps_multistate_rods import Simulation

def keyword_parse_pattern(lammps_keyword):
    '''
    Returns a regular expression string for the given LAMMPS dump keyword.
    
    Currently supported: id, type, mol, c_rod_cluster, x, y, z
    '''
    if lammps_keyword in ('id','type','mol','c_'+Simulation.cluster_compute):
        return r'(\d+)'
    elif lammps_keyword in ('x','y','z'):
        return r'([-\+\d\.eE]+)'
    else:
        print 'WARNING: Unsupported lammps_keyword ({}) in dump file!'.format(
            lammps_keyword)
        return r'(\S+)' #one or more non-whitespace characters
    
def parse_dump_file(dump_file_path):
    '''
    This method is a generator that reads a LAMMPS dump file with one or more simulation
    snapshots and yields data for one snapshot at a time.
    
    returns : a (timestep, box_bounds, data_structure, data) quadruplet per snapshot;
        "box_bounds" is a triplet of (lower, upper) bound values, "data_structure" is a
        list of LAMMPS keywords that describes the contents of each line and "data" is
        a list of dictionaries (one per line) with "data_structure" elements as keys.
    '''
    with open(dump_file_path, 'r') as dump_file:
        i = 0
        timestep = min_row = max_row = -1
        box_bounds = data_structure = data = None
        bounds_pattern = re.compile(r'([-\+\d\.eE]+) ([-\+\d\.eE]+)')
        data_pattern = None 
        for line in dump_file:
            i += 1
                    
            if i == 2:
                timestep = int(line)
            
            elif i == 4:
                atoms_to_read = int(line)
                max_row = 9 + atoms_to_read
                min_row = 10
                box_bounds = []
            
            elif i in (6,7,8):
                l_bound, r_bound = [float(match) for match in bounds_pattern.match(line).groups()]
                box_bounds.append((l_bound, r_bound))
                
            elif i == 9:
                new_data_struct = line.split()[2:]
                if data_structure == None:
                    data_structure = new_data_struct
                    data_pattern = re.compile(' '.join(map(keyword_parse_pattern, data_structure)))
                elif new_data_struct != data_structure:
                    raise Exception('ERROR (timestep {:d}): '\
                                    'Output has to be uniform throughout a dump file!'.format(timestep))
                data = []
                
            elif i >= min_row and i <= max_row:
                line_vars = {}
                for key, value in zip(data_structure, data_pattern.match(line).groups()):
                    line_vars[key] = value
                data.append(line_vars)
                if i == max_row:
                    yield (timestep, box_bounds, data_structure, data)
                    i = 0
            else:
                #print i,
                pass

def write_dump_snapshot(timestep, box_bounds, data_structure, data, output_path, append=False):
    '''
    Writes the single snapshot data to the given filepath in the proper LAMMPS dump format
    (i.e. inverse of "parse_dump_file").
    '''
    mode = 'a' if append else 'w'
    with open(output_path, mode) as out_file:
        out_file.write('ITEM: TIMESTEP\n')
        out_file.write('{:d}\n'.format(timestep))
        out_file.write('ITEM: NUMBER OF ATOMS\n')
        out_file.write('{:d}\n'.format(len(data)))
        out_file.write('ITEM: BOX BOUNDS pp pp pp\n')
        for dimension in box_bounds:
            out_file.write('{:f} {:f}\n'.format(*dimension))
        out_file.write('ITEM: ATOMS {:s}\n'.format(' '.join(data_structure)))
        for line_vars in data:
            line_parts = map(lambda key: str(line_vars[key]), data_structure)
            out_file.write('{:s}\n'.format(' '.join(line_parts)))
