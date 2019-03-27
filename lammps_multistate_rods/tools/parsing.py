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
    
    returns : a (timestep, box_bounds, data_structure, data) quadruplet;
        "box_bounds" is a triplet of (lower, upper) bound values, "data_structure" is a
        list of LAMMPS keywords that describes the contents of each line and "data" is
        a list of verbatim lines from the dump file for a single snapshot 
    '''
    with open(dump_file_path, 'r') as dump_file:
        i = 0
        timestep = min_row = max_row = -1
        box_bounds = data_structure = data = None
        for line in dump_file:
            i += 1
                    
            if i == 2:
                timestep = int(line)
            
            elif i == 4:
                atoms_to_read = int(line)
                max_row = 9 + atoms_to_read
                min_row = 10
                box_bounds = []
            
            elif i in (6,7,8) and len(box_bounds) < 3:
                l_bound, r_bound = [float(match) for match in re.compile(r'([-\+\d\.eE]+) ([-\+\d\.eE]+)').match(line).groups()]
                box_bounds.append((l_bound, r_bound))
                
            elif i == 9:
                data_structure = line.split()[2:]
                data = []
               
            elif i >= min_row and i <= max_row:
                data.append(line)
            
            if i == max_row:
                yield (timestep, box_bounds, data_structure, data)
                i = 0

def write_dump_snapshot(timestep, box_bounds, data_structure, data, output_path):
    '''
    Appends the single snapshot data to the given file (or creates a new one) in
    the proper LAMMPS dump format (i.e. inverse of "parse_dump_file").
    '''
    with open(output_path, 'a') as out_file:
        out_file.write('ITEM: TIMESTEP\n')
        out_file.write('{:d}\n'.format(timestep))
        out_file.write('ITEM: NUMBER OF ATOMS\n')
        out_file.write('{:d}\n'.format(len(data)))
        out_file.write('ITEM: BOX BOUNDS pp pp pp\n')
        for dimension in box_bounds:
            out_file.write('{:f} {:f}\n'.format(*dimension))
        out_file.write('ITEM: ATOMS {:s}\n'.format(' '.join(data_structure)))
        for line in data:
            out_file.write(line)