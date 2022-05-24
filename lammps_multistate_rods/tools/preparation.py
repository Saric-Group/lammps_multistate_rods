# encoding: utf-8
'''
This module contains methods to generate files with locations and rotations of rods
which can then be used to generate rods via the "create_rods" method of a
lammps_multistate_rods.Simulation class instance.

Created on 8 Jan 2019

@author: Eugen Rožić
'''
import numpy as np
import pyquaternion

def fibril(model, N, phi, theta, r0, data = None, out_path = None):
    '''
    This method generates locations and orientations of rods in a fibril that has the given
    length and global location and orientation.
    The generated per-rod data are 7-tuples of the following format:
        (x, y, z, theta, Rx, Ry, Rz)
    where theta is the angle of rotation (in radians) around a unit vector given by (Rx, Ry, Rz),
    and (x,y,z) is the insertion (center) point.
    
    model : A lammps_multistate_rods.Rod_model instance
    N : number of rods in the "fibril"
    phi : azimuth angle of the fibril (with respect to y-axis; in deg)
    theta : elevation angle of the fibril (with respect to x-y plane; in deg)
    r0 : centre of the fibril (x,y,z)
    
    data : a list to store the generated per-rod 7-tuples to (to be available to the caller)
    out_path : path/name of the output file to write the data to
    
    returns : a triplet of (min,max) coordinate pairs, one for each dimension, of
    extremal locations among all centers of the rods
    '''
    if data == None and out_path == None:
        raise Exception('At least one of "data" and "out_path" has to be given, otherwise the method '\
                        'does nothing.')
    
    out_file = None
    if out_path != None:
        out_file = open(out_path, 'w')
        out_file.write('monomers: {:d}\n\n'.format(N))
    
    if len(r0) == 3:
        r0 = np.array(r0)
    else:
        raise Exception("Center of the fibril has to be a 3-vector!")

    rod_radius = model.rod_radius
    R_z = pyquaternion.Quaternion(axis=[0,0,1], degrees=phi)
    R_x = pyquaternion.Quaternion(axis=[1,0,0], degrees=theta)
    if theta > 0:
        R_x_inv = pyquaternion.Quaternion(axis=[1,0,0], degrees=theta-180)
    else:
        R_x_inv = pyquaternion.Quaternion(axis=[1,0,0], degrees=theta+180)

    # correct composite rotation is to first rotate around z then around x', which is equivalent to
    # rotations first around x then around z for the same angles
    R_tot = R_z * R_x
    R_tot_inv = R_z * R_x_inv

    loc = [None]*N # list of position 3-vectors
    rot = [None]*N # list of tuples of a position 3-vector and angle
    mins = [np.infty]*3
    maxs = [-np.infty]*3

    for i in range(N):
        if i % 2 == 0:
            loc[i] = np.array([0, (i-N/2)*rod_radius, -rod_radius+0.50])#+0.27692])
            loc[i] = R_tot.rotate(loc[i]) + r0
            if R_tot.degrees == 0.0:
                rot[i] = (R_tot.degrees, [1.0, 0.0, 0.0])
            else:
                rot[i] = (R_tot.degrees, R_tot.axis)
        else:
            loc[i] = np.array([0, (i-N/2)*rod_radius, +rod_radius-0.50])#-0.27692])
            loc[i] = R_tot.rotate(loc[i]) + r0
            if R_tot_inv.degrees == 0.0:
                rot[i] = (R_tot_inv.degrees, [1.0, 0.0, 0.0])
            else:
                rot[i] = (R_tot_inv.degrees, R_tot_inv.axis)
        
        for j in range(3):
            if loc[i][j] > maxs[j]:
                maxs[j] = loc[i][j]
            if loc[i][j] < mins[j]:
                mins[j] = loc[i][j]
        
        rod_data = (loc[i][0], loc[i][1], loc[i][2], rot[i][0],
                    rot[i][1][0], rot[i][1][1], rot[i][1][2])
        if data != None:
            data.append(rod_data)
        if out_file != None:
            out_file.write('{:.2f} {:.2f} {:.2f} {:.2f} {:.3f} {:.3f} {:.3f}\n'.format(*rod_data))
    
    if out_file != None:
        out_file.write('\n')
        out_file.close()
    
    return zip(mins, maxs)

def single(r0, phi, theta, out_path = None):
    '''
    This method generates the location and orientation of a single rod (for LAMMPS).
    The generated data is a 7-tuple of the following format:
        (x, y, z, theta, Rx, Ry, Rz)
    where theta is the angle of rotation (in radians) around a unit vector given by (Rx, Ry, Rz),
    and (x,y,z) is the insertion (center) point.
    
    r0 : centre of the rod (x,y,z)
    phi : azimuth angle (with respect to x-axis; in deg)
    theta : elevation angle (with respect to x-y plane; in deg)
    
    out_path : path/name of the output file to write the data to
    
    returns : the rod location and orientation 7-tuple 
    '''    
    if len(r0) == 3:
        r0 = np.array(r0)
    else:
        raise Exception('Center of the rod has to be a 3-vector!')
        
    R_z = pyquaternion.Quaternion(axis = [0,0,1], degrees = phi)
    R_x = pyquaternion.Quaternion(axis = [1,0,0], degrees = theta)
    R_tot = R_z * R_x
    rot = R_tot.get_axis(undefined = [0.0, 0.0, 1.0])
    
    rod_data = (r0[0], r0[1], r0[2], R_tot.degrees, rot[0], rot[1], rot[2])
    
    if out_path != None:
        out_file = open(out_path, 'w')
        out_file.write('monomers: 1\n\n')
        out_file.write('{:.2f} {:.2f} {:.2f} {:.2f} {:.3f} {:.3f} {:.3f}\n'.format(*rod_data))
        out_file.write('\n')
        out_file.close()
    
    return rod_data
