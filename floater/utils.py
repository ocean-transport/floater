import numpy as np
# this is PyTables 2.4.0
import tables
# 3.0 versions change the function naming conventions
# http://pytables.github.io/MIGRATING_TO_3.x.html
import os
import fnmatch
import sys

def floats_to_tables(float_dir, output_fname,
                     float_file_prefix='float_trajectories',
                     fltBufDim = 14,
                     float_dtype = np.dtype('>f4'),
                     use_memmap=True,
                     quiet=False):
    """Translate an MITgcm float output file into pytables HDF format."""

    myfiles = []
    for file in os.listdir(float_dir):
        if fnmatch.fnmatch(file, '%s.*.data' % float_file_prefix):
            myfiles.append(file)

    if len(myfiles)==0:
        raise RuntimeError('No float files found in %s' % float_dir)

    assert fltBufDim in [8, 13, 14] # these are the only shapes I know

    flds = ['npart', 'time', 'x', 'y', 'z', 'i', 'j', 'k']
    if fltBufDim >= 8:
        flds += ['p', 'u', 'v', 't', 's']
    if fltBufDim >= 14:
        flds += ['vort']

    # lagrangian float
    class LFloat(tables.IsDescription):
        npart   = tables.UInt16Col()   # float id number, starts at 1
        time    = tables.Float32Col()  # time of the datapoint
        x       = tables.Float32Col()  # x position
        y       = tables.Float32Col()  # y position
        z       = tables.Float32Col()  # z position
        i       = tables.Float32Col()  # x index
        j       = tables.Float32Col()  # x index
        k       = tables.Float32Col()  # z index
        if fltBufDim >= 8:
            p   = tables.Float32Col()  # pressure
            u   = tables.Float32Col()  # zonal velocity
            v   = tables.Float32Col()  # meridional velocity
            t   = tables.Float32Col()  # temperature
            s   = tables.Float32Col()  # salinity
        if fltBufDim >= 14:
            vort= tables.Float32Col()  # vorticity
        # for keeping track of processor id
        nproc = tables.Float32Col()
        
    # set suffix    
    if output_fname[-3:] != '.h5':
        output_fname += '.h5'
    
    #h5file = tables.openFile(output_fname,
    with tables.openFile(output_fname,
                    mode='w', title='MITgcm Float Data') as h5file:
        group = h5file.createGroup("/", 'floats', 'Float Data')
        table = h5file.createTable(group, 'trajectories', LFloat, "Float Trajectories")
            
        for nproc, input_fname in enumerate(myfiles):
        
            fname = os.path.join(float_dir, input_fname)
        
            if use_memmap:
                traj = np.memmap(fname, dtype=float_dtype, mode='r')
            else:
                traj = np.fromfile(fname, dtype=float_dtype)
                         
            Nrecs = traj.shape[0]/fltBufDim
            traj.shape = (Nrecs, fltBufDim)
        
            lfloat = table.row
            # skip first record
            for n in range(1,Nrecs):
                
                status = 'Processing file %s (% 3d/% 3d): % 5.2f%%' % (
                            input_fname, nproc+1, len(myfiles), 100*n/float(Nrecs))
                if not quiet:
                    sys.stdout.write("\r" + status)
                    sys.stdout.flush()
                
                for nfld, k in enumerate(flds):
                    lfloat[k] = traj[n,nfld]
                # processor id
                lfloat['nproc'] = nproc
                lfloat.append()

        table.cols.npart.createIndex()
        table.cols.time.createIndex()
        table.flush()
    #h5file.close()

    