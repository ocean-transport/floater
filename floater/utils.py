import numpy as np
import tables
import os
import fnmatch
import sys
from . import input

def _maybe_add_h5_suffix(fname):
    if fname[-3:] != '.h5':
        fname += '.h5'
    return fname

def floats_to_tables(float_dir, output_fname,
                     float_file_prefix='float_trajectories',
                     fltBufDim = 14,
                     float_dtype = np.dtype('>f4'),
                     use_memmap=True,
                     progress=False,
		             write_blocksize_mb=64,
                     read_blocksize_mb=64,
                     max_write_blocks=np.inf):
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

    # figure out the blocksize in floats based on
    bytes_per_float = fltBufDim * float_dtype.itemsize
    blocksize_write = int(write_blocksize_mb * 1e6 / bytes_per_float)
    blocksize_read = int(read_blocksize_mb * 1e6 / bytes_per_float)
    count = 0
    nblocks = 0

    # lagrangian float
    class LFloat(tables.IsDescription):
        npart   = tables.Int32Col(pos=1)   # float id number, starts at 1
        #npart   = tables.Float32Col(pos=1)   # float id number, starts at 1
        time    = tables.Float32Col(pos=2)  # time of the datapoint
        x       = tables.Float32Col(pos=3)  # x position
        y       = tables.Float32Col(pos=4)  # y position
        z       = tables.Float32Col(pos=5)  # z position
        i       = tables.Float32Col(pos=6)  # x index
        j       = tables.Float32Col(pos=7)  # x index
        k       = tables.Float32Col(pos=8)  # z index
        if fltBufDim >= 8:
            p   = tables.Float32Col(pos=9)  # pressure
            u   = tables.Float32Col(pos=10)  # zonal velocity
            v   = tables.Float32Col(pos=11)  # meridional velocity
            t   = tables.Float32Col(pos=12)  # temperature
            s   = tables.Float32Col(pos=13)  # salinity
        if fltBufDim >= 14:
            vort= tables.Float32Col(pos=14)  # vorticity
        # for keeping track of processor id
        #nproc = tables.Float32Col(pos=fltBufDim+1)

    # for reading data
    #rec_dtype = np.dtype((float_dtype, fltBufDim))
    rec_dtype = np.dtype([ (k, float_dtype) for k in flds ])

    # need to convert to this for writing data
    new_dtype = tables.description.dtype_from_descr(LFloat)

    # set suffix
    output_fname = _maybe_add_h5_suffix(output_fname)

    # need to figure out the number of expected rows
    # do this by looking at the size of files
    total_bytes = 0
    for fname in [os.path.join(float_dir, fn) for fn in myfiles]:
        # size in bytes
        total_bytes += os.path.getsize(fname)
    expectedrows = total_bytes / bytes_per_float

    count = 0

    #h5file = tables.openFile(output_fname,
    with tables.openFile(output_fname,
                    mode='w', title='MITgcm Float Data') as h5file:
        group = h5file.createGroup("/", 'floats', 'Float Data')
        table = h5file.createTable(group, 'trajectories', LFloat, "Float Trajectories",
                                    expectedrows=expectedrows)

        for nproc, input_fname in enumerate(myfiles):

            fname = os.path.join(float_dir, input_fname)
            Nrecs_file = os.path.getsize(fname) / bytes_per_float - 1

            with open(fname, 'rb') as f:

                header = np.fromfile(f, dtype=rec_dtype, count=1)

                # loop and read data in blocks
                nreadblock = 0
                while True:

                    traj = np.fromfile(f, dtype=rec_dtype, count=blocksize_read)
                    Nrecs = len(traj)
                    if Nrecs==0:
                        # done reading
                        break

                    status = 'Processing file %s (% 3d/% 3d) block % 5d/% 5d' % (
                                input_fname, nproc+1, len(myfiles),
                                nreadblock, Nrecs_file/blocksize_read)
                    if progress:
                        sys.stdout.write("\r" + status)
                        sys.stdout.flush()

                    # append the data as a block - will this work?
                    table.append(traj.astype(new_dtype))
                    count += Nrecs
                    nreadblock += 1

                    if count >= blocksize_write:
                        table.flush()
                        count = 0
                        nblocks += 1
                        if nblocks >= max_write_blocks:
                            break

                if nblocks >= max_write_blocks:
                    break

        table.flush()
        table.cols.npart.createIndex()
        table.cols.time.createIndex()
        table.flush()


def floats_to_bcolz(input_dir, output_dir, progress=False, **kwargs):
    """Convert MITgcm float data to bcolz format.

    Paramters
    ---------
    input_dir : path
        Where to find the MITgcm output data
    output_dir : path
        Where to but the bcolz data store (equivalent to bcolz rootdir)
    kwargs :
        Extra keyword arguments to pass to floater.input_formats.MITgcmFloatData
    """
    import bcolz
    mfd = input.MITgcmFloatData(input_dir, **kwargs)
    ct = bcolz.fromiter(mfd.generator(progress=progress), dtype=mfd.rec_dtype, count=mfd.nrecs,
            mode='w', rootdir=output_dir)
    return ct

def floats_to_pandas(input_dir, output_fname, progress=False, **kwargs):
    """Convert MITgcm float data to pands hdf format.

    Paramters
    ---------
    input_dir : path
        Where to find the MITgcm output data
    output_fname : path
        Filename of the hdf data store
    kwargs :
        Extra keyword arguments to pass to floater.input_formats.MITgcmFloatData
    """
    import pandas as pd
    output_fname = _maybe_add_h5_suffix(output_fname)
    key = '/floats/trajectories'

    #store = pd.HDFStore(output_fname, mode='w')
    with pd.HDFStore(output_fname, mode='w') as store:
        mfd = input.MITgcmFloatData(input_dir, **kwargs)
        for block in mfd.generator(progress=progress, return_full_block=True):
            df = pd.DataFrame.from_records(block)
            store.append(key, df)
    #store.close()
