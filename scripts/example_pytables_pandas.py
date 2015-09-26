import tables
import pandas as pd
import numpy as np

output_fname = 'test.h5'
class LFloat(tables.IsDescription):
    npart   = tables.Int32Col(pos=1)   # float id number, starts at 1
    time    = tables.Float32Col(pos=2)  # time of the datapoint
    x       = tables.Float32Col(pos=3)  # x position
    y       = tables.Float32Col(pos=4)  # y position
    z       = tables.Float32Col(pos=5)  # z position

dtype = tables.description.dtype_from_descr(LFloat)

nrecs = 10
with tables.openFile(output_fname, mode='w', title='Float Data') as h5file:
    group = h5file.createGroup("/", 'floats', 'Float Data')
    table = h5file.createTable(group, 'trajectories', LFloat,
                                "Float Trajectories", expectedrows=nrecs)
    for n in range(nrecs):
        d = np.empty(1, dtype)
        d['npart'] = n
        table.append(d)

    table.cols.npart.createIndex()
    table.flush()

d = pd.read_hdf('test.h5', '/floats/trajectories', start=0, stop=5)
