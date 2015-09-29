import pandas as pd
import dask.dataframe as dd
import numpy as np

fname = 'testfile.h5'
key = '/group/dataset'

dtype = np.dtype([('idx','i4'),('val','f8')])
N = 1000000
#store = pd.HDFStore(output_fname, mode='w')
with pd.HDFStore(fname, mode='w') as store:
    recarray = np.empty(N, dtype)
    df = pd.DataFrame.from_records(recarray)
    store.append(key, df)

df = pd.read_hdf(fname, key)
ddf = dd.read_hdf(fname, key)
print(len(df))
print(len(ddf))
