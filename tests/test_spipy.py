#%%
import spipy
from spipy.cli import main

import numpy as np
import scipy

# %%
m = np.random.rand(10, 15)
u, s, vt = np.linalg.svd(m, False)

#%%
Dij = spipy.calc_spi_mtx(u, s, spipy.getintervals(s)) / 15.

Dij.shape
Dij

#%%
Cij = spipy.calc_spcorr_mtx(u, s, range(1,4))

#%%
I = np.identity(5)
squareform(Cij[:5, :5], "tovector", False)

#%%
from scipy.spatial.distance import squareform

#%%
squareform()

#%%
Cij = spipy.calc_spcorr_trace(u, s, [range(i, i+3) for i in range(10-2)])

#%%
list(map(list,[range(i, i+3) for i in range(10-2)]))

#%%
Cij.shape

#%%
Dij = spipy.calc_spi_trace(u, s, spipy.getintervals(s)) / 15.

#%%
Dij.shape

#%%
len(spipy.getintervals(s))

#%%
def test_calc_spi_mtx():
    pass

def test_main():
    assert main([]) == 0
