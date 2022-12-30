# %%
import spipy
from spipy.cli import main
from re import sub
import numpy as np

# %%
m = np.random.rand(10, 15)
u, s, vt = np.linalg.svd(m, False)

# %%
expS = np.array([
    10.0, 3.0, 3.0, 1., 1., 1., 0.,
])

# %%
mh = np.array([
    [0., 1., 0., 1., 1., 1.],
    [0., 1., 1., 0., 1., 1.],
    [1., 0., 1., 1., 0., 1.],
    [1., 0., 1., 1., 1., 0.],
])
labels = ["A", "B", "C", "D"]

# %%
uh, sh, vth = np.linalg.svd(mh, False)
dij = spipy.calc_spi_mtx(uh, sh, [[0], [1], [2, 3]])

# %%
Z = spipy.UPGMA_tree(dij)
Z_true = np.array([
    [2., 3., 2., 2.],
    [0., 1., 2., 2.],
    [4., 5., 7.46410162, 4.]
])

# %%
nwstring = spipy.nwstr(Z, labels, False)
nwstring_scrubbed = sub("(:)[^,)]*(?=[,);])", "", nwstring)

# %%


def test_getintervals():
    assert [list(i) for i in spipy.getintervals(expS)] == [[0], [1, 2], [3, 4, 5], [6]]


def test_getintervalsIQR():
    assert [list(i) for i in spipy.getintervalsIQR(expS)] == [[0, 1, 2, 3, 4, 5, 6]]


def test_calc_spi_mtx():
    uij = spipy.calc_spi_mtx(u, s, spipy.getintervals(s)) / 15.
    assert uij.shape[0] == uij.shape[1] == u.shape[0]

    vij = spipy.calc_spi_mtx(vt.transpose(), s, spipy.getintervals(s)) / 10.
    assert vij.shape[0] == vij.shape[1] == vt.shape[1]


def test_projectout():
    assert np.isclose(m, spipy.projectout(u, s, vt)).all()
    assert spipy.projectout(u, s, vt, range(1, 5)).shape == m.shape


def test_projectinLSV():
    assert np.isclose(u, spipy.projectinLSV(m, s, vt)).all()
    assert spipy.projectinLSV(m, s, vt, range(4)).shape == (m.shape[0], 4)


def test_projectinRSV():
    assert np.isclose(vt, spipy.projectinRSV(m, u, s)).all()
    assert spipy.projectinRSV(m, u, s, range(4)).shape == (4, m.shape[1])


def test_UPGMA_tree():
    assert np.isclose(Z, Z_true).all()


def test_nwstr():
    assert nwstring_scrubbed in [
        "((A,B),(C,D));", "((A,B),(D,C));", "((B,A),(C,D));", "((B,A),(D,C));",
        "((C,D),(A,B));", "((D,C),(A,B));", "((C,D),(B,A));", "((D,C),(B,A));"
    ]


# %%
N = 10000
smp = np.random.multivariate_normal([0, 0], [[1, .0], [.0, 1]], N)
msk_a = np.random.choice([True, False], N)
msk_b = np.random.choice([True, False], N)


def test_empiricalMI_2dcont():
    assert 0. < spipy.empiricalMI_2dcont(smp[:, 0], smp[:, 1]) < 1.


def test_empiricalMI_masked():
    assert 0. < spipy.empiricalMI_masked(smp[:, 0], msk_a) < 1.


def test_empiricalMI_categorical():
    assert 0. < spipy.empiricalMI_categorical(msk_a, msk_b) < 1.


# %%
def test_main():
    assert main([]) == 0
