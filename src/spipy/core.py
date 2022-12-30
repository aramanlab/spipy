import numpy as np
from numpy.linalg import inv
import scipy
from scipy.spatial.distance import pdist, squareform
from scipy.special import comb
import humanize

from typing import List, Union


def explainedvariance(vals: np.ndarray) -> np.ndarray:
    return vals**2 / np.sum(vals**2)


def scaledcumsum(vals: np.ndarray) -> np.ndarray:
    return np.cumsum(vals) / np.sum(vals)


def minspaceneeded(num, partitions, bits=64):
    return humanize.natrualsize(comb(num, 2) * partitions * bits)


def distmatrixsize(num, bits=64):
    return humanize.naturalsize(num**2 * bits)


def getintervalsIQR(S: np.ndarray, alpha: float = 1.5, ql: float = 0.25, qh: float = 0.75) -> List[range]:
    """    
    finds spectral partitions. Computes log difference between each subsequent singular
    value and by default selects the differences that are larger than `ALPHA * Q3(differences)`
    i.e. finds breaks in the spectrum that explain smaller scales of variance

    Args:
    * S = singular values of a SVD factorization
    * alpha = scalar multiple of `q`
    * q = which quantile of log differences to use; by default Q3 

    Returns:
    * AbstractVector{UnitRange} indices into S corresponding to the spectral partitions
    """
    # Compute the log difference between each subsequent singular value
    potentialbreaks = abs(np.diff(np.log(S+1)))
    # Compute the Q1 and Q3 quantiles of the log differences
    Q1, Q3 = np.quantile(potentialbreaks, [ql, qh])
    # Compute θ as the Q3 quantile plus a scalar multiple of the IQR
    θ = Q3 + alpha * (Q3 - Q1)
    # Find all log differences that are greater than θ
    breaks = np.argwhere(potentialbreaks > θ).flatten() + 1
    # Concatenate the indices of the breaks with the start and end indices
    starts, ends = np.concatenate(([0], breaks)), np.concatenate((breaks, [len(S)-1]))
    # Create a list of ranges from the starts and ends indices
    intervals = [range(s, e+1) for s, e in zip(starts, ends)]
    return intervals


def getintervals(S: np.ndarray, alpha: float = 1.0, q: float = .5) -> List[range]:
    """    
    finds spectral partitions. Computes log difference between each subsequent singular
    value and by default selects the differences that are larger than `1.0 * Q2(differences)`
    i.e. finds breaks in the spectrum that explain smaller scales of variance

    Args:
    * S = singular values of a SVD factorization
    * alpha = scalar multiple of `q`
    * q = which quantile of log differences to use; by default Q2 

    Returns:
    * AbstractVector{UnitRange} indices into S corresponding to the spectral partitions
    """
    # Compute the log difference between each subsequent singular value
    potentialbreaks = abs(np.diff(np.log(S+1)))
    # Compute the quantile of the log differences
    θ = alpha * np.quantile(potentialbreaks, q)
    # Find all log differences that are greater than θ
    breaks = np.argwhere(potentialbreaks > θ).flatten() + 1
    # Concatenate the indices of the breaks with the start and end indices
    starts, ends = np.concatenate(([0], breaks)), np.concatenate((breaks, [len(S)-1]))
    # Create a list of ranges from the starts and ends indices
    intervals = [range(s, e+1) for s, e in zip(starts, ends)]
    return intervals


def calc_spi_mtx(vecs: np.ndarray, vals: np.ndarray, intervals: Union[np.ndarray, List[range]]) -> np.ndarray:
    """
    computes the cumulative spectral residual distance for spectral phylogenetic inference
    ```(∑_{p ∈ P} ||UₚΣₚ||₂)²```
    where ``P`` are the spectral partitions found with `getintervals`. 

    Args:
    * A,usv = AbstractMatrix or SVD factorization (AbstractMatrix is just passed to `svd()` before calculation)
    * SPI.Left() computes SPI matrix for LSVs; SPI.Right() computes SPI matrix for RSVs
    * alpha, q are passed to `getintervals()` see its documentation

    Returns:
    * distance matrix
    """
    # Initialize the matrix of squared spectral partition distances with zeros
    sprmtx = np.zeros((vecs.shape[0], vecs.shape[0]))
    # weight components by singular values
    wvecs = vecs @ np.diag(vals)
    for grp in intervals:
        # Compute the pairwise weighted Euclidean distance between the rows of vecs for the indices in grp
        sprmtx += squareform(pdist(wvecs[:, grp], "euclidean"))
    return sprmtx**2


def calc_spi_trace(vecs: np.ndarray, vals: np.ndarray, intervals: Union[np.ndarray, List[range]]) -> np.ndarray:
    """
    calculates spectral residual within each partition of spectrum and each pair of taxa
    returns matrix where columns are spectral partitions and rows are taxa:taxa pairs
    ordered as the upper triangle in rowwise order, or lower triangle in colwise order.

    Args:
    * vecs: either usv.U or usv.V matrix
    * vals: usv.S singular values vector
    * groups: usually calculated with `getintervals(usv.S; alpha=alpha, q=q)`

    Returns:
    * matrix where each column is the condensed uppertriangle of a distance matrix, (use scipy.spatial.distance.squareform() to recover full matrix)
    """
    # Initialize the tensor of squared spectral partition distances with zeros
    sprmtx = np.zeros((int(comb(vecs.shape[0], 2)), len(intervals)))
    # weight components by singular values
    wvecs = vecs @ np.diag(vals)
    for (i, grp) in enumerate(intervals):
        # Compute the pairwise weighted Euclidean distance between the rows of vecs for the indices in grp
        sprmtx[:, i] = pdist(wvecs[:, grp], "euclidean")
    return sprmtx


def calc_spcorr_mtx(vecs: np.ndarray, vals: np.ndarray, window=None):
    """
    Calculates pairwise spectral (pearson) correlations for a set of observations. 

    Args:
    * `vecs`, set of left singular vectors or principal components with observations on rows and components on columns
    * `vals`, vector of singular values
    * `window`, set of indices of `vecs` columns to compute correlations across

    Returns:
    * correlation matrix where each pixel is the correlation between a pair of observations
    """
    window = range(len(vals)) if window == None else window
    wvecs = vecs[:, window] @ np.diag(vals[window])
    return np.corrcoef(wvecs)


def calc_spcorr_trace(vecs: np.ndarray, vals: np.ndarray, intervals: List[range]):
    """
    calculates spectral correlation (pearson) within each partition of spectrum and each pair of taxa
    returns matrix where columns are spectral partitions and rows are taxa:taxa pairs
    ordered as the upper triangle in rowwise order, or lower triangle in colwise order.

    Args:
    * vecs: either usv.U or usv.V matrix
    * vals: usv.S singular values vector
    * groups: usually overlapping windows such as `[range(i,(i+5)) for i in range(len(vals)-5+1)]`

    Returns:
    * matrix where each column is the condensed uppertriangle of a correlation matrix, (use scipy.spatial.distance.squareform() to recover full matrix)
    """
    wvecs = vecs @ np.diag(vals)
    sprmtx = np.zeros((int(comb(vecs.shape[0], 2)), len(intervals)))
    # I = np.identity(vecs.shape[0])
    for (i, grp) in enumerate(intervals):
        sprmtx[:, i] = squareform(np.corrcoef(wvecs[:, grp]), "tovector", False)
    return sprmtx


def projectout(U: np.ndarray, S: np.ndarray, Vh: np.ndarray, window=None) -> np.ndarray:
    """
    recreates original matrix i.e. calculates ``UΣV'`` or if window is included 
    creates a spectrally filtered version of the original matrix off of the provided components in `window`.
    """
    window = range(len(S)) if window == None else window
    return U[:, window] @ np.diag(S)[window, window] @ Vh[window, :]


def projectinLSV(M: np.ndarray, S: np.ndarray, Vh: np.ndarray, window=None) -> np.ndarray:
    """
    returns estimated left singular vectors (aka: LSV or Û) for new data based on already calculated SVD factorization
    """
    window = range(len(S)) if window == None else window
    return M @ Vh[window, :].transpose() @ inv(np.diag(S[window]))


def projectinRSV(M: np.ndarray, U: np.ndarray, S: np.ndarray, window=None) -> np.ndarray:
    """
    returns estimated transposed right singular vectors (RSV or V̂ᵗ) for new data based on already calculated SVD factorization
    """
    window = range(len(S)) if window == None else window
    return inv(np.diag(S[window])) @ U[:, window].transpose() @ M


def UPGMA_tree(Dij: np.ndarray) -> np.ndarray:
    """
    shorthand for `scipy.cluster.hierarchy.average(squareform(Dij))`
    """
    return scipy.cluster.hierarchy.average(squareform(Dij))


def nwstr(hc: np.ndarray, tiplabels: Union[List[str], np.ndarray] = None, labelinternalnodes: bool = False) -> str:
    """ 
    convert `scipy.cluster.hierarchy.linkage` matrix to newick tree for import to ete3 or other phylo packages
    Args:
    * hc, output from `scipy.cluster.hierarchy.linkage()` a np.ndarray with the first
        two columns corresponding to the left and right children that are being joined at
        row index `i` and the 3rd column holding the height at which these children are
        being joined. All other columns are ignored. see `scipy` docs for more details.
    * tiplabels, list of string names in same order as distance matrix
    """
    if tiplabels is None:
        tiplabels = [str(i) for i in range(1, len(hc)+1)]
    elif isinstance(tiplabels, np.ndarray):
        tiplabels = [str(x) for x in tiplabels]
    r = hc.shape[0]
    return _nwstr(hc[:, 0:2], hc[:, 3], r, r, r+1, tiplabels, labelinternalnodes) + ";"


def _nwstr(merges: np.ndarray, heights: np.ndarray, i: int, p: int, n: int, tiplabels: List[str], labelinternalnodes: bool) -> str:
    j, k = merges[i, :]
    a = f"{tiplabels[abs(j)]}:{heights[i]:.6e}" if j < n else _nwstr(merges, heights, j, i, n, tiplabels, labelinternalnodes)
    b = f"{tiplabels[abs(k)]}:{heights[i]:.6e}" if k < n else _nwstr(merges, heights, k, i, n, tiplabels, labelinternalnodes)
    nid = f"node{len(heights) + i + 1}" if labelinternalnodes else ""
    dist = f"{heights[p] - heights[i]:.6e}"
    return f"({a},{b}){nid}:{dist}"
