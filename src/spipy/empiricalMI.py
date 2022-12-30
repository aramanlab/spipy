import numpy as np

from scipy.stats import entropy
from scipy.stats.contingency import crosstab

from typing import List, Union


def empiricalMI_2dcont(a: Union[np.ndarray, List[float]], b: Union[np.ndarray, List[float]], nbins: int = 50, base: float = np.e, normalize: bool = False) -> float:
    """
    computes empirical MI from identity of ``H(a) + H(b) - H(a,b)``. where
    ``H := -sum(p(x)*log(p(x))) + log(Δ)``
    the ``+ log(Δ)`` corresponds to the log binwidth and unbiases the entropy estimate from binwidth choice.
    estimates are roughly stable from ``32`` (``32^2 ≈ 1000`` total bins) to size of sample. going from a small undersestimate to a small overestimate across that range.
    We recommend choosing the `sqrt(mean(1000, samplesize))` for `nbins` argument, or taking a few estimates across that range and averaging.

    Args:
    * a, vecter of length N
    * b, AbstractVector of length N
    * nbins, number of bins per side, use 1000 < nbins^2 < length(a) for best results
    * base, base unit of MI (defaults to nats with base=ℯ)
    * normalize, bool, whether to normalize with mi / mean(ha, hb)

    Returns:
    * MI
    """
    if isinstance(a, list):
        a = np.array(a)
    if isinstance(b, list):
        b = np.array(b)
    # num samples marginal then total
    N = len(a)
    breaks_a = np.linspace(np.min(a), np.max(a), nbins)
    breaks_b = np.linspace(np.min(b), np.max(b), nbins)

    ab_counts, _, _ = np.histogram2d(a, b, bins=(breaks_a, breaks_b))
    a_counts = np.sum(ab_counts, axis=1)
    b_counts = np.sum(ab_counts, axis=0)

    # frequency
    afreq = a_counts / N
    bfreq = b_counts / N
    abfreq = ab_counts / N

    Δ_a = breaks_a[1] - breaks_a[0]
    Δ_b = breaks_b[1] - breaks_b[0]

    # approx entropy
    ha = -np.sum(afreq * np.log(afreq)) + np.log(Δ_a)
    hb = -np.sum(bfreq * np.log(bfreq)) + np.log(Δ_b)
    hab = -np.sum(abfreq * np.log(abfreq)) + np.log(Δ_a * Δ_b)

    # mi
    mi = ha + hb - hab
    mi = 2*mi / (ha+hb) if normalize else mi
    return mi


def empiricalMI_masked(ab: Union[np.ndarray, List[float]], mask: Union[np.ndarray, List[bool]], nbins: int = 100, base: float = np.e, normalize: bool = False) -> float:
    """
    computes empirical MI from identity of ``H(a,b) - (Na/N * ha + Nb/N * hb)``. where
    ``H := -sum(p(x)*log(p(x))) + log(Δ)``
    the ``+ log(Δ)`` corresponds to the log binwidth and unbiases the entropy estimate from binwidth choice.
    estimates are roughly stable from ``32`` (``32^2 ≈ 1000`` total bins) to size of sample. going from a small undersestimate to a small overestimate across that range.
    We recommend choosing the `sqrt(mean(1000, samplesize))` for `nbins` argument, or taking a few estimates across that range and averaging.

    Args:
    * ab, vecter of length N continous variables, vector to be binned
    * mask, vector of bools that groups `ab` to an in vs. out group. Mutual information is computed between these two groups.
    * nbins, number of bins per side, use 1000 < nbins^2 < length(a) for best results
    * base, base unit of MI (defaults to nats with base=ℯ)
    * normalize, bool, whether to normalize with mi / mean(ha, hb)

    Returns:
    * MI
    """
    if len(ab) != len(mask):
        raise ValueError(f"length of vals and meta must match; got vals={len(ab)}, meta={len(mask)}")
    # num samples marginal then total
    N = len(ab)
    Na = np.sum(mask)
    Nb = len(mask) - Na
    mask = np.array(mask, dtype=bool)
    # if mask is not grouping than there is no added information
    if Na == 0 or Nb == 0:
        return 0.0

    ## otherwise ##

    # form edges
    edges = np.linspace(np.min(ab), np.max(ab), nbins)

    # fit hist and get counts
    a_counts, _ = np.histogram(ab[mask], edges)
    b_counts, _ = np.histogram(ab[np.logical_not(mask)], edges)
    ab_counts = a_counts + b_counts

    # get binwidth
    Δ = edges[1] - edges[0]

    # frequency
    afreq = a_counts / Na
    bfreq = b_counts / Nb
    abfreq = ab_counts / N

    # approx entropy
    ha = np.sum(afreq * np.log(afreq)) + np.log(base, Δ)
    hb = np.sum(bfreq * np.log(bfreq)) + np.log(base, Δ)
    hab = np.sum(abfreq * np.log(abfreq)) + np.log(base, Δ)

    # mi
    mi = hab - (Na / N * ha + Nb / N * hb)  # original had flipped signs
    # return (mi = mi, ha = ha, hb = hb, hab = hab)
    if normalize:
        return 2 * mi / (ha + hb)
    return mi


def empiricalMI_categorical(a: Union[np.ndarray, List[int], List[str]], b: Union[np.ndarray, List[int], List[str]], base: float = np.e, normalize: bool = False) -> float:
    """
    Standard mutual information calculation on catagorical variables. 
    computes a contigency table of the lists `a` and `b` 
    then calculates mutual information on the identity 
    ``H(a) + H(b) - H(a,b)``. where ``H := -sum(p(x)*log(p(x)))``

    Args:
    * a, vecter of length N 
    * v, vecter of length N 
    * base, base unit of MI (defaults to nats with base=ℯ)
    * normalize, bool, whether to normalize with mi / mean(ha, hb)

    Returns:
    * MI

    """
    (_, _), counts = crosstab(a, b)
    N = counts.sum()
    Ha = entropy(counts.sum(axis=1)/N, base=base)
    Hb = entropy(counts.sum(axis=0)/N, base=base)
    Hab = entropy(counts/N, base=base)
    mi = Ha + Hb - Hab
    mi = mi if not normalize else 2 * mi / (Ha+Hb)
    return mi
