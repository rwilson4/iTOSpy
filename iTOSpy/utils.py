import numpy as np
from scipy.signal import fftconvolve
from scipy.spatial.distance import cdist


def amplify(gamma, lambda_values):
    """Amplify sensitivity analysis.

    Amplifies a sensitivity analysis parameter gamma into delta values for a range of
    lambda values.

    Parameters
    ----------
    gamma : float
        Sensitivity parameter, must be greater than 1.
    lambda_values : array-like
        A list or array of lambda values, each greater than gamma.

    Returns
    -------
    delta : np.ndarray
        Array of delta values corresponding to input lambda values.

    """
    lambda_values = np.asarray(lambda_values)
    if not np.isscalar(gamma) or gamma <= 1:
        raise ValueError("gamma must be a scalar > 1")
    if np.any(lambda_values <= gamma):
        raise ValueError("All lambda values must be greater than gamma")
    delta = (gamma * lambda_values - 1) / (lambda_values - gamma)
    return delta


def gconv(g1, g2):
    """Convolve pmfs.

    Convolve two discrete probability mass functions using FFT-based convolution.

    Parameters
    ----------
    g1 : array-like
        First PMF array.
    g2 : array-like
        Second PMF array.

    Returns
    -------
    result : np.ndarray
        Convolution of g1 and g2, clipped to [0, 1].

    """
    g1 = np.asarray(g1)
    g2 = np.asarray(g2)[::-1]
    result = fftconvolve(g1, g2, mode="full")
    result = np.clip(result, 0, 1)
    return result


def addMahal(costmatrix, z, X):
    """Add Mahalanobis distances to a cost matrix for matching.

    Parameters
    ----------
    costmatrix : ndarray
        A 2D array with shape (treated, control) representing initial matching costs.
    z : array-like
        Binary treatment indicator (1 = treated, 0 = control).
    X : array-like
        Covariate matrix used to compute Mahalanobis distances.

    Returns
    -------
    adjusted_costmatrix : ndarray
        The updated cost matrix with Mahalanobis distances added.

    """
    z = np.asarray(z)
    X = np.asarray(X)
    assert set(np.unique(z)).issubset({0, 1}), "z must be binary"
    assert costmatrix.shape == (np.sum(z), np.sum(1 - z)), "costmatrix shape mismatch"

    if X.ndim == 1:
        X = X[:, np.newaxis]
    ranks = np.argsort(np.argsort(X, axis=0), axis=0).astype(float)
    cov = np.cov(ranks, rowvar=False)
    vuntied = np.var(np.arange(len(z)))
    rat = np.sqrt(vuntied / np.diag(cov))
    cov = np.diag(rat) @ cov @ np.diag(rat)
    icov = np.linalg.pinv(cov)

    Xt = ranks[z == 1]
    Xc = ranks[z == 0]
    dists = cdist(Xt, Xc, metric="mahalanobis", VI=icov)
    return costmatrix + dists


def addNearExact(costmatrix, z, exact, penalty=1000):
    """Add a penalty to the cost matrix for matches that differ on an exact match variable.

    Parameters
    ----------
    costmatrix : ndarray
        A 2D array (treated x control) of costs.
    z : array-like
        Treatment indicator.
    exact : array-like
        Categorical variable to match exactly or nearly.
    penalty : float
        Penalty to add for mismatches.

    Returns
    -------
    adjusted_costmatrix : ndarray

    """
    z = np.asarray(z)
    exact = np.asarray(exact)
    treated = exact[z == 1]
    control = exact[z == 0]
    mismatch = np.not_equal(treated[:, None], control[None, :])
    return costmatrix + mismatch * penalty


def addcaliper(costmatrix, z, p, caliper=None, penalty=1000, twostep=True):
    """Add a penalty to cost matrix for units with propensity score differences outside a caliper.

    Parameters
    ----------
    costmatrix : ndarray
        A 2D array (treated x control) of costs.
    z : array-like
        Treatment indicator.
    p : array-like
        Propensity scores.
    caliper : float or tuple, optional
        If float, applies symmetric caliper. If tuple, use (lower, upper) bounds.
    penalty : float
        Penalty for violating the caliper.
    twostep : bool
        If True, applies additional penalty for more extreme violations.

    Returns
    -------
    adjusted_costmatrix : ndarray

    """
    z = np.asarray(z)
    p = np.asarray(p)
    pt = p[z == 1]
    pc = p[z == 0]

    if caliper is None:
        sd = np.std(p)
        caliper = (-0.2 * sd, 0.2 * sd)
    elif isinstance(caliper, (int, float)):
        caliper = (-abs(caliper), abs(caliper))
    elif isinstance(caliper, (list, tuple)) and len(caliper) == 2:
        caliper = (min(caliper), max(caliper))
    else:
        raise ValueError("Invalid caliper")

    diff = pt[:, None] - pc[None, :]
    o = ((diff > caliper[1]) | (diff < caliper[0])) * penalty
    if twostep:
        o += ((diff > 2 * caliper[1]) | (diff < 2 * caliper[0])) * penalty
    return costmatrix + o


def addinteger(costmatrix, z, iscore, penalty=1000):
    """Add a penalty based on integer-valued scores between treated and control units.

    Parameters
    ----------
    costmatrix : ndarray
        A 2D array (treated x control) of initial costs.
    z : array-like
        Treatment indicator (1 = treated, 0 = control).
    iscore : array-like
        Integer-valued score for each unit.
    penalty : float
        Penalty multiplier.

    Returns
    -------
    adjusted_costmatrix : ndarray

    """
    z = np.asarray(z)
    iscore = np.asarray(iscore)
    treated_scores = iscore[z == 1]
    control_scores = iscore[z == 0]
    penalty_matrix = np.abs(treated_scores[:, None] - control_scores[None, :]) * penalty
    return costmatrix + penalty_matrix


def addquantile(costmatrix, z, p, pct=[0.2, 0.4, 0.6, 0.8], penalty=1000):
    """Bin scores into quantile categories and applies integer-based penalty.

    Parameters
    ----------
    costmatrix : ndarray
        A 2D array (treated x control) of initial costs.
    z : array-like
        Treatment indicator.
    p : array-like
        Continuous-valued scores (e.g., propensity scores).
    pct : list of float
        List of quantile thresholds between 0 and 1.
    penalty : float
        Penalty multiplier.

    Returns
    -------
    adjusted_costmatrix : ndarray

    """
    p = np.asarray(p)
    bins = np.quantile(p, [0] + pct + [1])
    iscore = np.digitize(p, bins, right=True)
    return addinteger(costmatrix, z, iscore, penalty=penalty)
