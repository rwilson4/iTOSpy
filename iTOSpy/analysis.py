import numpy as np
from scipy.stats import binomtest, ttest_ind


def evalBal(z, x, statistic="s", reps=1000, trunc=0.2, nunique=2, alpha=0.05):
    """Evaluate balance between treatment groups using permutation tests.

    Parameters
    ----------
    z : array-like
        Treatment indicator (1 = treated, 0 = control).
    x : array-like
        Covariates (vector or 2D array).
    statistic : str
        "s" for standardized difference, "t" for t-test p-value.
    reps : int
        Number of permutations.
    trunc : float
        Threshold for imbalance.
    nunique : int
        Minimum number of unique values to consider.
    alpha : float
        Significance level.

    Returns
    -------
    summary : dict
        Evaluation of covariate balance.
    """
    z = np.asarray(z)
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    treated = x[z == 1]
    control = x[z == 0]

    if statistic == "s":

        def stat(v1, v0):
            return abs(np.mean(v1) - np.mean(v0)) / np.std(np.concatenate([v1, v0]))

        label = "standardized difference"
    elif statistic == "t":

        def stat(v1, v0):
            return ttest_ind(v1, v0, equal_var=False).pvalue

        label = "two-sample t-statistic"
    else:
        raise ValueError("statistic must be 's' or 't'")

    obs_stats = np.array(
        [stat(treated[:, j], control[:, j]) for j in range(x.shape[1])]
    )
    perm_stats = np.zeros((reps, x.shape[1]))

    for r in range(reps):
        perm = np.random.permutation(z)
        pt = x[perm == 1]
        pc = x[perm == 0]
        perm_stats[r] = [stat(pt[:, j], pc[:, j]) for j in range(x.shape[1])]

    pvals = np.mean(perm_stats >= obs_stats[None, :], axis=0)
    return {"observed": obs_stats, "pvals": pvals, "statistic": label}


def noether(y, f=2 / 3, gamma=1, alternative="greater"):
    """Perform Noether's test for symmetry about zero, with optional sensitivity analysis.

    Parameters
    ----------
    y : array-like
        Differences or signed outcomes.
    f : float
        Proportion of largest values to count (0 < f < 1).
    gamma : float
        Sensitivity parameter (Gamma ≥ 1).
    alternative : str
        One of "greater", "less", or "two.sided".

    Returns
    -------
    p_value : float
    """
    y = np.asarray(y)
    n = len(y)
    cutoff = int(np.floor(n * f))
    largest = np.argsort(-np.abs(y))[:cutoff]
    count = np.sum(y[largest] > 0)

    if gamma != 1 and alternative == "two.sided":
        raise ValueError("Two-sided test not available with gamma != 1")

    if gamma == 1:
        pr = 0.5
    elif alternative == "less":
        pr = 1 / (1 + gamma)
    else:
        pr = gamma / (1 + gamma)

    if alternative == "two.sided":
        return binomtest(count, cutoff, pr, alternative="two-sided").pvalue
    elif alternative == "less":
        return binomtest(count, cutoff, pr, alternative="less").pvalue
    else:
        return binomtest(count, cutoff, pr, alternative="greater").pvalue


def startcost(z):
    """Create a zero cost matrix for all treated-control pairs.

    Parameters
    ----------
    z : array-like
        Treatment indicator vector.

    Returns
    -------
    cost_matrix : ndarray
        Zero matrix of shape (#treated, #control).
    """
    z = np.asarray(z)
    treated = np.sum(z == 1)
    control = np.sum(z == 0)
    return np.zeros((treated, control))


def ev(sc, z, m, g, method):
    """Compute expected value and variance under sensitivity analysis model.

    Parameters
    ----------
    sc : array-like
        Score or test statistic values.
    z : array-like
        Treatment indicator vector.
    m : int
        Number of treated units.
    g : float
        Sensitivity parameter (Gamma ≥ 1).
    method : str
        Method to use: "AD", "LS", or "BU".

    Returns
    -------
    dict : {"expect": float, "vari": float}
    """
    sc = np.asarray(sc)
    z = np.asarray(z)
    bigN = len(z)
    n = np.sum(z)
    assert bigN > n > 0 and bigN > m > 0
    q = np.sort(sc)
    u = np.concatenate([np.zeros(bigN - m), np.ones(m)])

    if method == "AD":
        if bigN > 200 and n > 50 and (bigN - n) > 50 and m > 50 and (bigN - m) > 50:
            method = "LS"
        else:
            method = "BU"

    if method in ["LS", "BU"]:
        if method == "LS":
            ets = np.exp(g * u)
            et = np.sum(ets)
            p = ets / et
            expect = np.dot(p, q)
            vari = np.dot(p, (q - expect) ** 2)
            return {"expect": expect, "vari": vari}
        else:
            # Exact enumeration for small N
            from itertools import combinations

            pos = np.where(u == 1)[0]
            combos = list(combinations(range(bigN), m))
            weights = []
            values = []
            for combo in combos:
                w = g ** np.sum(u[list(combo)])
                val = np.sum(q[list(combo)])
                weights.append(w)
                values.append(val)
            weights = np.array(weights)
            values = np.array(values)
            weights /= np.sum(weights)
            expect = np.sum(weights * values)
            vari = np.sum(weights * (values - expect) ** 2)
            return {"expect": expect, "vari": vari}
    else:
        raise ValueError("Invalid method. Use 'AD', 'LS', or 'BU'.")


def evall(sc, z, g, method):
    """Apply `ev` for all values of m from 1 to N-1.

    Parameters
    ----------
    sc : array-like
        Score or test statistic values.
    z : array-like
        Treatment indicator vector.
    g : float
        Sensitivity parameter (Gamma ≥ 1).
    method : str
        Method for ev(): "AD", "LS", or "BU".

    Returns
    -------
    DataFrame-like dict with keys "m", "expect", "var"
    """
    sc = np.asarray(sc)
    z = np.asarray(z)
    bigN = len(z)
    results = {"m": [], "expect": [], "var": []}
    for m in range(1, bigN):
        out = ev(sc, z, m, g, method)
        results["m"].append(m)
        results["expect"].append(out["expect"])
        results["var"].append(out["vari"])
    return results
