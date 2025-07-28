import numpy as np
from scipy.signal import fftconvolve
from scipy.spatial.distance import cdist


def amplify(gamma, lambda_values):
    """Amplification of sensitivity analysis in observational studies.

    Uses the method in Rosenbaum and Silber (2009) to interpret a value of the
    sensitivity parameter gamma. Each value of gamma amplifies to a curve (lambda,delta)
    in a two-dimensional sensitivity analysis, the inference being the same for all
    points on the curve. That is, a one-dimensional sensitivity analysis in terms of
    gamma has a two-dimensional interpretation in terms of (lambda,delta).

    Parameters
    ----------
    gamma : float
        gamma > 1 is the value of the sensitivity parameter, for instance the parameter
        in senmv. length(gamma)>1 will generate an error.
    lambda_values : array-like
        lambda is a vector of values > gamma. An error will result unless lambda[i] >
        gamma > 1 for every i.

    Returns
    -------
    delta : np.ndarray
        Array of delta values corresponding to input lambda values.

    Notes
    -----
    A single value of gamma, say gamma = 2.2 in the example, corresponds to a curve of
    values of (lambda, delta), including (3, 7), (4, 4.33), (5, 3.57), and (7, 3) in the
    example. An unobserved covariate that is associated with a lambda = 3 fold increase
    in the odds of treatment and a delta = 7 fold increase in the odds of a positive
    pair difference is equivalent to gamma = 2.2.

    The curve is gamma = (lambda*delta + 1)/(lambda+delta). Amplify is given one gamma
    and a vector of lambdas and solves for the vector of deltas. The calculation is
    elementary.

    This interpretation of gamma is developed in detail in Rosenbaum and Silber (2009),
    and it makes use of Wolfe's (1974) family of semiparametric deformations of an
    arbitrary symmetric distribuiton. See also Rosenbaum (2020, Section 3.6). For an
    elementary discussion, see Rosenbaum (2017, Table 9.1).

    Strictly speaking, the amplification describes matched pairs, not matched sets. The
    senm function views a k-to-1 matched set with k controls matched to one treated
    individual as a collection of k correlated treated-minus-control matched pair
    differences; see Rosenbaum (2007). For matched sets, it is natural to think of the
    amplification as describing any one of the k matched pair differences in a k-to-1
    matched set.

    The curve has asymptotes that the function amplify does not compute: gamma
    corresponds with (lambda,delta) = (gamma, Inf) and (Inf, gamma).

    A related though distict idea is developed in Gastwirth et al (1998). The two
    approaches agree when the outcome is binary, that is, for McNemar's test.

    The amplify function is also in the sensitivitymv package where a different example
    is used.

    Examples
    --------
    #  Consider a treated-control match pair as the unit of measure,
    #  analogous to one meter or one foot.  The calculation
    #  amplify(4,7) says that, in a matched pair, gamma=4
    #  is the same a bias that increases the odds of treatment
    #  7-fold and increases the odds of positive matched-pair
    #  difference in outcomes 9-fold.
    amplify(4,7)
    #  It is also true that, in a matched pair, gamma=4
    #  is the same a bias that increases the odds of treatment
    #  9-fold and increases the odds of positive matched-pair
    #  difference in outcomes 7-fold.
    amplify(4,9)
    #  It is also true that, in a matched pair, gamma=4
    #  is the same a bias that increases the odds of treatment
    #  5-fold and increases the odds of positive matched-pair
    #  difference in outcomes 19-fold.
    amplify(4,5)
    # The amplify function can produce the entire curve at once:
    amplify(4,5:19)

    References
    ----------
    Gastwirth, J. L., Krieger, A. M., Rosenbaum, P. R. (1998)
    <doi:10.1093/biomet/85.4.907> Dual and simultaneous sensitivity analysis for matched
    pairs. Biometrika, 85, 907-920.

    Rosenbaum, P. R. and Silber, J. H. (2009) <doi:10.1198/jasa.2009.tm08470>
    Amplification of sensitivity analysis in observational studies. Journal of the
    American Statistical Association, 104, 1398-1405.

    Rosenbaum, P. R. (2017) <doi:10.4159/9780674982697> Observation and Experiment: An
    Introduction to Causal Inference. Cambridge, MA: Harvard University Press. Table
    9.1.

    Rosenbaum, P. R. (2020) <doi:10.1007/978-3-030-46405-9> Design of Observational
    Studies (2nd ed.) NY: Springer. Section 3.6.

    Wolfe, D. A. (1974) <doi:10.2307/2286025> A charaterization of population weighted
    symmetry and related results. Journal of the American Statistical Association, 69,
    819-822.

    """
    lambda_values = np.asarray(lambda_values)
    if not np.isscalar(gamma) or gamma <= 1:
        raise ValueError("gamma must be a scalar > 1")
    if np.any(lambda_values <= gamma):
        raise ValueError("All lambda values must be greater than gamma")
    delta = (gamma * lambda_values - 1) / (lambda_values - gamma)
    return delta


def gconv(g1, g2):
    """Convolution of Two Probability Generating Functions.

    Computes the convolution of two probability generating functions using the convolve
    function in the stats package. The convolve function uses the fast fourier
    transform.

    Parameters
    ----------
    g1 : array-like
        A probability generating function. A vector g1 for a random variable X taking
        values 0, 1, 2, ..., length(g1)-1, where g1[i] = Pr(X=i-1)For example, g1 =
        c(2/3, 1/3) is the generating function of a binary random variable X with
        Pr(X=0)=2/3, Pr(X=1)=1/3. The random variable that is 0 with probability 1 has
        g1=1.
    g2 : array-like
        Another probability generating function for a random variable Y. For a fair die,
        g2 = c(0, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6).

    Returns
    -------
    result : np.ndarray
        The probability generating function of X+Y when X and Y are independent.

    Notes
    -----
    The gconv function is a slight modification of a similar function in the
    sensitivity2x2xk package.

    Examples
    --------
    gconv(c(2/3,1/3),c(2/3,1/3))

    gconv(1,c(2/3,1/3))

    round(gconv(c(0, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6),
         c(0, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6)),3)
    #
    # Compute the exact distribution of Quade's treated-control
    # statistic forI=3 blocks of size J=3.
    #
    # Block with range rank = 1
    rk1<-c(0,1/3,1/3,1/3)
    names(rk1)<-0:3
    rk1
    #
    # Block with range rank = 2
    rk2<-c(0,0,1/3,0,1/3,0,1/3)
    names(rk2)<-0:6
    rk2
    #
    # Block with range rank = 3
    rk3<-c(0,0,0,1/3,0,0,1/3,0,0,1/3)
    names(rk3)<-0:9
    rk3
    #
    # Convolution of rk1 and rk2
    round(gconv(rk1,rk2),3)
    1/(3^2)
    #
    # Convolution of rk1, rk2 and rk3
    round(gconv(gconv(rk1,rk2),rk3),3)
    1/(3^3)

    References
    ----------
    Pagano, M. and Tritchler, D. (1983) <doi:10.2307/2288653> On obtaining permutation
    distributions in polynomial time. Journal of the American Statistical Association,
    78, 435-440.

    Rosenbaum, P. R. (2020) <doi:10.1007/978-3-030-46405-9> Design of Observational
    Studies. New York: Springer. Chapter 3 Appendix: Exact Computations for Sensitivity
    Analysis, page 103.

    """
    g1 = np.asarray(g1)
    g2 = np.asarray(g2)[::-1]
    result = fftconvolve(g1, g2, mode="full")
    result = np.clip(result, 0, 1)
    return result


def addMahal(costmatrix, z, X):
    """Rank-Based Mahalanobis Distance Matrix.

    Adds a rank-based Mahalanobis distance to an exisiting distance matrix.

    Parameters
    ----------
    costmatrix : ndarray
        An existing cost matrix with sum(z) rows and sum(1-z) columns. The function
        checks the compatability of costmatrix, z and p; so, it may stop with an error
        if these are not of appropriate dimensions. In particular, costmatrix may come
        from startcost().
    z : array-like
        A vector with z[i]=1 if individual i is treated or z[i]=0 if individual i is
        control. The rows of costmatrix refer to treated individuals and the columns
        refer to controls.
    X : array-like
        A matrix with length(z) rows containing covariates.

    Returns
    -------
    adjusted_costmatrix : ndarray
        A new distance matrix that is the sum of costmatrix and the rank-based
        Mahalanobis distances.

    Notes
    -----
    The rank-based Mahalanobis distance is defined in section 9.3 of Rosenbaum (2020).

    Examples
    --------
    data(binge)
    # Select two treated and three controls from binge
    d<-binge[is.element(binge$SEQN,c(109315,109365,109266,109273,109290)),]
    z<-1*(d$AlcGroup=="B")
    names(z)<-d$SEQN
    attach(d)
    x<-cbind(age,female)
    detach(d)
    rownames(x)<-d$SEQN
    dist<-startcost(z)
    z
    x
    dist
    dist<-addMahal(dist,z,x)
    dist
    rm(z,x,dist,d)

    References
    ----------
    Rosenbaum, P. R. (2020) <doi:10.1007/978-3-030-46405-9> Design of Observational
    Studies (2nd Edition). New York: Springer.

    Rubin, D. B. (1980) <doi:10.2307/2529981> Bias reduction using Mahalanobis-metric
    matching. Biometrics, 36, 293-298.

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
    """Add a Near-exact Penalty to an Exisiting Distance Matrix.

    Parameters
    ----------
    costmatrix : ndarray
        An existing cost matrix with sum(z) rows and sum(1-z) columns. The function
        checks the compatability of costmatrix, z and p; so, it may stop with an error
        if these are not of appropriate dimensions. In particular, costmatrix may come
        from startcost().
    z : array-like
        A vector with z[i]=1 if individual i is treated or z[i]=0 if individual i is
        control. The rows of costmatrix refer to treated individuals and the columns
        refer to controls.
    exact : array-like
        A vector with the same length as z. Typically, exact take a small or moderate
        number of values.
    penalty : float
        One positive number.

    Returns
    -------
    adjusted_costmatrix : ndarray
        A penalized distance matrix.

    Notes
    -----
    If the ith treated individual and the jth control have different values of exact,
    then the distance between them in costmatrix is increased by adding penalty.

    A sufficiently large penalty will maximize the number of individuals exactly matched
    for exact. A smaller penalty will tend to increase the number of individuals matched
    exactly, without prioritizing one covariate over all others.

    If the left distance matrix is penalized, it will affect pairing and balance;
    however, if the right distance matrix is penalized it will affect balance only, as
    in the near-fine balance technique of Yang et al. (2012).

    Adding several near-exact penalties for different covariates on the right distance
    matrix implements a Hamming distance on the joint distribution of those covariates,
    as discussed in Zhang et al. (2023).

    Near-exact matching for a nominal covariate is discussed and contrasted with exact
    matching in Sections 10.3 and 10.4 of Rosenbaum (2020). Near-exact matching is
    always feasible, because it implements a constraint using a penalty. Exact matching
    may be infeasible, but when feasible it may be used to speed up computations. For an
    alternative method of speeding computations, see Yu et al. (2020) who identify
    feasible constraints very quickly prior to matching with those constraints.

    Examples
    --------
    data(binge)
    # Select two treated and three controls from binge
    d<-binge[is.element(binge$SEQN,c(109315,109365,109266,109273,109290)),]
    z<-1*(d$AlcGroup=="B")
    names(z)<-d$SEQN
    attach(d)
    x<-data.frame(age,female)
    detach(d)
    rownames(x)<-d$SEQN
    dist<-startcost(z)
    z
    x
    dist
    addNearExact(dist,z,x$female)
    addNearExact(dist,z,x$age<40,penalty=10)

    # Combine several penalties
    dist<-addNearExact(dist,z,x$female)
    dist<-addNearExact(dist,z,x$age<40,penalty=10)
    dist
    dist<-addNearExact(dist,z,x$age<60,penalty=5)
    dist
    # This distance suggests pairing 109315-109266
    # and 109365-109290
    rm(z,x,dist,d)

    References
    ----------
    Rosenbaum, P. R. (2020) <doi:10.1007/978-3-030-46405-9> Design of Observational
    Studies (2nd Edition). New York: Springer.

    Yang, D., Small, D. S., Silber, J. H. and Rosenbaum, P. R. (2012)
    <doi:10.1111/j.1541-0420.2011.01691.x> Optimal matching with minimal deviation from
    fine balance in a study of obesity and surgical outcomes. Biometrics, 68, 628-636.
    (Extension of fine balance useful when fine balance is infeasible. Comes as close as
    possible to fine balance. Implemented in makematch() by placing a large near-exact
    penalty on a nominal/integer covariate x1 on the right distance matrix.)

    Yu, R., Silber, J. H., Rosenbaum, P. R. (2020) <doi:10.1214/19-STS699> Matching
    Methods for Observational Studies Derived from Large Administrative Databases.
    Statistical Science, 35, 338-355.

    Zhang, B., D. S. Small, K. B. Lasater, M. McHugh, J. H. Silber, and P. R. Rosenbaum
    (2023) <doi:10.1080/01621459.2021.1981337> Matching one sample according to two
    criteria in observational studies. Journal of the American Statistical Association,
    118, 1140-1151.

    Zubizarreta, J. R., Reinke, C. E., Kelz, R. R., Silber, J. H. and Rosenbaum, P. R.
    (2011) <doi:10.1198/tas.2011.11072> Matching for several sparse nominal variables in
    a case control study of readmission following surgery. The American Statistician,
    65(4), 229-238.

    """
    z = np.asarray(z)
    exact = np.asarray(exact)
    treated = exact[z == 1]
    control = exact[z == 0]
    mismatch = np.not_equal(treated[:, None], control[None, :])
    return costmatrix + mismatch * penalty


def addcaliper(costmatrix, z, p, caliper=None, penalty=1000, twostep=True):
    """Add a Caliper to an Existing Cost Matrix.

    For one covariate, adds a caliper to an existing cost matrix.

    Parameters
    ----------
    costmatrix : ndarray
        An existing cost matrix with sum(z) rows and sum(1-z) columns. The function
        checks the compatability of costmatrix, z and p; so, it may stop with an error
        if these are not of appropriate dimensions. In particular, costmatrix may come
        from startcost().
    z : array-like
        A vector with z[i]=1 if individual i is treated or z[i]=0 if individual i is
        control. The rows of costmatrix refer to treated individuals and the columns
        refer to controls.
    p : array-like
        A vector with the same length as p. The vector p is the covariate for which a
        caliper is needed.
    caliper : float or tuple, optional
        Determines the type and length of the caliper. The caliper becomes a vector cvex
        with length 2. If is.null(caliper), then the caliper is +/- 0.2 times the
        standard deviation of p, namely cvec = c(-.2,.2)*sd(p). If caliper is a single
        number, then the caliper is +/- caliper, or cvec = c(-1,1)*abs(caliper). If
        caliper is a vector of length 2, then an asymmetric caliper is used, cvec =
        c(min(caliper),max(caliper)), where min(caliper) must be negative and max
        caliper must be positive.
    penalty : float
        Let I be the index of ith treated individual, 1,...,sum(z), and J be the index
        of the jth control, j=1,...,sum(1-z), so 1 <= I <= length(z) and so 1 <= J <=
        length(z). The penality added to costmatrix[i,j] is 0 if cvec[1] <= p[I]-p[J] <=
        cvex[2].
    twostep : bool
        If twostep is FALSE, then no action is taken. If twostep is true, no action is
        take if 2 cvec[1] <= p[I]-p[J] <= 2 cvex[2], and otherwise costmatrix[i,j] is
        further increased by adding penalty. In words, the penalty is doubled if
        p[I]-p[J] falls outside twice the caliper.

    Returns
    -------
    adjusted_costmatrix : ndarray
        A penalized costmatrix.

    Notes
    -----
    For discussion of directional calipers, see Yu and Rosenbaum (2019).

    Examples
    --------
    data(binge)
    # Select two treated and three controls from binge
    d<-binge[is.element(binge$SEQN,c(109315,109365,109266,109273,109290)),]
    z<-1*(d$AlcGroup=="B")
    names(z)<-d$SEQN
    attach(d)
    x<-data.frame(age,female)
    detach(d)
    rownames(x)<-d$SEQN
    dist<-startcost(z)
    z
    x
    dist

    # Ten-year age caliper
    addcaliper(dist,z,x$age,caliper=10,twostep=FALSE)

    # Ten-year age caliper with twostep=TRUE
    addcaliper(dist,z,x$age,caliper=10,twostep=TRUE)

    # Same ten-year age caliper with twostep=TRUE
    addcaliper(dist,z,x$age,caliper=c(-10,10))

    # Asymmetric, directional age caliper with twostep=TRUE
    addcaliper(dist,z,x$age,caliper=c(-2,10))
    # Treated 109315 aged 30 is more than 2 years younger
    # than control 109273 aged 36, 30-36<(-2), so
    # row 109315 column 109273 is penalized, indeed
    # double penalized, as 30-36<2*(-2)

    rm(z,x,dist,d)

    References
    ----------
    Cochran, William G., and Donald B. Rubin. Controlling bias in observational studies:
    A review. Sankhya: The Indian Journal of Statistics, Series A 1973;35:417-446.

    Yu, Ruoqi, and Paul R. Rosenbaum. <doi:10.1111/biom.13098> Directional penalties for
    optimal matching in observational studies. Biometrics 2019;75:1380-1390.

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
    """Add an Integer Penalty to an Existing Distance Matrix.

    Takes an integer valued covariate, and adds a penalty proportional to the difference
    in the integer values, with proportionality constant penalty.

    Parameters
    ----------
    costmatrix : ndarray
        An existing cost matrix with sum(z) rows and sum(1-z) columns. The function
        checks the compatability of costmatrix, z and p; so, it may stop with an error
        if these are not of appropriate dimensions. In particular, costmatrix may come
        from startcost().
    z : array-like
        A vector with z[i]=1 if individual i is treated or z[i]=0 if individual i is
        control. The rows of costmatrix refer to treated individuals and the columns
        refer to controls.
    iscore : array-like
        An vector of integers with length equal to length(z).
    penalty : float
        One positive number used to penalize mismatches for iscore.

    Returns
    -------
    adjusted_costmatrix : ndarray
        A penalized distance matrix.

    Notes
    -----
    If a treated and control individual differ on iscore in absolute value by dif, then
    the distance between them is increased by adding dif*penalty.

    Examples
    --------
    data(binge)
    # Select two treated and four controls from binge
    d<-binge[is.element(binge$SEQN,c(109315,109365,109266,109273,109290,109332)),]
    attach(d)
    z<-1*(AlcGroup=="B")
    names(z)<-d$SEQN
    rbind(z,education)
    dist<-startcost(z)
    addinteger(dist,z,education,penalty=3)
    detach(d)
    rm(d,dist,z)

    """
    z = np.asarray(z)
    iscore = np.asarray(iscore)
    treated_scores = iscore[z == 1]
    control_scores = iscore[z == 0]
    penalty_matrix = np.abs(treated_scores[:, None] - control_scores[None, :]) * penalty
    return costmatrix + penalty_matrix


def addquantile(costmatrix, z, p, pct=[0.2, 0.4, 0.6, 0.8], penalty=1000):
    """Cut a Covariate at Quantiles and Add a Penalty for Different Quantile Categories.

    Parameters
    ----------
    costmatrix : ndarray
        An existing cost matrix with sum(z) rows and sum(1-z) columns. The function
        checks the compatability of costmatrix, z and p; so, it may stop with an error
        if these are not of appropriate dimensions. In particular, costmatrix may come
        from startcost().
    z : array-like
        A vector with z[i]=1 if individual i is treated or z[i]=0 if individual i is
        control. The rows of costmatrix refer to treated individuals and the columns
        refer to controls.
    p : array-like
        A vector of length equal to length(z). Quantiles of p will penalize the
        distance.
    pct : list of float
        A vector of numbers strictly between 0 and 1. These determine the quantiles of
        p. For instance, c(.25,.5,.75) uses the quartiles of p.
    penalty : float
        One positive number used as a penalty.

    Returns
    -------
    adjusted_costmatrix : ndarray
        A penalized distance matrix.

    Notes
    -----
    The vector p is cut at its quantiles defined by pct, and the integer difference in
    quantile categories is multiplied by penalty and added to the distance matrix. The
    function is similar to addinteger(), except the integer values are not specified,
    but rather are deduced from the quantiles.

    If you cannot match for the quantile category of p, then addquantile() prefers to
    match from an adjacent quantile category.

    Examples
    --------
    data(binge)
    d<-binge[binge$AlcGroup!="N",]
    attach(d)
    z<-1*(AlcGroup=="B")
    names(z)<-SEQN
    dist<-startcost(z)
    quantile(age,pct=c(1/4,1/2,3/4))
    rbind(z,age)[,1:20]
    addquantile(dist,z,d$age,pct=c(1/4,1/2,3/4),penalty=5)[1:5,1:7]
    detach(d)
    rm(z,dist,d)

    """
    p = np.asarray(p)
    bins = np.quantile(p, [0] + pct + [1])
    iscore = np.digitize(p, bins, right=True)
    return addinteger(costmatrix, z, iscore, penalty=penalty)
