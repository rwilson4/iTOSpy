import numpy as np
from scipy.stats import binomtest, ttest_ind


def evalBal(z, x, statistic="s", reps=1000, trunc=0.2, nunique=2, alpha=0.05):
    """Evaluate Covariate Balance in a Matched Sample.

    The covariate balance in a matched sample is compared to the balance that would have
    been obtained in a completely randomized experiment built from the same people. The
    existing matched sample is randomized to treatment or control many times, and
    various measures of covariate balance are computed from the one matched sample and
    the many randomized experiments. The main elements of the method are from Hansen and
    Bowers (2008), Pimentel et al. (2015, Table 1), and Yu (2021).

    Parameters
    ----------
    z : array-like
        z is a vector, with z[i]=1 for treated and z[i]=0 for control.
    x : array-like
        x is a matrix or a data-frame containing covariates with no NAs. An error will
        result if length(z) does not equal the number of rows of x.
    statistic : str
        If statistic="t", the default two-sample t-test from the 'stats' package
        computes a two-sided P-value for each covariate in the matched sample and the
        many randomized experiments. If statistic="w", then the two-sample Wilcoxon rank
        sum test is used, as implemented in the 'stats' package. If statistic="s", the
        two-sample Wilcoxon rank sum test is used for numeric covariates with more than
        nunique distinct values, or the chi-square test for a two-way table is used for
        factors and for numeric covariates with at most nunique distinct values. The
        default value is nunique=2.
    reps : int
        A positive integer. A total of reps randomized experiments are compared to the
        one matched sample.
    trunc : float
        For each simulated randomized experiment, a P-value is computed for each
        covariate. Also computed is the statistic proposed by Zykin et al. (2002)
        defined as the product of those covariate-specific P-values that do not exceed
        trunc. This truncated product is not a P-value, but it is a statistic. See
        Details.
    nunique : int
        For each simulated randomized experiment, a P-value is computed for each
        covariate. Also computed is the statistic proposed by Zykin et al. (2002)
        defined as the product of those covariate-specific P-values that do not exceed
        trunc. This truncated product is not a P-value, but it is a statistic. See
        Details.
    alpha : float
        For each simulated randomized experiment, a P-value is computed for each
        covariate. Also computed is number of these P-values that are less than or equal
        to alpha.

    Returns
    -------
     test_name
        The name of the test used.
     actual
        For each covariate, the usual two sample P-values comparing the distributions of
        treated and control groups for each covariate in x. Also, the minimum P-value,
        the truncated product of P-values, and the number of P-values less than or equal
        to alpha -- none of these quantities is a P-value.
     simBetter
        Comparison of the covariate imbalance in the actual matched sample and the many
        simulated randomized experiment. Of the reps randomized experiments, how many
        were strictly better balanced than the one matched observational study.
     sim
        Details of the simulated randomized experiments.

    Notes
    -----
    Truncated Product: For independent uniform P-values, Zaykin et al. (2002) derive a
    true P-value from the null distribution of their truncated product of P-values. That
    null distribution can be computed using the truncatedP() function in the
    'sensitivitymv' package; however, it is not used here, because the P-values for
    dependent covariates are not independent. Rather, the actual randomization
    distribution of the truncated product is simulated. Taking trunc=1 yields Fisher's
    statistic, namely the product of all of the P-values.

    Examples
    --------
    # Evaluate the balance in the bingeM matched sample.
    # The more difficult control group, P, will be evaluated.
    data(bingeM)
    attach(bingeM)
    xBP<-data.frame(age,female,education,bmi,waisthip,vigor,smokenow,bpRX,smokeQuit)
    xBP<-xBP[bingeM$AlcGroup!="N",]
    detach(bingeM)
    z<-bingeM$z[bingeM$AlcGroup!="N"]

    # In a serious evaluation, take reps=1000 or reps=10000.
    # For a quick example, reps is set to reps=100 here.
    set.seed(5)
    balBP<-evalBal(z,xBP,reps=100)
    balBP$test.name
    # This says that age is compared using the Wilcoxon two-sample test,
    # and female is compared using the chi-square test for a 2x2 table.
    # Because the default, nunique=2, was used, education was evaluated
    # using Wilcoxon's test; however, changing nunique to 5 would evaluate
    # the 5 levels of education using a chi-square test for a 2x5 table.
    balBP$actual
    # In the matched sample, none of the 9 covariates has a P-value
    # of 0.05 or less.  The smallest of the 9 P-values is .366, and
    # their truncated product is 1, because, by definition, the truncated
    # product is 1 if all of the P-values are above trunc.
    apply(balBP$sim,2,median)
    # In the simulated randomized experiments, the median of the 100
    # P-values is close to 1/2 for all covariates.
    balBP$simBetter
    # Of the 100 simulated randomized experiments, only 3 were better
    # balanced than the matched sample in terms of the minimum P-value,
    # and none were better balanced in terms of the truncated product
    # of P-values.
    #
    # There were too few controls in the P control group who smoked
    # on somedays to match exactly for smokenow.  Nonetheless, only
    # 13/100 randomized experiments were better balanced for smokenow.
    #
    # Now compare the binge group B to the combination of the two
    # control groups.
    attach(bingeM)
    x<-data.frame(age,female,education,bmi,waisthip,vigor,smokenow,bpRX,smokeQuit)
    detach(bingeM)
    set.seed(5)
    balAll<-evalBal(bingeM$z,x,reps=100,trunc=1)
    balAll$actual
    balAll$simBetter
    # This time, Fisher's product of all P-values is used, with trunc=1.
    # In terms of the minimum P-value and the product of P-values,
    # none of the 100 randomized experiments is better balanced than the
    # matched sample.

    References
    ----------
    Hansen, B. B., and Bowers, J. (2008) <doi:10.1214/08-STS254> Covariate balance in
    simple, stratified and clustered comparative studies. Statistical Science, 23,
    219-236.

    Pimentel, S. D., Kelz, R. R., Silber, J. H. and Rosenbaum, P. R. (2015)
    <doi:10.1080/01621459.2014.997879> Large, sparse optimal matching with refined
    covariate balance in an observational study of the health outcomes produced by new
    surgeons. Journal of the American Statistical Association, 110, 515-527.

    Yu, R. (2021) <doi:10.1111/biom.13098> Evaluating and improving a matched comparison
    of antidepressants and bone density. Biometrics, 77(4), 1276-1288.

    Zaykin, D. V., Zhivotovsky, L. A., Westfall, P. H. and Weir, B. S. (2002)
    <doi:10.1002/gepi.0042> Truncated product method of combining P-values. Genetic
    Epidemiology, 22, 170-185.

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
    """Sensitivity Analysis Using Noether's Test for Matched Pairs.

    Computes a sensitivity analysis for treated-minus-control matched pair differences
    in observational studies.

    Parameters
    ----------
    y : array-like
        A vector of treated-minus-control matched pair differences.
    f : float
        A nonnegative number strictly less than 1. Suppose that there are I matched pair
        differences, length(y)=I. Rank the absolute pair differences from 1 to I with
        average ranks for ties. Noether's statistic looks at the roughly (1-f)I pair
        differences with absolute ranks that are at least fI, and computes the sign test
        from these fI pair differences. With f=0, Noether's statistic is the usual sign
        test statistic. With f=2/3, Noether's statistic focuses on the 1/3 of pairs with
        the largest absolute pair differences. In his article, Noether suggested f=1/3
        for randomized matched pair differences from a Normal distribution, but f=2/3 is
        better for sensitivity analyses in observational studies. Pair differences that
        are zero are not counted, but this is uncommon for f=2/3.
    gamma : float
        A number greater than or equal to 1. gamma is the sensitivity parameter, where
        gamma=1 for a randomization test, and gamma>1 for a sensitivity analysis.
    alternative : str
        The possible alternatives are "greater", "less" or "two.sided"; however,
        "two.sided" is available only for gamma=1.

    Returns
    -------
     number_pairs : int
        Number of pairs used by Noether's statistic, roughly fI.
     positive_pairs : int
        Number of positive pair differences among used pairs.
     pval
        P-value testing the null hypothesis of no treatment effect. Obtained from the
        binomial distribution.

    Notes
    -----
    Noether's (1973) strengthens the sign test. In a randomized experiment, it increase
    power. In an observational study, it increases design sensitivity and the Bahadur
    efficiency of a sensitivity analysis.

    Because the test has a binomial null distribution in both a randomized experiment
    and in an observational study, Noether's test is used in a number of problems in
    Introduction to the Theory of Observational Studies.

    Noether's test is related to methods of Gastwirth (1966), Brown (1981), and
    Markowski and Hettmansperger (1982). Its properties in an observational study are
    discussed Rosenbaum (2012, 2015).

    As noted in the Preface to Introduction to the Theory of Observational Studies,
    Noether's statistic is used in a sequence of Problems that appear in various
    chapters.

    Examples
    --------
    set.seed(1)
    y<-rnorm(1000)+.5
    noether(y,f=0,gamma=3)
    noether(y,f=2/3,gamma=3)

    References
    ----------
    Brown, B. M. (1981) <doi:10.1093/biomet/68.1.235> Symmetric quantile averages and
    related estimators. Biometrika, 68(1), 235-242.

    Gastwirth, J. L. (1966) <doi:10.1080/01621459.1966.10482185> On robust procedures.
    Journal of the American Statistical Association, 61(316), 929-948.

    Markowski, E. P. and Hettmansperger, T. P. (1982)
    <doi:10.1080/01621459.1982.10477905> Inference based on simple rank step score
    statistics for the location model. Journal of the American Statistical Association,
    77(380), 901-907.

    Noether, G. E. (1973) <doi:10.1080/01621459.1973.10481411> Some simple
    distribution-free confidence intervals for the center of a symmetric distribution.
    Journal of the American Statistical Association, 68(343), 716-719.

    Rosenbaum, P. R. (2012) <10.1214/11-AOAS508> An exact adaptive test with superior
    design sensitivity in an observational study of treatments for ovarian cancer.
    Annals of Applied Statistics, 6, 83-105.

    Rosenbaum, P. R. (2015) <doi:10.1080/01621459.2014.960968> Bahadur efficiency of
    sensitivity analyses in observational studies. Journal of the American Statistical
    Association, 110(509), 205-217.

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
    """Initialize a Distance Matrix.

    Creates an distance matrix of zeros of dimensions compatible with the treatment
    indicator vector z.

    Parameters
    ----------
    z : array-like
        A vector with z[i]=1 if individual i is treated or z[i]=0 if individual i is
        control. The rows of costmatrix refer to treated individuals and the columns
        refer to controls. Although not strictly required, it is best that z has names
        that are the same as the names of the data frame dat that will be used in
        matching.

    Returns
    -------
    cost_matrix : ndarray
        A matrix of zeros with sum(z) rows and sum(1-z) columns. If z has names, then
        they become the row and column names of this matrix.

    Examples
    --------
    data(binge)
    # Select two treated and three controls from binge
    d<-binge[is.element(binge$SEQN,c(109315,109365,109266,109273,109290)),]
    z<-1*(d$AlcGroup=="B")
    names(z)<-d$SEQN
    dist<-startcost(z)
    dist
    rm(z,dist,d)

    """
    z = np.asarray(z)
    treated = np.sum(z == 1)
    control = np.sum(z == 0)
    return np.zeros((treated, control))


def ev(sc, z, m, g, method):
    r"""Compute the null expectation and variance for one stratum.

    Of limited interest to most users, the ev function plays an internal role in
    2-sample and stratified sensitivity analyses. The expectation and variance returned
    by the ev function are defined in the third paragraph of section 4, page 495, of
    Rosenbaum and Krieger (1990).

    Parameters
    ----------
    sc : array-like
        A vector of scored outcomes for one stratum. For instance, for Wilcoxon's rank
        sum test, these would be the ranks of the outcomes in the current stratum.
    z : array-like
        Treatment indicators, with length(z)=length(sc). Here, z[i]=1 if i is treated
        and z[i]=0 if i is control.
    m : int
        The unobserved covariate u has length(z)-m 0's followed by m 1's.
    g : float
        The sensitivity parameter \eqn{\Gamma}, where \eqn{\Gamma \ge 1}.
    method : str
        If method="RK" or if method="BU", exact expectations and variances are used in a
        large sample approximation. Methods "RK" and "BU" should give the same answer,
        but "RK" uses formulas from Rosenbaum and Krieger (1990), while "BU" obtains
        exact moments for the extended hypergeometric distribution using the BiasedUrn
        package and then applies Proposition 20, page 155, section 4.7.4 of Rosenbaum
        (2002). In contrast, method="LS" does not use exact expectations and variances,
        but rather uses the large sample approximations in section 4.6.4 of Rosenbaum
        (2002). Finally, method="AD" uses method="LS" for large strata and method="BU"
        for smaller strata.

    Returns
    -------
     expect : float
        Null expectation of the test statistic.
     vari : float
        Null variance of the test statistic.

    Notes
    -----
    The function ev() is called by the function evall(). The ev() function is from the
    senstrat package.

    Examples
    --------
    ev(1:5,c(0,1,0,1,0),3,2,"RK")
    ev(1:5,c(0,1,0,1,0),3,2,"BU")

    References
    ----------
    Rosenbaum, P. R. and Krieger, A. M. (1990) <doi:10.2307/2289789> Sensitivity of
    two-sample permutation inferences in observational studies. Journal of the American
    Statistical Association, 85, 493-498.

    Rosenbaum, P. R. (2002). Observational Studies (2nd edition). New York: Springer.
    Section 4.6.

    Rosenbaum, P. R. (2018) <doi:10.1214/18-AOAS1153> Sensitivity analysis for
    stratified comparisons in an observational study of the effect of smoking on
    homocysteine levels. The Annals of Applied Statistics, 12(4), 2312-2334.

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
    r"""Compute expectations and variances for one stratum.

    Of limited interest to most users, the evall() function plays an internal role in
    2-sample and stratified sensitivity analyses. The expectation and variance returned
    by the evall() function are defined in the third paragraph of section 4, page 495,
    of Rosenbaum and Krieger (1990). The function evall() calls the function ev() to
    determine the expectation and variance of the test statistic for an unobserved
    covariate u with length(z)-m 0's followed by m 1's, doing this for
    m=1,...,length(z)-1.

    Parameters
    ----------
    sc : array-like
        A vector of scored outcomes for one stratum. For instance, for Wilcoxon's rank
        sum test, these would be the ranks of the outcomes in the current stratum.
    z : array-like
        Treatment indicators, with length(z)=length(sc). Here, z[i]=1 if i is treated
        and z[i]=0 if i is control.
    g : float
        The sensitivity parameter \eqn{\Gamma}, where \eqn{\Gamma \ge 1}.
    method : str
        If method="RK" or if method="BU", exact expectations and variances are used in a
        large sample approximation. Methods "RK" and "BU" should give the same answer,
        but "RK" uses formulas from Rosenbaum and Krieger (1990), while "BU" obtains
        exact moments for the extended hypergeometric distribution using the BiasedUrn
        package and then applies Proposition 20, page 155, section 4.7.4 of Rosenbaum
        (2002). In contrast, method="LS" does not use exact expectations and variances,
        but rather uses the large sample approximations in section 4.6.4 of Rosenbaum
        (2002). Finally, method="AD" uses method="LS" for large strata and method="BU"
        for smaller strata.

    Returns
    -------
    A data.frame with length(z)-1 rows and three columns. The first column, m, gives the
    number of 1's in the unobserved covariate vector, u. The second column, expect, and
    the third column, var, give the expectation and variance of the test statistic for
    this u.

    Notes
    -----
    The evall() function is called by the sen2sample() function and the senstrat()
    function.

    The example is from Table 1, page 497, of Rosenbaum and Krieger (1990). The example
    is also Table 4.15, page 146, in Rosenbaum (2002). The example refers to Cu cells.
    The data are orignally from Skerfving et al. (1974). The evall function is from the
    senstrat package.

    Examples
    --------
    z<-c(rep(0,16),rep(1,23))
    CuCells<-c(2.7, .5, 0, 0, 5, 0, 0, 1.3, 0, 1.8, 0, 0, 1.0, 1.8,
               0, 3.1, .7, 4.6, 0, 1.7, 5.2, 0, 5, 9.5, 2, 3, 1, 3.5,
               2, 5, 5.5, 2, 3, 4, 0, 2, 2.2, 0, 2)
    evall(rank(CuCells),z,2,"RK")

    References
    ----------
    Rosenbaum, P. R. and Krieger, A. M. (1990) <doi:10.2307/2289789> Sensitivity of
    two-sample permutation inferences in observational studies. Journal of the American
    Statistical Association, 85, 493-498.

    Rosenbaum, P. R. (2002). Observational Studies (2nd edition). New York: Springer.
    Section 4.6.

    Rosenbaum, P. R. (2018) <doi:10.1214/18-AOAS1153> Sensitivity analysis for
    stratified comparisons in an observational study of the effect of smoking on
    homocysteine levels. The Annals of Applied Statistics, 12(4), 2312-2334.

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
