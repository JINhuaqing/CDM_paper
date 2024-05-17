import numpy as np
import scipy.stats as ss
from rpy2 import robjects as robj

def gen_momsps(n, nu=2):
    """
    Generate random samples from a moment (nonlocal) distributio
    Parameters:
    - n (int): The number of samples to generate.
    - nu (>=1, float, optional): The shape parameter of the gamma distribution. Default is 2
    Returns:
    - sps (numpy.ndarray): An array of random samples from the mixture distribution.
    """
    r = robj.r
    r("""
            rmomprior <- function(n, nu=2){
                shape.alpha <- nu+0.5
                rate.beta <- 0.5
                samples <- rgamma(n, shape=shape.alpha, rate=rate.beta)
                bidata <- rbinom(n, 1, 0.5)
                samples <- samples ** 0.5
                samples[bidata==0] <- -samples[bidata==0] 
                return(samples)
            }
         """
    )
    sps = np.array(r["rmomprior"](n, nu))
    return sps



def get_pdf_at_xx(Y1, Y2):
    """
    Calculate the probability density function (PDF) for two sets of data, Y1 and Y2, at a given range of values.

    Parameters:
    Y1 (array-like): First set of data.
    Y2 (array-like): Second set of data.

    Returns:
    xx (ndarray): Array of values within the range of Y1 and Y2.
    p_pdf1 (ndarray): PDF values for Y1 at each value in xx.
    p_pdf2 (ndarray): PDF values for Y2 at each value in xx.
    """
    pdf1 = ss.gaussian_kde(Y1)
    pdf2 = ss.gaussian_kde(Y2)
    l1, u1 = np.quantile(Y1, [0.01, 0.99])
    l2, u2 = np.quantile(Y2, [0.01, 0.99])
    xmin = min(l1, l2)
    xmax = max(u1, u2)
    xx = np.linspace(xmin, xmax, 100)
    p_pdf1 = pdf1(xx);
    p_pdf2 = pdf2(xx);
    return xx, p_pdf1, p_pdf2
def get_kl(Y1, Y2):
    """
    Calculate the Kullback-Leibler (KL) divergence between two data spses.

    Parameters:
    Y1 (array-like): The first data sps.
    Y2 (array-like): The second sps .

    Returns:
    tuple: A tuple containing the KL divergence from Y1 to Y2 and the KL divergence from Y2 to Y1.
    """
    xx, p1, p2 = get_pdf_at_xx(Y1, Y2)
    kl_div_12 = np.sum(np.where(p1 != 0, p1 * np.log(p1 / p2), 0))
    kl_div_21 = np.sum(np.where(p2 != 0, p2 * np.log(p2 / p1), 0))
    return kl_div_12, kl_div_21


# from https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
def weighted_quantile(values, quantiles, sample_weight=None, 
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


def smoothed_weighted_quantile(values, sample_weight, alpha):
    """Get smoothed weighted quantile based on 
        https://github.com/rohanhore/RLCP/blob/main/utils/methods.R
        args:
            values: the values for getting quantile
            sample_weight: The corresponding weights
            alpha: sig level, return (1-alpha) quantile
    """
    sort_idx = np.argsort(values)
    values_sorted = values[sort_idx]
    sample_weight_sorted = sample_weight[sort_idx]
    
    ws = sample_weight_sorted/sample_weight_sorted.sum()
    
    # get unique value and its indices
    values_sorted_uni = np.sort(np.unique(values_sorted))
    if len(values_sorted_uni) == len(values_sorted):
        ws_uni = ws
        inds = [np.array([ix]) for ix in range(len(values_sorted))]
    else:
        ws_uni = np.zeros(len(values_sorted_uni))
        inds = []
        for ix, v in enumerate(values_sorted_uni):
            inds.append(np.where(values_sorted==v)[0])
            ws_uni[ix] = ws[inds[ix]].sum()
        
    u = np.random.rand(1)[0]
    ws_uni1 = ws_uni.copy()
    ws_uni1[-1] = ws_uni[-1] * u
    pvals = np.cumsum(ws_uni1[::-1])[::-1]
    
    if np.sum(pvals > alpha) > 0:
        idx = np.max(np.where(pvals > alpha))
        qv = values_sorted_uni[idx]
        if idx == (len(ws_uni1)-1):
            closed = False
        elif idx == (len(ws_uni1)-2):
            closed = np.sum(u * ws_uni[-2:]) > alpha
        else:
            closed = (np.sum(ws_uni[(idx+1):-1]) + u * (ws_uni[-1] + ws_uni[idx]))>alpha
            
    else:
        qv = -np.inf
        closed = False
    return qv, closed
