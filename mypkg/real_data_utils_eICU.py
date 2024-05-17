import scipy.stats as ss
import numpy as np
from easydict import EasyDict as edict
from functools import partial
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import QuantileRegressor
from rpy2 import robjects as robj
from CQR import array2d2Robj
import pdb
r = robj.r
r["library"]('grf')


def log_norm_kernel(X1s, X2s, h, is_shift=True):
    """
        calculate kernel between X1s and X2s, 
    args:
        X1s (tensor): n1 x d, the X you want to evaluate
        X2s (tensor): n2 x d
        h (float): bw
    returns: 
        a matrix of n2 x n1 or n1 (when vec=True and n1=n2)
    """
    if X1s.ndim == 1:
        X1s = X1s[None]
    if X2s.ndim == 1:
        X2s = X2s[None]
    diff = X1s[:, None] - X2s[None]
    vs = -np.linalg.norm(diff, axis=-1)**2/2/h/h
    if is_shift:
        vs = vs - vs.mean(axis=1, keepdims=1)
    return vs

# estimate sigma(x) with a kernel
def _sigmaX_fn(eXs, baseXs, h, resis):
    if eXs.ndim == 1:
        eXs = eXs[None]
    logws = log_norm_kernel(eXs, baseXs, h=h).T;
    logws[logws>=50] = 50
    logws[logws<=-50] = -50
    
    ws = np.exp(logws)/np.exp(logws).sum(axis=0, keepdims=1);
    wmean = (resis[:, None] * ws).sum(axis=0);
    varvs = ((resis[:, None] - wmean[None])**2 * ws).sum(axis=0);
    return np.sqrt(varvs)

def clean_data_fn(df0,  
               Yvar = "cumulated_balance",
               Tvar = "ivfluid" ,
               rm_outlier = True, 
               rm_Xvars=[]):
    
    """
    Cleans the input dataframe by removing outliers, standardizing continuous variables,
    and selecting relevant columns for analysis.

    Parameters:
    - df0 (pandas.DataFrame): The input dataframe to be cleaned.
    - Yvar (str): The name of the target variable column.
    - Tvar (str): The name of the treatment variable column.
    - rm_outlier (bool): Flag indicating whether to remove outliers from the target variable.

    Returns:
    - clean_data (edict): A dictionary containing the cleaned dataframe and other relevant information.
        - df (pandas.DataFrame): The cleaned dataframe.
        - Yvar (str): The name of the target variable column.
        - Tvar (str): The name of the treatment variable column.
        - Xvars (list): The names of the predictor variable columns.
    """
    if isinstance(rm_Xvars, list):
        rm_Xvars = list(rm_Xvars)
    df = df0.copy()
    
    # rm val of -1
    sel_vars = [v for v in df.columns if np.sum(df[v]==-1) < 300]
    kpidxs1 = np.sum(df[sel_vars] == -1, axis=1) ==0
    kpidxs1 = np.bitwise_and(df["GCS"] != -3, kpidxs1)
    df = df[kpidxs1]
    df = df.reset_index(drop=True)
    df = df[sel_vars]
    
    Y = df[Yvar]
    if rm_outlier:
        cutv = np.quantile(np.abs(Y), [0.99])[0]
        kpidx = np.abs(Y) < cutv
        df = df[kpidx]
        df = df.reset_index(drop=True)
    T = df[Tvar]
    Y = df[Yvar];
    
    is_dis = lambda x: len(np.unique(x)) < 10
    Xvars = [v for v in df.columns if v not in [Tvar, Yvar, "patientunitstayid"]+rm_Xvars]
    Xvars_cts = [v for v in Xvars if not is_dis(df[v])]
    Xvars_dis = [v for v in Xvars if is_dis(df[v])];
    
    # std continuous vars
    #df[Xvars_cts] = df[Xvars_cts].apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=0)
    df[Xvars_cts] = df[Xvars_cts].apply(lambda x: (x-x.mean())/x.std(), axis=0)
    df[Yvar] = (Y - Y.mean())/Y.std()
    
    clean_data = edict()
    clean_data.df = df
    clean_data.Yvar = Yvar
    clean_data.Tvar = Tvar
    clean_data.Xvars = Xvars
    clean_data.Xvars_dis = Xvars_dis
    clean_data.Xvars_cts = Xvars_cts
    return clean_data
    



def get_real_data_Y1fns(Xs, Ys, Ts):
    """
    Returns a dictionary of functions that can be used to compute various values based on the input data.

    Parameters:
    - Xs (array-like): Input features.
    - Ys (array-like): Target values.
    - Ts (array-like): Treatment values.

    Returns:
    - all_fn (dict): A dictionary containing the following functions:
        - _efn: Function to compute the predicted probabilities.
        - _Y0fn: Function to compute the predicted target values for treatment value 0.
        - _CATEfn: Function to compute the predicted CATE (Conditional Average Treatment Effect).
        - _IQRT1fn: Function to compute the predicted IQR (Interquartile Range) for treatment value 1.
        - _IQRT0fn: Function to compute the predicted IQR (Interquartile Range) for treatment value 0.
    """
    # the ps function
    clf = RandomForestClassifier(max_depth=20, 
                                 criterion="entropy",
                                 n_estimators=50,
                                 random_state=0)
    clf.fit(Xs, Ts)
    def _efn(X):
        if X.ndim == 1:
            X = X[None, :]
        prb = clf.predict_proba(X)[:, 1]
        prb[prb<0.1] = 0.10
        prb[prb>0.9] = 0.90
        return prb
    
        
    # Y(1) fn
    regr = RandomForestRegressor(max_depth=20, 
                                 n_estimators=50,
                                 random_state=0)
    #regr = LinearRegression()
    regr.fit(Xs[Ts==1], Ys[Ts==1]);
    
    def _Y1fn(X):
        if X.ndim == 1:
            X = X[None, :]
        y1s = regr.predict(X)
        return y1s
    
    
    # IQR fn
    X1s, Y1s = Xs[Ts==1], Ys[Ts==1]
    reg1 = QuantileRegressor(quantile=0.25, alpha=0).fit(X1s, Y1s)
    reg2 = QuantileRegressor(quantile=0.75, alpha=0).fit(X1s, Y1s)
    def _IQRT1fn(X):
        if X.ndim == 1:
            X = X[None, :]
        p1s = reg1.predict(X)
        p2s = reg2.predict(X)
        rvs = p2s - p1s
        #rvs[rvs<=0] = 1e-10
        return rvs
            
        
    
    all_fn = edict()
    all_fn._efn = _efn
    all_fn._Y1fn = _Y1fn
    all_fn._IQRT1fn = _IQRT1fn
    all_fn.reg1 = reg1
    all_fn.reg2 = reg2
    return all_fn


def gen_real_data_eICUY1(Xs, all_fn, seed=0):
    """
    Generate a synthetic dataset for the eICU data using the given input features and functions.

    Parameters:
    - Xs: numpy.ndarray
        The input features for generating the dataset.
    - all_fn: object
        An object containing the necessary functions for generating the dataset.

    Returns:
    - dataset: object
        An object containing the generated dataset with the following attributes:
        - X: numpy.ndarray
            The input features.
        - ps: numpy.ndarray
            The propensity scores.
        - Y: numpy.ndarray
            The outcome variable.
        - Y1: numpy.ndarray
            The potential outcome variable under treatment.
        - T: numpy.ndarray
            The treatment assignment indicator.
        - tau: numpy.ndarray
            The treatment effect.

    """
    np.random.seed(seed)
    if Xs.ndim == 1:
        Xs = Xs[None, :]
    n, p = Xs.shape
    ps = all_fn._efn(Xs);
    T = (ss.uniform.rvs(size=n) < ps).astype(int)
    sds_vec = (1/1.35)*all_fn._IQRT1fn(Xs)
    Y1 = (all_fn._Y1fn(Xs)  + sds_vec*ss.norm.rvs(size=n))
    Y = Y1.copy()
    Y[T==0] = 0
    
    dataset = edict()
    dataset.X = Xs
    dataset.ps = ps
    dataset.Y = Y
    dataset.Y1 = Y1
    dataset.Y0 = np.zeros(n)
    dataset.T = T
    return dataset
