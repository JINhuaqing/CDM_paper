from utils.stats import weighted_quantile, smoothed_weighted_quantile
from utils.utils import _set_verbose_level, _update_params
from weighted_conformal_inference import WeightedConformalInference
import pdb

import torch
import numpy as np
from easydict import EasyDict as edict
from torch.distributions.multivariate_normal import MultivariateNormal

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    ch = logging.StreamHandler() # for console. 
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch) 
def box_kernel_dist(n, d):
    """ draw sps from box kernel dist, given X (n x d), Xtilde = X + X + h*sps
    from https://github.com/rohanhore/RLCP/blob/main/utils/methods.R (runifball)
    args:
        n (int): Num of sps
        d (int): Feature dim
    return:
        sps (n x d): n sps
    """
    zs = MultivariateNormal(loc=torch.zeros(d), 
                            covariance_matrix=torch.eye(d)).sample((n, )) # n x d
    us = torch.rand(n)[:, None]; # n x 1
    sps = us**(1/d) * zs/torch.norm(zs, p=2, dim=-1).unsqueeze(-1)
    return sps

def box_kernel(X1s, X2s, h, vec=False):
    """
        calculate box kernel between X1s and X2s, 
    args:
        X1s (tensor): n1 x d
        X2s (tensor): n2 x d
        h (float): bw
    returns: 
        a matrix of n2 x n1 or n1 (when vec=True and n1=n2)
    """
    if X1s.ndim == 1:
        X1s = X1s[None]
    if X2s.ndim == 1:
        X2s = X2s[None]
    if vec:
        assert X1s.shape[0] - X2s.shape[0] == 0
        diff = X1s - X2s
    else:
        diff = X1s[None] - X2s[:, None]
    diff_norm = torch.norm(diff, p=2, dim=-1)
    vs = (diff_norm <= h).to(X1s.dtype)
    return vs
    
def log_norm_kernel(X1s, X2s, h, vec=False):
    """
        calculate kernel between X1s and X2s, 
    args:
        X1s (tensor): n1 x d
        X2s (tensor): n2 x d
        h (float): bw
    returns: 
        a matrix of n2 x n1 or n1 (when vec=True and n1=n2)
    """
    if X1s.ndim == 1:
        X1s = X1s[None]
    if X2s.ndim == 1:
        X2s = X2s[None]
    if vec:
        assert X1s.shape[0] - X2s.shape[0] == 0
        diff = X1s - X2s
    else:
        diff = X1s[None] - X2s[:, None]
    return -torch.norm(diff, p=2, dim=-1)**2/2/h/h
    #return -torch.norm(diff, p=2, dim=-1)**2/2/h/h/X1s.shape[-1]

class LocalWeightedConformalInference(WeightedConformalInference):
    def add_data(self, X):
        """
        args:
            X (tensor, d or n x d):  The covariates of the test point 
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.gen_params.gen_type.startswith("reg"):
            Y_hat = self._gen_fn(X)
        else:
            Y_hat = self.gen_fn_wrapper(X, 
                                    nsps=self.wcf_params.K, 
                                    inf_bs=self.inf_bs, 
                                    seed=self.seed, 
                                    n_jobs=self.n_jobs)
        
        self.test_X = X
        self.test_X_ws = self.ws_fn(X) 
        self.test_Y_hat = Y_hat
    def __call__(self, local_method=None, alpha=0.05, lm_params={}):
        """Do local conformal inference (consider the test and train cov shift)
        args:
            local_method: the local_method to use, baseLCP, calLCP, RLCP or None
                        "g-baseLCP": baseLCP with gaussian kernel
                        "b-baseLCP": baseLCP with box kernel
                        if None, do not LCP but do regular CP
            alpha (float): The output CI is (1-alpha) coverage
        """
        if local_method is not None:
            local_method = local_method.lower()
            #assert local_method in ["baselcp", "rlcp"], f"{local_method} is not supported now."
        if local_method is not None:
            lm_params_def = edict({
                "h": 1
            })
        else:
            lm_params_def = edict({})
        lm_params = _update_params(lm_params, lm_params_def, logger)
            
        # ws is the base ws for causal not for test and train cov shift
        if self.ws is None:
            self.get_weights()
        if self.Es is None:
            self.get_dist()
            
        if local_method is None:
            add_ws = torch.ones(self.test_X.shape[0], self.cal_X.shape[0]+1) # num_test x (num_cal+1)
        elif local_method.endswith("lcp"):
            h = lm_params["h"]
            if local_method.startswith("g-rlcp"):
                cov_mat = torch.eye(self.test_X.shape[1])*h*h
                sps = MultivariateNormal(loc=torch.zeros(self.test_X.shape[1]), covariance_matrix=cov_mat).sample((self.test_X.shape[0], ))
                sps = self.test_X + sps
            elif local_method.startswith("b-rlcp"):
                sps = h*box_kernel_dist(n=self.test_X.shape[0], d=self.test_X.shape[1])
                sps = self.test_X + sps
            elif local_method.endswith("baselcp"):
                sps = self.test_X
            if local_method.startswith("g"):
                add_logws1 = log_norm_kernel(self.cal_X, sps, h=h)
                add_logws2 = log_norm_kernel(self.test_X, sps, h=h, vec=True)
                add_logws = torch.cat([add_logws1, add_logws2[:, None]], dim=1)
                # remove the mean, otherwise, when d is large, exp(logws) -> 0
                #shift_value = (add_logws.max(dim=1, keepdims=True)[0] + add_logws.min(dim=1, keepdims=True)[0])/2
                shift_value = add_logws.max(dim=1, keepdims=True)[0]
                add_logws = add_logws - shift_value
                add_ws = add_logws.exp()
            elif local_method.startswith("b"):
                add_ws1 = box_kernel(self.cal_X, sps, h=h)
                add_ws2 = box_kernel(self.test_X, sps, h=h, vec=True)
                add_ws = torch.cat([add_ws1, add_ws2[:, None]], dim=1)
        nadd_ws=add_ws/add_ws.mean(dim=1, keepdims=True);
            
            
        # num of test x (num_cal+1)
        # ws for causal inference
        ws_wtest = torch.concat([self.ws[None, :].repeat(self.test_X_ws.shape[0], 1), self.test_X_ws[:, None]], axis=1)
        nws_wtest = ws_wtest/ws_wtest.mean(axis=1, keepdims=True);
        
        tws_wtest = nws_wtest * nadd_ws
        tws_wtest = tws_wtest/tws_wtest.mean(axis=1, keepdims=True);
        
        tws_wtest[tws_wtest<self.wcf_params.nwlow] = self.wcf_params.nwlow
        tws_wtest[tws_wtest>self.wcf_params.nwhigh] = self.wcf_params.nwhigh
        
        # now, I use a loop, but it can be improved
        qvs = []
        
        if self.wcf_params.smooth_qv:
            np.random.seed(self.seed)
            Es_wi = torch.cat([self.Es, torch.tensor([torch.inf], dtype=self.df_dtype, device=self.device)])
            for t_idx in range(len(self.test_X)):
                qv = smoothed_weighted_quantile(values=Es_wi.cpu().numpy(), 
                                                sample_weight=tws_wtest[t_idx, :].numpy(), 
                                                alpha=alpha)[0]
                if qv == np.inf:
                    qv = self.Es.max().item()
                elif qv == -np.inf:
                    qv = 0
                qvs.append(qv)
        else:
            qts = (1 + tws_wtest[:, -1]/ tws_wtest[:, :-1].sum(axis=1)) * (1 - alpha)
            for t_idx, qt in enumerate(qts):
                if qt <= 1:
                    qv = weighted_quantile(self.Es, [qt], sample_weight=tws_wtest[t_idx, :-1])[0];
                else:
                    if self.wcf_params.useinf:
                        qv = np.inf
                    else:
                        qv = self.Es.max()
                qvs.append(qv)
            #pdb.set_trace()
        qvs = torch.tensor(qvs, dtype=self.df_dtype, device=self.device)
            
        intvs = self.get_intvs(self.test_Y_hat, qvs);
        #pdb.set_trace()
        return intvs
        
        
    
def get_opth(X, hmin=0.1, local_method="RLCP", target_eff_size=None):
    """Get opt h such that h is smallest one reaching target effective size.
    args:
        X (tensor): n x d, the dataset to get h
        hmin (float): min value of h,
        local_method (str): local method, can be RLCP, baseLCP and calLCP
        target_eff_size(float): the target eff size
    """
    if target_eff_size is None:
        target_eff_size = int(X.shape[0]/2)
    def binary_search(xs, cutoff, fun):
        low, high = 0, len(xs) - 1
        while low <= high:
            mid = (low + high) // 2
            value = fun(xs[mid])
            if value < cutoff:      
                low = mid + 1      
            elif mid > 0 and fun(xs[mid-1]) < cutoff:  
                return xs[mid]
            else:
                high = mid - 1    
        return None  

    local_method = local_method.lower()
    def _get_eff_size(h):
        cov_mat = torch.eye(X.shape[1]) *h*h
        if local_method.startswith("rlcp"):
            sps = MultivariateNormal(loc=torch.zeros(X.shape[1]), covariance_matrix=cov_mat).sample((X.shape[0], ))
            sps = X + sps
        else:
            sps = X
            
        add_logws = log_norm_kernel(X, sps, h=h, vec=False)
        shift_value = (add_logws.max(dim=1, keepdims=True)[0] + add_logws.min(dim=1, keepdims=True)[0])/2
        add_logws = add_logws-shift_value
        add_ws = add_logws.exp();
        nadd_ws = add_ws/add_ws.sum(dim=1, keepdims=True)
        
        eff_size = X.shape[0]/torch.norm(nadd_ws, p="fro")**2 -1
        #print(h, eff_size)
        return eff_size
    
    can_hs = np.linspace(hmin, 20, 100);
    ## in my case, binary_search is close to loop
    opt_h = binary_search(can_hs, target_eff_size, _get_eff_size)
    while (_get_eff_size(can_hs[-1]) > target_eff_size) and (opt_h is None):
        opt_h = binary_search(can_hs, target_eff_size, _get_eff_size)
    if opt_h is None:
        opt_h = can_hs[-1]
    #for can_h in can_hs:
    #    cur_eff_size = _get_eff_size(can_h)
    #    if cur_eff_size > target_eff_size:
    #        break
    #opt_h = can_h
    return opt_h
