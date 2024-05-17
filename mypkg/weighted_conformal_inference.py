import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from joblib import Parallel, delayed
from utils.utils import _set_verbose_level, _update_params
from utils.misc import merge_intervals
from utils.stats import weighted_quantile, smoothed_weighted_quantile
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


if not logger.hasHandlers():
    ch = logging.StreamHandler() # for console. 
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch) 

class WeightedConformalInference():
    def __init__(self, cal_X, cal_Y, gen_fn, gen_type="ddpm", 
                 ws_fn=None, verbose=2, inf_bs=40, seed=1, n_jobs=1, device="cpu", wcf_params={}, 
                 gen_params={}):
        """
        args:
            cal_X (tensor, n_cal x d):  The covariates of the calibration set. 
            cal_Y (tensor, n_cal):  The response of the calibration set. 
            gen_fn (fn): A generating function f(Y|X), given X, it can produce predicted Y.
            inf_bs (int): The inference batch size, for DDIM or DDPM, 40 is a good number; 
                            for other gen_type like MLP, it can be as large as 512 (but now not used)
            ws_fn (fn): A weights function to assign weight to each obs based on X. If None, all weights are 1.
                        Only need it when we conduct causal inference.
        """
        _set_verbose_level(verbose, logger)
        wcf_params_def = edict({
            "K": 40, # num of sps for each X
            "nwhigh" : 20,
            "nwlow" : 0.05,
            "useinf" :False, 
            "cf_type": "PCP", 
            "smooth_qv": False
        })
        if gen_type.lower().startswith("ddpm"):
            gen_parmas_def = edict({
            })
        elif gen_type.lower().startswith("ddim"):
            gen_parmas_def = edict({
                "ddim_timesteps": 50, 
                "ddim_eta": 0
            })
        elif gen_type.lower().startswith("reg"):
            gen_parmas_def = edict({
            })
            wcf_params_def["cf_type"] = "naive"
        
        
        wcf_params = _update_params(wcf_params, wcf_params_def, logger)
        gen_params = _update_params(gen_params, gen_parmas_def, logger)
        gen_params["gen_type"] = gen_type.lower()
        logger.info(f"wcf params is {wcf_params}")
        logger.info(f"gen params is {gen_params}")
        
        
        if ws_fn is None:
            def _ws_fn(x):
                if x.ndim == 1:
                    x = x.reshape(1, -1)
                return torch.ones(x.shape[0])
            ws_fn = _ws_fn
            
        if gen_params.gen_type.startswith("ddpm"):
            def _gen_fn(X):
                return gen_fn.sample(X, device=device)
        elif gen_params.gen_type.startswith("ddim"):
            def _gen_fn(X):
                return gen_fn.sample_ddim(X, 
                                          ddim_timesteps=gen_params.ddim_timesteps,
                                          ddim_eta=gen_params.ddim_eta,
                                          device=device)
        elif gen_params.gen_type.startswith("reg"):
            def _gen_fn(X):
                with torch.no_grad():
                    gen_fn.eval()
                    Y = gen_fn(X)
                return Y.to(device).squeeze()
            
        else:
            raise NotImplementedError(f'No supported.')
        
        
        self.cal_X = cal_X
        self.cal_Y = cal_Y
        self.ws_fn = ws_fn
        self._gen_fn = _gen_fn
        self.verbose = verbose
        self.wcf_params = wcf_params
        self.gen_params = gen_params
        self.seed = seed
        self.n_jobs = n_jobs
        self.inf_bs = inf_bs
        self.df_dtype = torch.get_default_dtype()
        self.device = device
        
        self.Es = None
        self.ws = None

        
    def gen_fn_wrapper(self, X, nsps=1, inf_bs=40, seed=0, n_jobs=1):
        """A simple wrapper of original gen_fn function
        args:
            X (tensor, d or n x d):  The covariates
            nsps (int): The number of sps you want to draw for each x in X.
            inf_bs (int): The inference batch size, how many x's you do sample for each time. 
                          So the total batch is nsps x inf_bs
            seed (int): sampling seed
        return: 
            Y (tensor, n x nsps): The samples from X
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        num_iters = int(np.ceil(X.shape[0]/inf_bs));
        
        if self.verbose >= 2:
            pbar = tqdm(range(num_iters), total=num_iters)
        else:
            pbar = range(num_iters)
            
        def _sps_fn(ix):
            torch.set_default_dtype(self.df_dtype)
            torch.set_default_device(self.device)
            torch.manual_seed(seed)
            X_cur = X[(ix*inf_bs):(ix*inf_bs+inf_bs)]
            X_cur_mul = X_cur.repeat(nsps, 1);
            with torch.no_grad():
                Y_cur, _ = self._gen_fn(X_cur_mul)
            Y_cur = Y_cur.reshape(-1);
            Y_cur = Y_cur.reshape(nsps, -1);
            return Y_cur
        with Parallel(n_jobs=n_jobs) as parallel:
            Ys = parallel(delayed(_sps_fn)(ix) for ix in pbar)
        Ys = torch.concat(Ys, axis=1).T;
        return Ys
    
    def PCP_dist_fn(self, K=40):
        cal_Y_preds = self.gen_fn_wrapper(self.cal_X, nsps=self.wcf_params.K, 
                                          inf_bs=self.inf_bs, 
                                          seed=self.seed, 
                                          n_jobs=self.n_jobs)
        Es = torch.abs(cal_Y_preds-self.cal_Y[:, None]).min(axis=1)[0];
        return Es
    
    def naive_dist_fn(self):
        assert not self.gen_params.gen_type.startswith("ddpm")
        assert not self.gen_params.gen_type.startswith("ddim")
        cal_Y_preds = self._gen_fn(self.cal_X)
        Es = torch.abs(cal_Y_preds.squeeze()-self.cal_Y);
        return Es
    
    def PCP_intv_fn(self, Y_hat, qvs):
        intvs_raw = torch.stack([Y_hat - qvs[:, None], Y_hat + qvs[:, None]]);
        intvs_raw.transpose_(1, 0);
        intvs_raw.transpose_(1, 2);
        intvs = [merge_intervals(intv_raw.numpy()) for intv_raw in intvs_raw]
        return intvs
    
    def naive_intv_fn(self, Y_hat, qvs):
        intvs_raw = torch.stack([Y_hat-qvs, Y_hat+qvs]).T
        intvs_raw = list(intvs_raw.numpy())
        intvs = [intv[None] for intv in intvs_raw]
        return intvs
    def get_intvs(self, Y_hat, qvs):
        if self.wcf_params.cf_type.startswith("PCP"):
            return self.PCP_intv_fn(Y_hat, qvs)
        else:
            return self.naive_intv_fn(Y_hat, qvs)
    
    def get_dist(self):
        if self.wcf_params.cf_type.startswith("PCP"):
            self.Es = self.PCP_dist_fn(K=self.wcf_params.K)
        elif self.wcf_params.cf_type.startswith("naive"):
            self.Es = self.naive_dist_fn()
        else:
            pass
        return self.Es
    
    def get_weights(self):
        self.ws = self.ws_fn(self.cal_X)
        return self.ws 
    
    
    def __call__(self, X, alpha=0.05):
        """Do conformal inference
        args:
            X (tensor, d or n x d):  The covariates.
            alpha (float): The output CI is (1-alpha) coverage
        """
        if self.ws is None:
            self.get_weights()
        if self.Es is None:
            self.get_dist()
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
        X_ws = self.ws_fn(X)
        # num of test x (num_cal+1)
        ws_wtest = torch.concat([self.ws[None, :].repeat(X_ws.shape[0], 1), X_ws[:, None]], axis=1)
        nws_wtest = ws_wtest/ws_wtest.mean(axis=1, keepdims=True);
        nws_wtest[nws_wtest<self.wcf_params.nwlow] = self.wcf_params.nwlow
        nws_wtest[nws_wtest>self.wcf_params.nwhigh] = self.wcf_params.nwhigh
        
        
        qvs = []
        if self.wcf_params.smooth_qv:
            np.random.seed(self.seed)
            Es_wi = torch.cat([self.Es, torch.tensor([torch.inf], dtype=self.df_dtype, device=self.device)])
            for t_idx in range(len(X_ws)):
                qv = smoothed_weighted_quantile(values=Es_wi.cpu().numpy(), 
                                                sample_weight=nws_wtest[t_idx, :].numpy(), 
                                                alpha=alpha)[0]
                if qv == np.inf:
                    qv = self.Es.max().item()
                elif qv == -np.inf:
                    qv = 0
                qvs.append(qv)
        else:
            qts = (1 + nws_wtest[:, -1]/ nws_wtest[:, :-1].sum(axis=1)) * (1 - alpha)
            # now, I use a loop, but it can be improved
            for t_idx, qt in enumerate(qts):
                if qt <= 1:
                    qv = weighted_quantile(self.Es, [qt], sample_weight=nws_wtest[t_idx, :-1])[0];
                else:
                    if self.wcf_params.useinf:
                        qv = np.inf
                    else:
                        qv = self.Es.max()
                qvs.append(qv)
        qvs = torch.tensor(qvs, dtype=self.df_dtype, device=self.device)
            
        intvs = self.get_intvs(Y_hat, qvs);
        return intvs
        
        