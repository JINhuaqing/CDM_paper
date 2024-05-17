import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from joblib import Parallel, delayed
from utils.utils import _set_verbose_level, _update_params
from utils.misc import merge_intervals
from utils.stats import weighted_quantile
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


if not logger.hasHandlers():
    ch = logging.StreamHandler() # for console. 
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch) 

class NaiveSample():
    def __init__(self, gen_fn, gen_type="ddpm", 
                 verbose=2, device="cpu", gen_params={}):
        """
        args:
            gen_fn (fn): A generating function f(Y|X), given X, it can produce predicted Y.
        """
        _set_verbose_level(verbose, logger)
        if gen_type.lower().startswith("ddpm"):
            gen_parmas_def = edict({
            })
        elif gen_type.lower().startswith("ddim"):
            gen_parmas_def = edict({
                "ddim_timesteps": 50, 
                "ddim_eta": 1
            })
        
        
        gen_params = _update_params(gen_params, gen_parmas_def, logger)
        gen_params["gen_type"] = gen_type.lower()
        logger.info(f"gen params is {gen_params}")
        
        
            
        if gen_params.gen_type.startswith("ddpm"):
            def _gen_fn(X):
                return gen_fn.sample(X, device=device)
        elif gen_params.gen_type.startswith("ddim"):
            def _gen_fn(X):
                return gen_fn.sample_ddim(X, 
                                          ddim_timesteps=gen_params.ddim_timesteps,
                                          ddim_eta=gen_params.ddim_eta,
                                          device=device)
        else:
            raise NotImplementedError(f'No supported.')
        
        
        self._gen_fn = _gen_fn
        self.verbose = verbose
        self.gen_params = gen_params
        self.df_dtype = torch.get_default_dtype()
        self.device = device
        
        
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
    