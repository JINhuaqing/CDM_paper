import torch.nn as nn
from constants import RES_ROOT
from .mlp import MLPnet
from sklearn.linear_model import LinearRegression
import torch
import numpy as np
from easydict import EasyDict as edict
from tqdm import trange
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR 
import pdb
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    ch = logging.StreamHandler() # for console. 
    ch.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
class TrainMLP():
    def __init__(self, data_train, save_dir=None, prefix="", verbose=False, **input_params):
        super().__init__()
        if verbose:
            logger.handlers[0].setLevel(logging.INFO)
        else:
            logger.handlers[0].setLevel(logging.WARNING)
        
        if save_dir is not None:
            save_dir = RES_ROOT/save_dir
            if not save_dir.exists():
                save_dir.mkdir()
            logger.info(f"The results are saved at {save_dir}.")
        if len(prefix) > 0 and (not prefix.endswith("_")):
            prefix = prefix + "_"
        
        params = edict({
            "lr": 1e-4, 
            "batch_size": 128, 
            "device": "cpu",
            "n_infeat": 256,
            "n_downblk": 1, 
            # lr decreasing step factor
            "lr_gamma": 0.9,
            # lr decreasing step time, in epoch
            "lr_step": 1000, 
            # in epoch
            "test_intv": 100, 
            # regularization of adam
            "weight_decay": 1e-2
        })
        
        for key in input_params.keys():
            if key not in params.keys():
                logger.warning(f"{key} is not used, please check your input.")
            else:
                params[key] = input_params[key]
        
        # get the MLP
        nn_model=MLPnet(in_fs=1, 
                        n_infeat=params.n_infeat, 
                        n_context_fs=data_train[0][0].shape[0], 
                        n_downblk=params.n_downblk
                                )
        nn_model.weights_init()
        if verbose:
            nn_model.get_num_params()
        
        self.nn_model = nn_model
        self.nn_model.to(params.device);
        
        # the optim and loss for training
        self.optim = torch.optim.AdamW(self.nn_model.parameters(), lr=params.lr, weight_decay=params.weight_decay);
        self.lr_scheduler = ExponentialLR(self.optim, gamma=params.lr_gamma, verbose=verbose) 
        
        if verbose:
            logger.info(f"The params is {params}")
            
        self.data_train_loader = DataLoader(data_train, 
                                            batch_size=params.batch_size, 
                                            shuffle=True,
                                            generator=torch.Generator(device=params.device))
        self.params = params
        self.loss_mse = nn.MSELoss()
        self.dftype = torch.get_default_dtype();
        self.verbose = verbose
        self.losses = [] # loss
        self.losses_sm = [] # smoother version of loss
        self.losses_val = [] # test error
        self.prefix = prefix
        self.save_dir = save_dir
            
            
    def train(self, n_epoch, data_val=None, save_snapshot=True, early_stop=False, early_stop_dict={}):
        """args
            n_epoch: num of epochs
            data_val: a edict including
                c: bs x d array, feature
                x: bs/ bsx1, target 
        """
        assert not (save_snapshot and (self.save_dir is None)), "if you want to save model, plz provide save dir"
        if early_stop: 
            assert data_val, "If you want to to early stop, a val data must be provided."
            # control how many pts used for linear regression.
            early_stop_dict_def = {}
            early_stop_dict_def["early_stop_len"] = 50
            early_stop_dict_def["early_stop_eps"] = 5e-4
            
            for ky, v in early_stop_dict_def.items():
                if ky not in early_stop_dict.keys():
                    early_stop_dict[ky] = v
            logger.info(f"Early stop params are {early_stop_dict}")
                    
            early_stop_len = early_stop_dict["early_stop_len"]
            early_stop_eps = early_stop_dict["early_stop_eps"]
        else:
            if len(early_stop_dict) > 0:
                logger.warning(f"We do not do early stop, so any args in early_stop_dict are ignored.")
                
        if self.verbose:
            pbar = trange(n_epoch)
        else:
            pbar = range(n_epoch)
        
        if data_val is not None:
            c_test, x_test = data_val.c, data_val.x
            if not isinstance(data_val.c, torch.Tensor):
                c_test = torch.tensor(c_test, dtype=self.dftype).to(self.params.device)
            if not isinstance(data_val.x, torch.Tensor):
                x_test = torch.tensor(x_test, dtype=self.dftype).to(self.params.device)
            if x_test.ndim == 1:
                x_test = x_test[:, None]
            
        for ep in pbar:
            out_dict = {}
            self.nn_model.train()
            loss_sm = None # smoother version of loss
            for data in self.data_train_loader:
                self.optim.zero_grad()
                c, x = data # c is features, x is target
                if x.ndim == 1:
                    x = x[:, None]
                c, x = c.to(self.params.device), x.to(self.params.device)
                c, x = c.type(self.dftype), x.type(self.dftype)
                b_size = x.shape[0] # note it can be different from the batch_size
                if b_size == 1:
                    # let us ignore the batch with size 1, cause error (on Dec 8, 2023)
                    continue
                    
                x1 = self.nn_model(c);
                loss = self.loss_mse(x, x1)
                loss.backward()
                if loss_sm is None:
                    loss_sm = loss.item()
                else:
                    loss_sm = 0.95 * loss_sm + 0.05 * loss.item()
                if self.verbose:
                    pbar.set_description(f"loss: {loss_sm:.4f}")
                self.optim.step()
                self.losses.append((ep+1, loss.item()))
                self.losses_sm.append((ep+1, loss_sm))
                
            if data_val is not None:
                if (ep+1)%self.params.test_intv == 0:
                    #pdb.set_trace()
                    self.nn_model.eval()
                    with torch.no_grad():
                        loss_test = self.loss_mse(x_test, self.nn_model(c_test))
                    self.nn_model.train()
                    self.losses_val.append((ep+1, loss_test.item()))
                    
                    out_dict["val loss"] = loss_test.item()
                        
                
                    #only do earlystop when having new val loss
                    if early_stop and (len(self.losses_val)>=early_stop_len):
                        eps_val = np.array(self.losses_val)[-early_stop_len:, 0][:, None]
                        loss_val = np.array(self.losses_val)[-early_stop_len:, 1][:, None];
                        fit = LinearRegression().fit(X=eps_val, y=loss_val)
                        coef = fit.coef_[0][0]
                        out_dict["reg_coef"] = coef
                        if coef > early_stop_eps: 
                            torch.save(self.nn_model.state_dict(), self.save_dir/f"{self.prefix}mlp_epoch{ep+1}.pth")
                            logger.warning(f"Save model {self.prefix}mlp_epoch{ep+1}.pth due to eaprly stop.")
                            break
                        
                    if self.verbose:
                        pbar.set_postfix(out_dict, refresh=True)
                
                
            
            if (ep+1) % self.params.lr_step == 0:
                # change lr
                self.lr_scheduler.step()
                
    
            if save_snapshot:
                if isinstance(save_snapshot, bool):
                    save_int = n_epoch 
                else:
                    save_int = save_snapshot
                if (ep+1) % save_int == 0:
                    logger.info(f"Save model {self.prefix}mlp_epoch{ep+1}.pth.")
                    torch.save(self.nn_model.state_dict(), self.save_dir/f"{self.prefix}mlp_epoch{ep+1}.pth")
                    
    
    def get_model(self, nep):
        """Get trained model at (or closed to) epoch nep
        """
        all_models = list(self.save_dir.glob(f"{self.prefix}mlp_epoch*.pth"))
        _tmpf = lambda mp: int(mp.stem.split("epoch")[-1])
        all_models = sorted(all_models, key=_tmpf)
        neps = np.array([_tmpf(m) for m in all_models])
        model_idx = np.argmin(np.abs(neps-nep))
        
        logger.info(f"We load model {all_models[model_idx]}.")
        self.nn_model.load_state_dict(torch.load(all_models[model_idx]))
        self.nn_model.eval()
        return self.nn_model
