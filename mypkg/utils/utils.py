from torch.utils.data import Dataset
import numpy as np
from easydict import EasyDict as edict
import logging



def _set_verbose_level(verbose, logger):
    if verbose == 0:
        logger.handlers[0].setLevel(logging.ERROR)
    elif verbose == 1:
        logger.handlers[0].setLevel(logging.WARNING)
    elif verbose == 2:
        logger.handlers[0].setLevel(logging.INFO)
    elif verbose == 3:
        logger.handlers[0].setLevel(logging.DEBUG)

def _update_params(input_params, def_params, logger):
    for ky, v in input_params.items():
        if ky not in def_params.keys():
            logger.warning(f"Check your input, {ky} is not used.")
        else:
            if v is not None:
                def_params[ky] = v
    return edict(def_params)



class MyDataSet(Dataset):
    def __init__(self, Y, X):
        
        assert Y.shape[0] == X.shape[0], "X and Y should have same length"
        self.Y = Y
        self.X = X
        
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, idx):
        y = self.Y[idx]
        x = self.X[idx]
        return x, y
    
    


def get_idx_sets(all_idxs, ratios=[0.25, 0.2]):
    """
    Splits the given indices into sets based on the provided ratios.

    Parameters:
    all_idxs (array-like): The indices to be split into sets.
    ratios (array-like, optional): The ratios for splitting the indices. Default is [0.5, 0.1].

    Returns:
    list: A list of sets containing the split indices.
    """
    assert np.sum(ratios) <= 1
    if np.sum(ratios) == 1:
        ratios = ratios[:-1]
    if not isinstance(ratios, np.ndarray):
        ratios = np.array(ratios)
    n_els = len(all_idxs) * ratios
    n_els = n_els.astype(int)
    
    rem_idxs = all_idxs.copy();
    sets = []
    for n_el in n_els:
        set_idxs = np.sort(np.random.choice(rem_idxs, size=n_el, replace=False));
        sets.append(set_idxs)
        rem_idxs = np.setdiff1d(rem_idxs, sets[-1]);
    sets.append(rem_idxs)    
    return sets