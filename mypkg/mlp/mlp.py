# compared to ddmp.py, here I remove the drop-context training part.
# i.e., no w for inference.
import torch
import torch.nn as nn
from tqdm import trange
import numpy as np
import pdb

def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.get_default_dtype()) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class ResidualMLPBlock(nn.Module):
    def __init__(
        self, in_fs: int, out_fs: int, is_res: bool = False, is_out:bool = False) -> None:
        super().__init__()
        '''
        standard ResNet style MLP block
        '''
        self.same_fs = in_fs==out_fs
        self.is_res = is_res
        self.layer1 = nn.Sequential(
            nn.Linear(in_fs, out_fs), 
            nn.BatchNorm1d(out_fs),
            nn.GELU(), 
        )
        if is_out:
            self.layer2 = nn.Linear(out_fs, out_fs)
        else:
            self.layer2 = nn.Sequential(
                nn.Linear(out_fs, out_fs), 
                nn.BatchNorm1d(out_fs),
                nn.GELU(), 
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        if self.is_res:
            # this adds on correct residual in case fs have increased
            if self.same_fs:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            return x2
        
class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if x.ndim == 1:
            x = x.view(-1, self.input_dim)
        return self.model(x)
    
class MLPnet(nn.Module):
    def __init__(self, in_fs, n_context_fs, n_infeat=256, n_downblk=1):
        """args:
            in_fs: the dim of target (x), default is 1
            n_context_fs: the dim of feature (c), default is d
            n_infeat: the num of features during inference, default 256
            n_downblk: the number of down blocks, default 1
        """
        super().__init__()

        self.in_fs = in_fs
        self.n_context_fs = n_context_fs

        self.contextembed = EmbedFC(n_context_fs, 2*n_infeat)
        
        down_blks =[
            ResidualMLPBlock(2*n_infeat, (1+int(ix!=(n_downblk-1)))*n_infeat, is_res=True, is_out=False)
            for ix in range(n_downblk)
        ]

        self.down = nn.Sequential(
            *down_blks
        )
        self.out = nn.Sequential(
            ResidualMLPBlock(n_infeat, in_fs, is_res=False, is_out=True)
        )
        self.dropout1 = nn.Dropout(0.5)

    def forward(self, c):
        """args:
        c: the context (feature), bs x d
        """
        # x is (noisy) image, c is context label, t is timestep, 

        # embed context, time step
        cemb = self.contextembed(c) # bs x 2*n_infeat
        down = self.down(cemb)  # add and multiply embeddings
        down = self.dropout1(down)
        out = self.out(down)
        return out
    
    def get_num_params(self):
        num = sum(p.numel() for p in self.parameters())
        print(f"The num of params is {num/1e6:.2f}m. ")
    
    def weights_init(self):
        def _weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Linear') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
        self.apply(_weights_init)
        
        
