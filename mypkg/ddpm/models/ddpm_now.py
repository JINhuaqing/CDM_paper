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
    
class ContextNet(nn.Module):
    def __init__(self, in_fs, n_context_fs, n_infeat=256, n_upblk=1, n_downblk=1):
        """args:
            in_fs: the dim of target (x), default is 1
            n_context_fs: the dim of feature (c), default is d
            n_infeat: the num of features during inference, default 256
            n_upblk: the number of up blocks, default 1
            n_downblk: the number of down blocks, default 1
        """
        super().__init__()

        self.in_fs = in_fs
        self.n_context_fs = n_context_fs

        self.init_fc = EmbedFC(in_fs, n_infeat)

        self.timeembed = EmbedFC(1, 2*n_infeat)
        self.contextembed = EmbedFC(n_context_fs, 2*n_infeat)
        
        up_blks =[
            ResidualMLPBlock((int(ix!=0)+1)*n_infeat, 2*n_infeat, is_res=True, is_out=False)
            for ix in range(n_upblk)
        ]
        down_blks =[
            ResidualMLPBlock(2*n_infeat, (1+int(ix!=(n_downblk-1)))*n_infeat, is_res=True, is_out=False)
            for ix in range(n_downblk)
        ]

        self.up = nn.Sequential(
            *up_blks
        )
        self.down = nn.Sequential(
            *down_blks
        )
        self.out = nn.Sequential(
            ResidualMLPBlock(2*n_infeat, in_fs, is_res=False, is_out=True)
        )
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x, c, t):
        """args:
        x: the target, bs x 1
        c: the context (feature), bs x d
        t: the time step, bs or bs x 1
        """
        # x is (noisy) image, c is context label, t is timestep, 

        x = self.init_fc(x) # bs x n_infeat
        x = self.dropout1(x)
        up = self.up(x) # bs x 2n_infeat

        # embed context, time step
        cemb = self.contextembed(c) # bs x 2*n_infeat
        temb = self.timeembed(t) # bs x 2*n_infeat

        down = self.down(cemb*up+ temb)  # add and multiply embeddings
        down = self.dropout2(down)
        out = self.out(torch.cat((down, x), 1))
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
        
        

class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, verbose=False):
        super().__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.loss_mse = nn.MSELoss()
        self.verbose = verbose

    def get_num_params(self):
        num = sum(p.numel() for p in self.parameters())
        print(f"The num of params is {num/1e6:.2f}m. ")
    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        args:
            x: the target, (batch_size, 1) or bs
            c: the features, (batch_size, d);
        """
        if x.ndim == 1:
            x = x[:, None]

        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None] * x
            + self.sqrtmab[_ts, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        
        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T))

    def sample(self, c_i, device="cpu", is_store=False):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

        c_i = c_i.to(device)
        n_sample = c_i.shape[0]
        x_i = torch.randn(n_sample, 1).to(device)  # x_T ~ N(0, 1), sample initial noise

        if False:
        #if self.verbose:
            pbar = trange(self.n_T, 0, -1)
        else:
            pbar = range(self.n_T, 0, -1)
        x_i_store = [] # keep track of generated steps in case want to plot something 
        for i in pbar:
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1)

            z = torch.randn(n_sample, 1).to(device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(x_i, c_i, t_is)
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if (i%20==0 or i==self.n_T or i<8) and (is_store):
                x_i_store.append(x_i.detach().cpu().numpy())
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store
    
    def sample_ddim(self, c_i, 
                    ddim_timesteps=50, 
                    device="cpu", 
                    ddim_discr_method="uniform",
                    ddim_eta=0, is_store=False):
        """args
            c_i(tensor, bs x d): the indepedent varaible
            ddim_timesteps (int): The num of steps for DDIM
            ddim_eta (float): the eta control DDIM randomness, eta=0, DDIM, eta=1 DDPM
        """
        assert (self.n_T % ddim_timesteps) == 0, "n_T should be divisible by ddim_timesteps."
        
        if ddim_discr_method == 'uniform':
            intv = self.n_T // ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(self.n_T, 0, -intv)))
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')
            
        ddim_timestep_seq = ddim_timestep_seq[::-1]
        # previous sequence
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])
        
        if False:
            pbar = tqdm(reversed(range(0, ddim_timesteps)), desc='DDIM sampling', total=ddim_timesteps)
        else:
            pbar = reversed(range(0, ddim_timesteps))
        
        c_i = c_i.to(device)
        n_sample = c_i.shape[0]
        x_i = torch.randn(n_sample, 1).to(device)  # x_T ~ N(0, 1), sample initial noise
        x_i_store = []
        for i in pbar:
            t_is = torch.tensor([ddim_timestep_seq[i]/ self.n_T], dtype=torch.get_default_dtype()).to(device)
            t_is = t_is.repeat(n_sample,1)
            z = torch.randn(n_sample, 1).to(device) if i > 1 else 0
            
            # get the pred noice
            eps = self.nn_model(x_i, c_i, t_is)
            
            alpt = self.alphabar_t[ddim_timestep_seq[i]]
            prev_alpt = self.alphabar_t[ddim_timestep_prev_seq[i]]
            
            # 3. get predicted x0
            pred_x0 = (x_i- torch.sqrt((1. - alpt)) * eps) / torch.sqrt(alpt)
            clip_denoised = False
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)
                
            # 4. compute variance: "sigma_t(η)" -> see formula (16)
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            sigmas_t = ddim_eta * torch.sqrt(
                (1 - prev_alpt) / (1 - alpt) * (1 - alpt/ prev_alpt))
            
            # 5. compute "direction pointing to x_t" of formula (12)
            pred_dir_xt = torch.sqrt(1 - prev_alpt- sigmas_t**2) * eps
            
            # 6. compute x_{t-1} of formula (12)
            z = torch.randn(n_sample, 1).to(device) if i >1 else 0
            x_prev = torch.sqrt(prev_alpt) * pred_x0 + pred_dir_xt + sigmas_t * z
            x_i = x_prev
            
            if (i%3==0 or ddim_timestep_seq[i]==self.n_T or i<8) and (is_store):
                x_i_store.append(x_i.detach().cpu().numpy())

        x_i_store = np.array(x_i_store)
        return x_i, x_i_store
