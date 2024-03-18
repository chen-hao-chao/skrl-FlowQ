import torch
import numpy as np
from . import flows

# Transforms to be applied to data as preprocessing


class Logit(flows.Flow):
    """Logit mapping of image tensor, see RealNVP paper

    ```
    logit(alpha + (1 - alpha) * x) where logit(x) = log(x / (1 - x))
    ```

    """

    def __init__(self, alpha=0.05, MaP=False):
        """Constructor

        Args:
          alpha: Alpha parameter, see above
        """
        super().__init__()
        self.alpha = alpha
        self.MaP = MaP

    def forward(self, z):
        beta = 1 - 2 * self.alpha
        z = (torch.sigmoid(z) - self.alpha) / beta

        if self.MaP:
            log_det = torch.zeros(z.shape[0], device=z.device)
        else:
            sum_dims = list(range(1, z.dim()))
            ls = torch.sum(torch.nn.functional.logsigmoid(z), dim=sum_dims)
            mls = torch.sum(torch.nn.functional.logsigmoid(-z), dim=sum_dims)
            log_det = -np.log(beta) * np.prod([*z.shape[1:]]) + ls + mls
        
        return z, log_det

    @torch.jit.export
    def inverse(self, z):
        beta = 1 - 2 * self.alpha
        z = self.alpha + beta * z
        logz = torch.log(z)
        log1mz = torch.log(1 - z)
        z = logz - log1mz
        
        if self.MaP:
            log_det = torch.zeros(z.shape[0], device=z.device)
        else:
            sum_dims = list(range(1, z.dim()))
            log_det = (
                np.log(beta) * np.prod([*z.shape[1:]])
                - torch.sum(logz, dim=sum_dims)
                - torch.sum(log1mz, dim=sum_dims)
            )
        return z, log_det


class Shift(flows.Flow):
    """Shift data by a fixed constant

    Default is -0.5 to shift data from
    interval [0, 1] to [-0.5, 0.5]
    """

    def __init__(self, shift=-0.5, MaP=False):
        """Constructor

        Args:
          shift: Shift to apply to the data
        """
        super().__init__()
        self.shift = shift
        self.MaP = MaP

    def forward(self, z):
        z = z - self.shift
        log_det = torch.zeros(z.shape[0], dtype=z.dtype,
                              device=z.device)
        return z, log_det

    @torch.jit.export
    def inverse(self, z):
        z = z + self.shift
        log_det = torch.zeros(z.shape[0], dtype=z.dtype,
                              device=z.device)
        return z, log_det

# (Lance implemented - 20230117)
class Scale(flows.Flow):
    def __init__(self, scale=0.5, MaP=False):
        super().__init__()
        # forward: in / scale
        # inverse: in * scale
        self.scale = scale
        self.MaP = MaP

    def forward(self, z):
        z = z / (self.scale)
        if self.MaP:
            log_det = torch.zeros(z.shape[0], device=z.device)
        else:
            # log |det| = -D * log (self.scale)
            log_det = -torch.ones(z.shape[0], device=z.device) * np.log(np.abs(self.scale)) * z.shape[1]
        return z, log_det

    @torch.jit.export
    def inverse(self, z):
        z = z * (self.scale)
        if self.MaP:
            log_det = torch.zeros(z.shape[0], device=z.device)
        else:
            # log |det| = D * log (self.scale)
            log_det = torch.ones(z.shape[0], device=z.device) * np.log(np.abs(self.scale)) * z.shape[1]
        return z, log_det

class arcTanh(flows.Flow):
    def __init__(self, MaP=False):
        super().__init__()
        self.MaP = MaP
        self.eps = np.finfo(np.float32).eps.item()

    def forward(self, z):
        z_ = torch.tanh(z)
        if self.MaP:
            log_det = torch.zeros(z.shape[0], device=z.device)
        else:
            log_det = torch.log(1-z_.pow(2) + self.eps).sum(-1, keepdim=False)
        return z_, log_det

    @torch.jit.export
    def inverse(self, z):
        z_ = torch.atanh(z)
        if self.MaP:
            log_det = torch.zeros(z.shape[0], device=z.device)
        else:
            log_det = -torch.log(1-z.pow(2) + self.eps).sum(-1, keepdim=False)
        return z_, log_det

# (Lance implemented - 20240121)
class Clip(flows.Flow):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, z_):
        # (Generation direaction) output must be [-1, 1] (this is defined according to the env.)
        # z_ = torch.clamp(z_, -1+self.eps, 1-self.eps)
        z_ = torch.clamp(z_, -1, 1) 
        return z_, torch.zeros(z_.shape[0], device=z_.device)
    
    @torch.jit.export
    def inverse(self, z):
        # (Density estimation direction) input must be [-1+esp, 1-esp] (prevent having nan in fwd passing logit or tanh...)
        z = torch.clamp(z, -1+self.eps, 1-self.eps) 
        return z, torch.zeros(z.shape[0], device=z.device)
    
# (Roy implemented - 20240121)
class ClipArcTanh(flows.Flow):
    def __init__(self, eps=1e-5, MaP=False):
        super().__init__()
        self.MaP = MaP
        self.eps = eps # np.finfo(np.float32).eps.item()

    def forward(self, z):
        z_ = torch.tanh(z)
        z_ = torch.clamp(z_, -1+self.eps, 1-self.eps)
        if self.MaP:
            log_det = torch.zeros(z.shape[0], device=z.device)
        else:
            log_det = torch.log(1-z_.pow(2)).sum(-1, keepdim=False)
        return z_, log_det

    @torch.jit.export
    def inverse(self, z):
        z = torch.clamp(z, -1+self.eps, 1-self.eps)
        z_ = torch.atanh(z)
        if self.MaP:
            log_det = torch.zeros(z.shape[0], device=z.device)
        else:
            log_det = -torch.log(1-z.pow(2)).sum(-1, keepdim=False)
        return z_, log_det
    

# # original
# class Preprocessing(flows.Flow):
#     def __init__(self, option='eye', clip=True):
#         super().__init__()
#         option = option + "_"
#         self.func = option.split('_')[0]
#         self.MaP = option.split('_')[1] == "map"

#         if self.func == 'eye': # identity
#             trans = [Shift(shift=0.)]
#         elif self.func == 'atanh': # [-1, 1] -> (-infty, infty)
#             trans = [arcTanh(MaP=self.MaP)] # Inv. Pass in this direction ( (-infty, infty) <= [-1, 1])
#         elif self.func == 'scaleatanh': # [-1, 1] -> [-s, s] -> (-infty, infty)
#             trans = [arcTanh(MaP=self.MaP), Scale(scale=0.9, MaP=self.MaP)] # Inv. Pass in this direction ( (-infty, infty) <= [-1, 1])
#         elif self.func == 'logit': # [-1, 1] -> [-0.5, 0.5] -> [0, 1] -> (-infty, infty)
#             trans = [Logit(MaP=self.MaP), Shift(shift=0.5, MaP=self.MaP), Scale(scale=0.5, MaP=self.MaP)] # Inv. Pass in this direction ( (-infty, infty) <= [-1, 1])
#         elif self.func == 'clipatanh':
#             trans = [ClipArcTanh(eps=1e-5, MaP=self.MaP)]
#         else:
#             raise NotImplementedError("Sorry, not implemented!")

#         if clip:
#             trans += [Clip(eps=1e-5)]
#         self.trans = torch.nn.ModuleList(trans)

#     def forward(self, z, context=None):
#         log_det = torch.zeros(z.shape[0], device=z.device)
#         for flow in self.trans:
#             z, log_d = flow.forward(z)
#             log_det += log_d
#         return z, log_det

#     def inverse(self, z, context=None):
#         log_det = torch.zeros(z.shape[0], device=z.device)
#         for i in range(len(self.trans) - 1, -1, -1):
#             z, log_d = self.trans[i].inverse(z)
#             log_det += log_d
#         return z, log_det

#     def get_qv(self, z, context=None):
#         z_, q = self.inverse(z, context)
#         v = torch.zeros(z.shape[0], device=z.device)
#         return z_, q, v


class Preprocessing(flows.Flow):
    def __init__(self, option='eye', clip=True):
        super().__init__()
        option = option + "_"
        self.func = option.split('_')[0]
        self.MaP = option.split('_')[1] == "map"

        if self.func == 'eye': # identity
            trans = [Shift(shift=0.)]
        elif self.func == 'atanh': # [-1, 1] -> (-infty, infty)
            trans = [arcTanh(MaP=self.MaP)] # Inv. Pass in this direction ( (-infty, infty) <= [-1, 1])
        elif self.func == 'scaleatanh': # [-1, 1] -> [-s, s] -> (-infty, infty)
            trans = [arcTanh(MaP=self.MaP), Scale(scale=0.9, MaP=self.MaP)] # Inv. Pass in this direction ( (-infty, infty) <= [-1, 1])
        elif self.func == 'logit': # [-1, 1] -> [-0.5, 0.5] -> [0, 1] -> (-infty, infty)
            trans = [Logit(MaP=self.MaP), Shift(shift=0.5, MaP=self.MaP), Scale(scale=0.5, MaP=self.MaP)] # Inv. Pass in this direction ( (-infty, infty) <= [-1, 1])
        elif self.func == 'clipatanh':
            trans = [ClipArcTanh(eps=1e-5, MaP=self.MaP)]
        else:
            raise NotImplementedError("Sorry, not implemented!")

        if clip:
            trans += [Clip(eps=1e-5)]
        self.trans = torch.nn.ModuleList(trans)

    def forward(self, z, context=None):
        log_det = torch.zeros(z.shape[0], device=z.device)
        for flow in self.trans:
            z, log_d = flow.forward(z)
            log_det += log_d
        return z, log_det

    @torch.jit.export
    def inverse(self, z, context=None):
        log_det = torch.zeros(z.shape[0], device=z.device)
        for flow in self.trans[::-1]:
            z, log_d = flow.inverse(z)
            log_det += log_d
        return z, log_det
    
    @torch.jit.export
    def get_qv(self, z, context):
        z_, q = self.inverse(z, context)
        v = torch.zeros(z.shape[0], device=z.device)
        return z_, q, v