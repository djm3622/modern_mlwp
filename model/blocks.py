import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import copy
from galerkin import GalerkinAttention
from afno import AdaptiveFNO


# patch embedding layer to encode regions of space
class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        pass


# resolution invariant block with AFNO, Galerkin Attention, pw convs
class InvariantBlock(nn.Module):
    def __init__(self, dim, modes=16, expansion=8, heads=8):
        super().__init__()
        self.afno = AdaptiveFNO(dim, modes)
        self.attn = GalerkinAttention(
            n_head = heads, 
            d_model = dim
        )

        self.pw_up = nn.Conv2d(dim, dim*expansion, 1)
        self.pw_down = nn.Conv2d(dim*expansion, dim, 1)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.gelu = nn.GELU()
        
    def forward(self, x):
        # AFNO path
        x = x + self.afno(self.norm1(x))
        
        # Galerkin Attention path
        x = x + self.attn(self.norm2(x))
        
        # Channel mixing
        x = x.permute(0, 3, 1, 2)
        h = self.pw_up(x)
        h = self.gelu(h)
        h = self.pw_down(h)
        x = x + h

        return x.permute(0, 2, 3, 1)