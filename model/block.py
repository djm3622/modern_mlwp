import torch.nn as nn
import model.afno as afno # type: ignore
import model.galerkin as galerkin # type: ignore


# patch embedding layer to encode regions of space
class PatchEmbedding(nn.Module):
    def __init__(self, c, dim, patch_size):
        super().__init__()

        self.embedding = nn.Conv2d(in_channels=c, out_channels=dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        return self.embedding(x)


class PatchReconstruction(nn.Module):
    def __init__(self, hidden_dim, out_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.reconstruction = nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=out_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        return self.reconstruction(x)  


# resolution invariant block with AFNO, Galerkin Attention, pw convs
class InvariantBlock(nn.Module):
    def __init__(self, 
        dim: int, 
        num_blocks: int = 16, 
        expansion: int = 8, 
        heads: int = 8
    ) -> None:
        super().__init__()

        self.afno = afno.AFNO2D(
            dim, num_blocks=num_blocks, 
            hidden_size_factor=expansion
        )
        self.attn = galerkin.GalerkinAttention(
            n_head = heads, 
            d_model = dim
        )

        self.pw_up = nn.Conv2d(dim, dim*expansion, kernel_size=1)
        self.pw_down = nn.Conv2d(dim*expansion, dim, kernel_size=1)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.gelu = nn.GELU()
        
    def forward(self, x):
        # AFNO path
        x = x + self.afno(self.norm1(x))
        
        # Galerkin Attention path
        x = self.norm2(x)
        B, H, W, C = x.shape
        h = x.reshape(B, H * W, C)
        x = x + self.attn(h, h, h).reshape(B, H, W, C)
        
        # Channel mixing
        x = self.norm3(x)
        x = x.permute(0, 3, 1, 2)
        h = self.pw_up(x)
        h = self.gelu(h)
        h = self.pw_down(h)
        x = x + h

        return x.permute(0, 2, 3, 1)