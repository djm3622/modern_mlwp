import torch.nn as nn
import model.block as blocks


# scale equivarient ffn
class FouCGalNet(nn.Module):
    def __init__(self, 
        in_dim: int, 
        out_dim: int, 
        hidden: int = 128, 
        depth: int = 8, 
        heads: int = 8,
        patch_size: int = 4,
        num_blocks: int = 16,
        expansion: int = 8
    ) -> None:
        super().__init__()
        
        self.embed = blocks.PatchEmbedding(in_dim, hidden, patch_size=patch_size)
        self.blocks = nn.ModuleList([
            blocks.InvariantBlock(
                hidden, num_blocks=num_blocks, 
                expansion=expansion, heads=heads
            ) for _ in range(depth)
        ])
        self.recon = blocks.PatchReconstruction(hidden, out_dim, patch_size=patch_size)
        
    def forward(self, x):
        x = self.embed(x)
        x = x.permute(0, 2, 3, 1)

        for block in self.blocks:
            x = block(x)

        x = x.permute(0, 3, 1, 2)
        x = self.recon(x)

        return x
