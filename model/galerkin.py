import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from typing import Tuple

# Galerkin Attention block adapted from https://github.com/scaomath/galerkin-transformer
class GalerkinAttention(nn.Module):
    def __init__(
        self, 
        n_head: int, 
        d_model: int,
        dropout: float = 0.0,
        xavier_init: float = 1e-4,
        diagonal_weight: float = 1e-2,
        symmetric_init: bool = False,
        norm: bool = False,
        norm_type: str = 'layer',
        eps: float = 1e-5,
        debug: bool = False
    ) -> None:
    
        super().__init__()
        assert d_model % n_head == 0
        self.d_k = d_model // n_head
        self.n_head = n_head

        self.linears = nn.ModuleList(
            [copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(3)]
        )
        
        self.xavier_init = xavier_init
        self.diagonal_weight = diagonal_weight
        self.symmetric_init = symmetric_init

        if self.xavier_init > 0:
            self._reset_parameters()

        self.add_norm = norm
        self.norm_type = norm_type
        
        if norm:
            self._get_norm(eps=eps)

        self.dropout = nn.Dropout(dropout)
        self.debug = debug

    def forward(self, query, key, value):
        bsz = query.size(0)

        query, key, value = [
            layer(x).view(bsz, -1, self.n_head, self.d_k).transpose(1, 2) 
            for layer, x in zip(self.linears, (query, key, value))
        ]

        if self.add_norm:
            if self.norm_type == 'instance':
                key, value = key.transpose(-2, -1), value.transpose(-2, -1)

            key = torch.stack(
                [norm(x) for norm, x in zip(self.norm_K, (key[:, i, ...] for i in range(self.n_head)))], dim=1
            )
            value = torch.stack(
                [norm(x) for norm, x in zip(self.norm_V, (value[:, i, ...] for i in range(self.n_head)))], dim=1
            )

            if self.norm_type == 'instance':
                key, value = key.transpose(-2, -1), value.transpose(-2, -1)

        x = linear_attention(
            query, key, value, dropout=self.dropout
        )

        out_dim = self.n_head * self.d_k
        att_output = x.transpose(1, 2).contiguous().view(bsz, -1, out_dim)

        return att_output

    def _reset_parameters(self):
        for param in self.linears.parameters():
            if param.ndim > 1:
                xavier_uniform_(param, gain=self.xavier_init)
                if self.diagonal_weight > 0.0:
                    param.data += self.diagonal_weight * \
                        torch.diag(torch.ones(
                            param.size(-1), dtype=torch.float))
                if self.symmetric_init:
                    param.data += param.data.T
            else:
                constant_(param, 0)

    def _get_norm(self, eps):
        if self.norm_type == 'instance':
            self.norm_K = self._get_instancenorm(self.d_k, self.n_head, eps=eps, affine=True)
            self.norm_V = self._get_instancenorm(self.d_k, self.n_head, eps=eps, affine=True)
        elif self.norm_type == 'layer':
            self.norm_K = self._get_layernorm(self.d_k, self.n_head, eps=eps)
            self.norm_V = self._get_layernorm(self.d_k, self.n_head, eps=eps)

    @staticmethod
    def _get_layernorm(normalized_dim, n_head, **kwargs):
        return nn.ModuleList(
            [copy.deepcopy(nn.LayerNorm(normalized_dim, **kwargs)) for _ in range(n_head)])

    @staticmethod
    def _get_instancenorm(normalized_dim, n_head, **kwargs):
        return nn.ModuleList(
            [copy.deepcopy(nn.InstanceNorm1d(normalized_dim, **kwargs)) for _ in range(n_head)])


def linear_attention(
    query, key, value, dropout=None
) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    Adapted from lucidrains' implementaion
    https://github.com/lucidrains/linear-attention-transformer/blob/master/linear_attention_transformer/linear_attention_transformer.py
    to https://nlp.seas.harvard.edu/2018/04/03/attention.html template
    linear_attn function
    Compute the Scaled Dot Product Attention globally
    '''

    seq_len = query.size(-2)
    scores = torch.matmul(key.transpose(-2, -1), value)

    p_attn = scores / seq_len

    if dropout is not None:
        p_attn = F.dropout(p_attn)

    out = torch.matmul(query, p_attn)
    return out