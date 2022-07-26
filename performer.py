import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
import numpy as np
from torch import Tensor
from typing import Optional

from gpu_mem_track import MemTracker
import inspect

# from fast_transformers.attention.causal_linear_attention import causal_linear
from torch import Tensor
from typing import Optional
from einops import rearrange
import torch.linalg

##################### performer
def orthogonal_matrix_chunk(cols, device = None, dtype=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    # q, r = torch.linalg.qr(unstructured_block.cpu(), mode='reduced')
    q, r = torch.qr(unstructured_block.cpu(), some=True)
    q, r = map(lambda t: t.to(device), (q, r))
    return q.t().to(dtype)

def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, seed=0, device=None, dtype=None):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []
    cur_seed = seed

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device=device, dtype=dtype)
        block_list.append(q)
        cur_seed = cur_seed + 1

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device=device, dtype=dtype)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    multiplier = torch.randn((nb_rows, nb_columns), device=device, dtype=dtype).norm(dim=1)

    return torch.diag(multiplier) @ final_matrix

def create_proj_matrix(num_heads, proj_dim, input_dim, ortho=False, seed=0, device=None, dtype=None):
    if ortho:
        return torch.stack(
            [
                gaussian_orthogonal_random_matrix(proj_dim, input_dim, seed=seed + h * 1000, device=device, dtype=dtype)
                for h in range(num_heads)
            ], dim=0)
    else:
        return torch.randn(num_heads, proj_dim, input_dim, device=device, dtype=dtype)

def favorp_projection(
        data: torch.Tensor,
        projection_matrix: torch.Tensor,
        is_query: bool,
        eps: float=0.0001):
    """
    Constructs nonnegative kernel features for fast softmax attention.
    Args:
      data: input for which features are computes
      projection_matrix: random matrix used to compute features
      batch_dims_t: tuple of batch dimensions
      is_query: predicate indicating whether input data corresponds to queries or
        keys
      eps: numerical stabilizer.
    Returns:
      Random features for fast softmax attention.
    """
    # We have e^{qk^T/sqrt{d}} = e^{q_norm k_norm^T}, where
    # w_norm = w * data_normalizer for w in {q,k}.
    data_normalizer = (data.shape[-1] ** -0.25)
    ratio = (projection_matrix.shape[1] ** -0.5)
    data_dash = torch.einsum('bh...d,hjd->bh...j', 
                            (data_normalizer * data),
                            projection_matrix)
    diag_data = torch.sum(data ** 2, dim=-1)
    diag_data = (diag_data / 2.0) * data_normalizer * data_normalizer
    diag_data = diag_data.unsqueeze(-1)
    
    if is_query:
        data_dash_log = data_dash - diag_data
        stabilizer = torch.amax(data_dash, dim=-1, keepdim=True).detach()
        data_dash = ratio * torch.exp(data_dash_log - stabilizer) + eps
    else:
        data_dash_log = data_dash - diag_data
        stabilizer = torch.amax(data_dash, dim=(-1, -2), keepdim=True).detach()
        data_dash = ratio * torch.exp(data_dash_log - stabilizer) + eps
    return data_dash

class PerformerAttention(nn.Module):
    def __init__(self,embed_dim,num_heads,kdim=None,vdim=None,dropout_rate=0.0,causal=False,use_sum=True, sr_ratio=1,fr_ratio=1, linear=False, se_reduction=2, prod_type="right"):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
		# q, k, v projection
        self.head_dim = self.embed_dim // self.num_heads
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.approx_attn_dim = 64
        self.use_random_proj = True
        self.register_buffer('eval_proj', create_proj_matrix(
            self.num_heads, 
            self.approx_attn_dim, 
            self.head_dim, 
            ortho=True
            )
        )
        frame = inspect.currentframe()         
        self.gpu_tracker = MemTracker(frame, path='performer')

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, query: Tensor,H, W):
        self.gpu_tracker.track()
        num_heads = self.num_heads
        # H = int(self.m)
        # W = int(self.n)

        B, N, C = query.shape
        # query = query.permute(1,0,2)
        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)
        # (B, H, L, D)
        # q = q.contiguous().view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        # k = k.contiguous().view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        # v = v.contiguous().view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        # print("========================")
        # print(q.shape)

        if self.training:
            projection_matrix = create_proj_matrix(
                self.num_heads, self.approx_attn_dim, self.head_dim, ortho=False, device=q.device, dtype=q.dtype)
        else:
            projection_matrix = self.eval_proj
        q_prime, k_prime = self.q_k_projection(q, k, projection_matrix)
        # print(q_prime.shape)

        eps = 1e-2
        kv = torch.einsum('...nm,...nd->...md', k_prime, v)
        qkv = torch.einsum('...nm,...md->...nd', q_prime, kv)
        normalizer = torch.einsum('...nm,...m->...n', q_prime, k_prime.sum(dim=-2))
        output = qkv / normalizer.unsqueeze(-1).clamp(min=eps)
        # print(output.shape)

        # attn_output = output.contiguous().view(B, N, self.num_heads, self.head_dim)
        attn_output = rearrange(output, 'b h n d -> b n (h d)', h=self.num_heads)
        attn_output = self.out_proj(attn_output)
        # print(output.shape)
        # print("========================")
        self.gpu_tracker.track()
        return attn_output

    def q_k_projection(self, q, k, random_proj=None):
        assert random_proj is not None
        feature_proj = partial(favorp_projection, projection_matrix = random_proj)

        q = feature_proj(q, is_query = True)
        k = feature_proj(k, is_query = False)
        return q, k

    def comput_mask(self, m, n):
        # m = W, n = H
        c = np.pi / 2
        seq_len = m * n
        # 1, n, 1
        index = torch.arange(seq_len).reshape(1, -1, 1)
        a = c * (index // m) / n
        b = c * (index % m) / m
        # 1, n, n
        mask = torch.cos(a - a.transpose(1, 2)) + torch.cos(b - b.transpose(1, 2))

        return nn.Parameter(mask, requires_grad=False)



def default(val, default_val):
    return val if val is not None else default_val

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

class LinformerSelfAttention(nn.Module):
    def __init__(self, dim, seq_len, k = 256, heads = 8, dim_head = None, one_kv_head = False, share_kv = False, dropout = 0.):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.seq_len = seq_len
        self.k = k

        self.heads = heads

        dim_head = default(dim_head, dim // heads)
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * heads, bias = False)

        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        self.to_k = nn.Linear(dim, kv_dim, bias = False)
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias = False)
            self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * heads, dim)
        frame = inspect.currentframe()         
        self.gpu_tracker = MemTracker(frame, path='linformer')

    def forward(self, x, H, W):
        self.gpu_tracker.track()
        b, n, d, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k

        context = None
        kv_len = n if context is None else context.shape[1]
        assert kv_len == self.seq_len, f'the sequence length of the key / values must be {self.seq_len} - {kv_len} given'

        queries = self.to_q(x)

        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)

        kv_input = x if context is None else context

        keys = self.to_k(kv_input)
        values = self.to_v(kv_input) if not self.share_kv else keys

        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

        # project keys and values along the sequence length dimension to k

        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values

        queries = queries.reshape(b, n, h, -1).transpose(1, 2)

        merge_key_values = lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))

        # attention
        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        # split heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        self.gpu_tracker.track()
        return self.to_out(out)

