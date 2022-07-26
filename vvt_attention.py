from sre_constants import AT_NON_BOUNDARY
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

import shutil
import os
# from soft import SoftmaxFreeAttention
# from quadtree import QuadtreeAttention
from performer import PerformerAttention, LinformerSelfAttention

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# swin does not work
class SwinAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        frame = inspect.currentframe()         
        self.gpu_tracker = MemTracker(frame, path='swin')

    def forward(self, x, H, W, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        self.gpu_tracker.track()
        print(x.shape)
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        self.gpu_tracker.track()
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class Vanilla_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        frame = inspect.currentframe()         
        self.gpu_tracker = MemTracker(frame, path='vit')

    def forward(self, x, H, W):
        self.gpu_tracker.track()
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        self.gpu_tracker.track()
        return x

class pvt_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        print('linear:', linear, 'sr_ratio:', sr_ratio)

        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)

        frame = inspect.currentframe()         
        self.gpu_tracker = MemTracker(frame, path='pvtv2')
    
    def get_attn(self):
        return self.attn_map

    def save_attn(self, attn):
        self.attn_map = attn

    def save_attn_gradients(self, attn_gradients):
        self.attn_map_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_map_gradients

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

    def forward(self, x, H, W):
        self.gpu_tracker.track()
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B, h, N, C/h

        if not self.linear: 
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # save the attention map for visualization
        if x.requires_grad:
            self.save_attn(attn)
            attn.register_hook(self.save_attn_gradients)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        self.gpu_tracker.track()
        return x

class VicinityVisionAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None,
                dropout_rate=0.0, causal=False, use_sum=True, 
                sr_ratio=1, fr_ratio=1, linear=False, se_reduction=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
		# q, k, v projection
        self.fr = fr_ratio 
        # feature reduction
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.fr)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.fr)
        self.q_proj = nn.Linear(embed_dim, embed_dim // self.fr)

        self.sr_ratio = sr_ratio
        self.linear = linear
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(embed_dim, embed_dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(embed_dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(embed_dim)
            self.act = nn.GELU()

        self.apply(self._init_weights)
        # outprojection
        self.out_proj = nn.Linear(embed_dim//self.fr, embed_dim)
		# dropout rate
        self.dropout_rate = dropout_rate
		# causal
        self.causal = causal

        assert (self.embed_dim % self.num_heads == 0), "embed_dim must be divisible by num_heads"
        self.use_sum = use_sum
        if self.use_sum:
            print('use sum')
            print('linear:', linear, 'sr_ratio:', sr_ratio, 'fr_ratio:', fr_ratio, 'se_ratio:', se_reduction)
        else:
            print('use production')
            print('linear:', linear, 'sr_ratio:', sr_ratio, 'fr_ratio:', fr_ratio, 'se_ratio:', se_reduction)

        # se block:
        reduction = se_reduction
        self.se_pool = nn.AdaptiveAvgPool1d(1)
        self.se_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // reduction, embed_dim, bias=False),
            nn.Sigmoid()
        )

        self.clip = True

        frame = inspect.currentframe()         
        self.gpu_tracker = MemTracker(frame, path='vvt')

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
    
    def abs_clamp(self, t):
        min_mag = 1e-4
        max_mag = 10000
        sign = t.sign()
        return t.abs_().clamp_(min_mag, max_mag)*sign
        
    def get_index(self, m, n):
        """
        m = width, n = height
        """
        c = np.pi / 2
        seq_len = m * n
        index = torch.arange(seq_len).reshape(1, -1, 1, 1)
        a = c * (index // m) / n
        b = c * (index % m) / m

        seq_len = (m/self.sr_ratio) * (n/self.sr_ratio)
        index = torch.arange(seq_len).reshape(1, -1, 1, 1)
        a_sr = c * (index // (m/self.sr_ratio) ) / (n/self.sr_ratio)
        b_sr = c * (index % (m/self.sr_ratio)) / (m/self.sr_ratio)

        return nn.Parameter(a, requires_grad=False), nn.Parameter(b, requires_grad=False), \
               nn.Parameter(a_sr, requires_grad=False), nn.Parameter(b_sr, requires_grad=False)



    def forward(self, query, H, W):
        self.gpu_tracker.track()
        # H: height, W: weight
        num_heads = self.num_heads
        B, N, C = query.shape
        query_se = query.permute(0, 2, 1)
        query_se = self.se_pool(query_se).view(B, C)
        query_se = self.se_fc(query_se).view(B, C, 1)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = query.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                k = self.k_proj(x_)
                v = self.v_proj(x_)
            else:
                k = self.k_proj(query)
                v = self.v_proj(query)
        else:
            x_ = query.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            k = self.k_proj(x_)
            v = self.v_proj(x_)

        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)

        query = query.permute(1,0,2)
        tgt_len, bsz, embed_dim = query.size()
        head_dim = embed_dim // num_heads
        # (L, N, E)
        q = self.q_proj(query)

		# multihead
		# (N, L, h, d)
        q = q.contiguous().view(tgt_len, bsz, num_heads, head_dim//self.fr).transpose(0, 1)
		# (N, S, h, d)
        k = k.contiguous().view(-1, bsz, num_heads, head_dim//self.fr).transpose(0, 1)
		# (N, S, h, d)
        v = v.contiguous().view(-1, bsz, num_heads, head_dim//self.fr).transpose(0, 1)
		# relu
        q = F.relu(q)
        k = F.relu(k)

        a, b, a_sr, b_sr = self.get_index(W, H)
        a = a.to(q)
        b = b.to(q)
        a_sr = a_sr.to(q)
        b_sr = b_sr.to(q)

        # print(q.shape, a.shape)

        if self.use_sum:
            # sum
            q_ = torch.cat([q * torch.cos(a), \
                            q * torch.sin(a), \
                            q * torch.cos(b), \
                            q * torch.sin(b)], \
                            dim=-1)
            # (N, S, h, 4 * d)
            k_ = torch.cat([k * torch.cos(a_sr), \
                            k * torch.sin(a_sr), \
                            k * torch.cos(b_sr), \
                            k * torch.sin(b_sr)], \
                            dim=-1)
        else:
            q_ = torch.cat([q * torch.cos(a) * torch.cos(b), \
                            q * torch.cos(a) * torch.sin(b), \
                            q * torch.sin(a) * torch.cos(b), \
                            q * torch.sin(a) * torch.sin(b)], \
                            dim=-1)
            # (N, S, h, 4 * d)
            k_ = torch.cat([k * torch.cos(a_sr) * torch.cos(b_sr), \
                            k * torch.cos(a_sr) * torch.sin(b_sr), \
                            k * torch.sin(a_sr) * torch.cos(b_sr), \
                            k * torch.sin(a_sr) * torch.sin(b_sr)], \
                            dim=-1)

        eps = 1e-4

        #---------------------------------------------------------------------------------
        kv_ = torch.matmul(k_.permute(0, 2, 3, 1), v.permute(0, 2, 1, 3))  # no einsum  
        if self.clip:
            kv_ = self.abs_clamp(kv_)
        #---------------------------------------------------------------------------------

        #--------------------------------------------------------------------------------
        k_sum = torch.sum(k_, axis=1, keepdim=True) # no einsum                         
        z_ = 1 / (torch.sum(torch.mul(q_, k_sum), axis=-1) + eps) # no einsum           
        if self.clip:
            z_ = self.abs_clamp(z_)
        #--------------------------------------------------------------------------------

        # no einsum---------------------------------------------------------------------
        attn_output = torch.matmul(q_.transpose(1, 2), kv_).transpose(1, 2)
        if self.clip:
            attn_output = self.abs_clamp(attn_output)
        # nlhm,nlh -> nlhm
        attn_output = torch.mul(attn_output, z_.unsqueeze(-1))
        if self.clip:
            attn_output = self.abs_clamp(attn_output)
        #--------------------------------------------------------------------------------
        
        # (N, L, h, d) -> (L, N, h, d) -> (L, N, E)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim//self.fr)

        attn_output = self.out_proj(attn_output)
        if self.clip:
            attn_output = self.abs_clamp(attn_output)

        # ------------------------------------- se block
        attn_output = attn_output.permute(1,2,0)
        attn_output = attn_output + attn_output * query_se.expand_as(attn_output)
        if self.clip:
            attn_output = self.abs_clamp(attn_output)
        attn_output = attn_output.permute(2,0,1)
        # -------------------------------------------------

        attn_output = attn_output.permute(1,0,2)
        self.gpu_tracker.track()
        return attn_output



# try:
#     os.remove('/home/users/u5876230/vvt/gpu_mem_track.txt')
# except:
#     pass


embed_dim = 96
num_heads = 1
fr_ratio = 2
se_reduction = 1

# attention = VicinityVisionAttention(embed_dim=embed_dim, num_heads=num_heads, fr_ratio=fr_ratio, sr_ratio=4, se_reduction=1)
# attention = pvt_Attention(dim=embed_dim, num_heads=num_heads, sr_ratio=8)
# attention = Vanilla_Attention(dim=embed_dim, num_heads=num_heads)
# attention =  SwinAttention(dim=embed_dim, window_size=[7,7], num_heads=num_heads)
# attention = SoftmaxFreeAttention(dim=embed_dim, num_heads=num_heads, ratio=8, conv_size=3)
# attention = QuadtreeAttention(dim=embed_dim, num_heads=num_heads, sr_ratio=4)
# attention = PerformerAttention(embed_dim=embed_dim, num_heads=num_heads)


H = W = 20
input = torch.rand(1,H*W,embed_dim).cuda()

for iter in range(100): 
    attention = LinformerSelfAttention(dim=embed_dim, seq_len=H*H, heads = num_heads)
    attention.cuda()
    a = attention(input, H, W)

try:
    os.remove('/home/users/u5876230/vvt/linformer.txt')
except:
    pass


for H in range(20, 2000, 40):
    attention = LinformerSelfAttention(dim=embed_dim, seq_len=H*H, heads = num_heads)
    attention.cuda()
    for iter in range(1):
        input = torch.rand(1, H*H, embed_dim).cuda()
        print(input.shape, H)
        a = attention(input, H, H)




# for i in range(5, 10, 1):

#     H = 2**i
#     input = torch.rand(1,H*H,embed_dim).cuda()
#     print(input.shape, H)
#     a = attention(input, H, H)




# for H in range(49, 20000, 49):
    
#     # input = torch.rand(1,H*H,embed_dim).cuda()

#     nw = H//49
#     input = torch.rand(nw,49,embed_dim).cuda()

#     print(input.shape, H)
#     a = attention(input, H, H)
