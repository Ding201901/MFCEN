import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from cnn_backbone import Backbone, BandWiseFusion


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        # x:[b,n,dim]
        b, n, _, h = *x.shape, self.heads

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        # mask value: -inf
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim = -1)
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel, mode):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout = dropout)))
            ]))

        self.mode = mode
        self.skipcat = nn.ModuleList([])
        for _ in range(depth - 2):
            self.skipcat.append(nn.Conv2d(num_channel + 1, num_channel + 1, [1, 2], 1, 0))

    def forward(self, x, mask = None):
        if self.mode == 'ViT':
            for attn, ff in self.layers:
                x = attn(x, mask = mask)
                x = ff(x)
        elif self.mode == 'CAF':
            last_output = []
            nl = 0
            for attn, ff in self.layers:
                last_output.append(x)
                if nl > 1:
                    x = self.skipcat[nl - 2](
                        torch.cat([x.unsqueeze(3), last_output[nl - 2].unsqueeze(3)], dim = 3)).squeeze(3)
                x = attn(x, mask = mask)
                x = ff(x)
                nl += 1

        return x


# 同一scale的特征进行融合, 特征均为向量
class ScaleFeatFuse(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x1, x2):
        # x:[b,n,dim]
        b, n, _, h = *x1.shape, self.heads

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x1).chunk(3, dim = -1)
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q1, k1, v1 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x2).chunk(3, dim = -1)
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q2, k2, v2 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots12 = torch.einsum('bhid,bhjd->bhij', q1, k2) * self.scale
        dots21 = torch.einsum('bhid,bhjd->bhij', q2, k1) * self.scale

        # softmax normalization -> attention matrix
        attn12 = dots12.softmax(dim = -1)
        attn21 = dots21.softmax(dim = -1)
        # value * attention matrix -> output
        out12 = torch.einsum('bhij,bhjd->bhid', attn12, v1)
        out21 = torch.einsum('bhij,bhjd->bhid', attn21, v2)
        # cat all output -> [b, n, head_num*head_dim]
        out12 = rearrange(out12, 'b h n d -> b n (h d)')
        out12 = self.to_out(out12)
        out21 = rearrange(out21, 'b h n d -> b n (h d)')
        out21 = self.to_out(out21)
        return out21 - out12


class ViT(nn.Module):
    def __init__(self, patch_size, num_feats, band_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls',
                 channels = 1, dim_head = 16, dropout = 0., emb_dropout = 0., mode = 'ViT'):
        super().__init__()

        self.pos_embedding = nn.Parameter(torch.randn(1, num_feats + 1, dim))
        self.cnn_backbone = Backbone(patch_size, band_size = band_size, dim = dim)  # Siamese network
        self.cnn_backbone1 = Backbone(patch_size, band_size = band_size, dim = dim)  # Siamese network
        self.scale_fuse = ScaleFeatFuse(dim, heads, dim_head, dropout)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, num_feats, mode)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x, mask = None):
        # x: [batch, 2 * band_size, patch_size, patch_size]
        x1, x2 = torch.split(x, int(x.shape[1] / 2), 1)

        # cnn to extract multi-scale features
        feat1, feat2, feat3, feat4 = self.cnn_backbone(x1, x2)

        feats_seq = torch.cat((feat1, feat2, feat3, feat4), 1)  # feat1, feat2, feat3, feat4

        b, n, _ = feats_seq.shape

        # add position embedding
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)  # [b,1,dim]
        x = torch.cat((cls_tokens, feats_seq), dim = 1)
        # x = x + self.pos_embedding
        x = self.dropout(x)

        # transformer: x[b,n + 1,dim] -> x[b,n + 1,dim]
        x = self.transformer(x, mask)

        # classification: using cls_token output
        x = self.to_latent(x[:, 0])

        # MLP classification layer
        return self.mlp_head(x)


# Example
# if __name__ == '__main__':
#     x = torch.rand([32, 340, 27])
#     model = ViT(
#         image_size = 3,
#         near_band = 3,
#         num_patches = 2 * 170,
#         num_classes = 2,
#         dim = 64,
#         depth = 5,
#         heads = 4,
#         mlp_dim = 8,
#         dropout = 0.1,
#         emb_dropout = 0.1,
#         mode = 'CAF'
#     )
#
#     out = model(x)
#     x1 = torch.rand([32, 8, 64])
#     x2 = torch.rand([32, 8, 64])
#     model = ScaleFeatFuse(64, 4, 16, 0.1)
#     out = model(x1, x2)
