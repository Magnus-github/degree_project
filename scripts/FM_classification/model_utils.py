import torch.nn as nn
import torch
import numpy as np


def AvgPool(kernel_size: int):
    return nn.AvgPool1d(kernel_size)


def MaxPool(kernel_size: int):
    return nn.MaxPool1d(kernel_size)


class Voting:
    def __init__(self, dummy):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def __call__(self, x):
        x = x.to('cpu')
        x = torch.mode(x, dim=-1).values
        return x.to(self.device)
    

class TrigonometricPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, seq_len, cls_token=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.cls_token = cls_token

    # sine-cosine positional encoding
    def get_1d_sincos_pos_embed(self):
        """
        seq_len: int of the sequence length
        return:
        pos_embed: [seq_len, embed_dim] or [1+seq_len, embed_dim] (w/ or w/o cls_token)
        """
        pos = np.arange(self.seq_len, dtype=np.float32)
        pos_embed = self._get_1d_sincos_pos_embed_from_grid(pos)
        if self.cls_token:
            pos_embed = np.concatenate([np.zeros([1, self.embed_dim]), pos_embed], axis=0)
        return pos_embed


    def _get_1d_sincos_pos_embed_from_grid(self, pos):
        """
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        out: (M, D)
        """
        assert self.embed_dim % 2 == 0
        omega = np.arange(self.embed_dim // 2, dtype=np.float64)
        omega /= self.embed_dim / 2.
        omega = 1. / 10000**omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

        emb_sin = np.sin(out) # (M, D/2)
        emb_cos = np.cos(out) # (M, D/2)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)

        return torch.tensor(emb.transpose(1, 0), dtype=torch.float32)  # (D, M)
    
    def forward(self, x):
        return x + self.get_1d_sincos_pos_embed()


class RandomMask(nn.Module):
    def __init__(self, num_keep: int):
        super().__init__()
        self.num_keep = num_keep

    def forward(self, x):
        B, N, D = x.shape

        noise = torch.rand(B, N)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :self.num_keep]
        x_masked = x.gather(1, ids_keep[:, :, None].expand(-1, -1, D))

        mask = torch.ones(B, N)
        mask[:, :self.num_keep] = 0

        mask = mask[:,ids_restore][0]

        return x_masked, mask, ids_restore
