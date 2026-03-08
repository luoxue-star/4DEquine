import torch
import torch.nn as nn
import numpy as np
from einops import rearrange


class TemporalAttention(nn.Module):
    def __init__(self, in_dim=1280, out_dim=1280, hdim=512, nlayer=6, nhead=4, residual=False):
        super(TemporalAttention, self).__init__()
        self.hdim = hdim
        self.out_dim = out_dim
        self.residual = residual
        self.l1 = nn.Linear(in_dim, hdim)
        self.l2 = nn.Linear(hdim, out_dim)

        self.pos_embedding = PositionalEncoding(hdim, dropout=0.1)
        TranLayer = nn.TransformerEncoderLayer(d_model=hdim, nhead=nhead, dim_feedforward=1024,
                                               dropout=0.1, activation='gelu')
        self.trans = nn.TransformerEncoder(TranLayer, num_layers=nlayer)
        
        nn.init.xavier_uniform_(self.l1.weight, gain=0.01)
        nn.init.xavier_uniform_(self.l2.weight, gain=0.01)

    def forward(self, x):
        x = x.permute(1,0,2)  # (b,t,c) -> (t,b,c)

        h = self.l1(x)
        h = self.pos_embedding(h)
        h = self.trans(h)
        h = self.l2(h)

        if self.residual:
            x = x[..., :self.out_dim] + h
        else:
            x = h
        x = x.permute(1,0,2)

        return x

    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)