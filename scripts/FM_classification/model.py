import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable



class Attention(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, hid_dim//2),
            nn.Tanh(),
            nn.Linear(hid_dim//2, 1)
        )

    def forward(self, x):#[N,*,c]
        attn = self.attention(x)#[N,*,1]
        attn = torch.softmax(attn, dim=1) 
        x = torch.bmm(torch.transpose(attn, 1, 2), x)
        x = x.squeeze(dim=1) #[N, h]        
        return x#,attn
    
    
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=240):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False).to('cuda' if torch.cuda.is_available() else 'cpu')
        return self.dropout(x)


class STTransformer(nn.Module):
    def __init__(self, joint_in_channels=2, joint_hidden_channels=64, time_window=9, time_step=3, dropout=0.4):
        super().__init__()
        print("DROPOUT", dropout)
        #x
        self.cnn = nn.Conv2d(joint_in_channels, joint_hidden_channels, [1,14],[1,1])
        self.tcn = nn.Conv2d(joint_hidden_channels, joint_hidden_channels, [time_window,1],[time_step,1])
        self.norm = nn.BatchNorm2d(joint_hidden_channels)
        encoder_layer = nn.TransformerEncoderLayer(d_model=joint_hidden_channels, nhead=4, dim_feedforward=4*joint_hidden_channels, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(joint_hidden_channels, joint_hidden_channels)
        self.pe = PositionalEncoding(joint_hidden_channels, dropout=dropout)
        self.linear = nn.Linear(joint_hidden_channels, 3)
        #instance to bag
        # self.attention2 = Attention(2, joint_hidden_channels//2)

    def forward(self, x):
        B, T, c, j, j_ = x.shape # Batch, num_Frames, channels, Joints, Joints

        # split the sequence into subsequences of length t
        t = 240
        x = x[:, T%t:] # cut the sequence to be divisible by t
        K = T//t
        x = x.view(B, K, t, c, j, j_)

        x = x.reshape(B*K*t,c,j,j_)
        x = self.cnn(x) # [B*K*t, c', j, j'] (j'=1)
        # x = F.relu(x)
        x = F.sigmoid(x)
        x = self.dropout(x)
        x = x.view(B*K,t,-1,j)
        x = x.permute(0, 2, 1, 3).contiguous() # [B*K, c', t, j]
        # x = self.tcn(x) # [B*K, c', t', j]
        # x = self.norm(x)
        # # x = F.relu(x)
        # x = F.sigmoid(x)
        # x = self.dropout(x)
        # x = self.tcn(x) # [B*K, c', t'', j]
        # x = self.norm(x)
        # # x = F.relu(x)
        # x = F.sigmoid(x)
        # x = self.dropout(x)

        # transformer layers
        BK, c_, t_, j = x.shape # Batch*num_Clips, hidden_channels, num_Frames/clip, Joints
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(BK*t_, j, c_)
        x = self.transformer_encoder(x) # [B*K*t_, j, c_]
        x = self.attention(x) # [B*K*t_, c_]
        x = x.view(BK, t_, c_)
        x = self.pe(x)
        x = self.transformer_encoder(x) # [B*K, t_, c_]
        x = self.attention(x) # [B*K, c_]

        # clip/instance classification
        clip_cls = self.linear(x) # [B*K, 3]
        # clip_cls = torch.softmax(clip_cls, dim=-1)

        # bag classification using max pooling
        # pool = nn.MaxPool1d(K)
        pool = nn.AvgPool1d(K)
        vid_cls = clip_cls.view(B, K, 3).permute(0, 2, 1).contiguous()
        vid_cls = pool(vid_cls).squeeze(dim=-1)

        # vid_cls = torch.softmax(vid_cls, dim=-1)

        return vid_cls


if __name__ == "__main__":
    example = torch.randn(1, 240*2, 2, 14, 14)

    model = STTransformer()

    out = model(example)