import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from typing import Optional
# from scripts.utils.str_to_class import str_to_class

def str_to_class(string: str):
    module_name, object_name = string.split(":")
    module = __import__(module_name, fromlist=[object_name])
    return getattr(module, object_name)

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
    def __init__(self, joint_in_channels=2, joint_hidden_channels=64, num_joints=14, clip_len=240, num_classes=3, time_window=9, time_step=3, dropout=0.4, pool_method: str = None):
        super().__init__()
        #x
        self.cnn = nn.Conv2d(joint_in_channels, joint_hidden_channels, [1,num_joints],[1,1])
        self.tcn = nn.Conv2d(joint_hidden_channels, joint_hidden_channels, [time_window,1],[time_step,1])
        self.norm = nn.BatchNorm2d(joint_hidden_channels)
        encoder_layer = nn.TransformerEncoderLayer(d_model=joint_hidden_channels, nhead=4, dim_feedforward=4*joint_hidden_channels, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(joint_hidden_channels, joint_hidden_channels)
        self.pe = PositionalEncoding(joint_hidden_channels, dropout=dropout, max_len=clip_len)
        self.linear = nn.Linear(joint_hidden_channels, num_classes)
        #instance to bag
        if pool_method == "learnable":
            NotImplementedError
        self.pool_method = pool_method

        self.clip_len = clip_len

        # self.attention2 = Attention(2, joint_hidden_channels//2)

    def forward(self, x):
        B, T, c, j, j_ = x.shape # Batch, num_Frames, channels, Joints, Joints

        # split the sequence into subsequences of length t
        t = self.clip_len
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
        print("SIZE ", x.shape)

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
        clip_cls = self.linear(x) # [B*K, num_classes]
        # clip_cls = torch.softmax(clip_cls, dim=-1)
        # max_ind = clip_cls.argmax(dim=-1).view(B, K)
        
        # bag classification using max pooling
        pool = str_to_class(self.pool_method)(K)
        # pool = torch.nn.MaxPool1d(K)
        vid_cls = clip_cls.view(B, K, clip_cls.shape[-1]).permute(0, 2, 1).contiguous()
        vid_cls = pool(vid_cls).squeeze(dim=-1) # [B,1]
        print(vid_cls.shape)

        # TODO: INVESTIGATE THIS:
        # It seems like i used clip_cls instead of vid_cls for the pooling operation (see if it throws error)
        # with pool(clip_cls) the train loss went down but not with pool(vid_cls)...

        # # pool = str_to_class(self.pool_method)(K)
        # pool = torch.nn.MaxPool1d(K)
        # vid_cls = clip_cls.view(B, K, 3).permute(0, 2, 1).contiguous()
        # vid_cls = pool(clip_cls).squeeze(dim=-1) # [B,1]

        return vid_cls


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model=64, clip_len=240):
        super(LearnablePositionalEncoding, self).__init__()
        self.pe = nn.Parameter(torch.randn(clip_len, d_model))

    def forward(self, x):
        return x + self.pe


class TimeFormer(torch.nn.Module):
    def __init__(self, joint_in_channels=7, joint_hidden_channels=64, num_encoder_layers=2, num_heads=4, num_joints=18, clip_len=240, num_classes=3, dropout=0.4, pool_method: str = None):
        super(TimeFormer, self).__init__()
        # self.cnn = nn.Conv1d(joint_in_channels, joint_hidden_channels, num_joints)
        # self.norm = nn.BatchNorm1d(joint_hidden_channels)
        joint_hidden_channels = joint_in_channels*num_joints
        self.pe = LearnablePositionalEncoding(joint_hidden_channels, clip_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=joint_hidden_channels, nhead=num_heads, dim_feedforward=4*joint_hidden_channels, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.attention = Attention(joint_hidden_channels, joint_hidden_channels)
        self.dropout = nn.Dropout(dropout)
        hidden_dim_mlp = 8*joint_hidden_channels
        if hidden_dim_mlp < 64:
            hidden_dim_mlp = 64
        self.mlp = nn.Sequential(
            nn.Linear(joint_hidden_channels, hidden_dim_mlp),
            nn.ReLU(),
            nn.Linear(hidden_dim_mlp, hidden_dim_mlp//2),
            nn.ReLU(),
            nn.Linear(hidden_dim_mlp//2, hidden_dim_mlp//4),
            nn.ReLU(),
            nn.Linear(hidden_dim_mlp//4, num_classes)
        )
        self.pool_method = pool_method
        self.clip_len = clip_len

    def forward(self, x, **kwargs):
        B, T, c, j = x.shape
        # split the sequence into subsequences of length t
        t = self.clip_len
        stride = t
        x = x.unfold(1, t, stride).permute(0,1,4,2,3) # [B, K, t, c, j]
        # x = x[:, T%t:] # cut the sequence to be divisible by t
        # K = T//t
        # x = x.view(B, K, t, c, j)
        B, K, t, c, j = x.shape
        x = x.reshape(B*K,t,c*j)


        # embed the joint dimension with CNN layer 
        # x = self.cnn(x)
        # x = F.sigmoid(x)
        # x = self.dropout(x)
        # x = x.view(B*K, t, -1)

        # temporal transformer
        BK, t, c_ = x.shape
        x = self.pe(x)
        x = self.transformer_encoder(x)
        x = F.gelu(x)
        x = self.attention(x)
        x = self.dropout(x)

        # x = x.reshape(B*K, -1)

        clip_cls = self.mlp(x)

        # pool = nn.MaxPool1d(K)
        pool = str_to_class(self.pool_method)(K)
        vid_cls = clip_cls.view(B, K, -1).permute(0, 2, 1)
        vid_cls = pool(vid_cls).squeeze(dim=-1)

        return vid_cls
    

import torch_geometric.nn as gnn

class GCN_TimeFormer(nn.Module):
    def __init__(self, joint_in_channels=7, joint_hidden_channels=64, num_encoder_layers=2, num_heads=4, num_joints=18, clip_len=240, num_classes=3, dropout=0.4, pool_method: str = None):
        super(GCN_TimeFormer, self).__init__()

        self.gcn = gnn.GCNConv(joint_in_channels, joint_hidden_channels)
        joint_hidden_channels = joint_hidden_channels*num_joints
        self.pe = LearnablePositionalEncoding(joint_hidden_channels, clip_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=joint_hidden_channels, nhead=num_heads, dim_feedforward=4*joint_hidden_channels, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.attention = Attention(joint_hidden_channels, joint_hidden_channels)
        self.dropout = nn.Dropout(dropout)
        hidden_dim_mlp = 64#8*joint_hidden_channels
        if hidden_dim_mlp < 64:
            hidden_dim_mlp = 64
        self.mlp = nn.Sequential(
            nn.Linear(joint_hidden_channels, hidden_dim_mlp),
            nn.ReLU(),
            nn.Linear(hidden_dim_mlp, hidden_dim_mlp//2),
            nn.ReLU(),
            nn.Linear(hidden_dim_mlp//2, hidden_dim_mlp//4),
            nn.ReLU(),
            nn.Linear(hidden_dim_mlp//4, num_classes)
        )
        self.pool_method = pool_method
        self.clip_len = clip_len

    def forward(self, x, **kwargs):
        edges = kwargs["edges"]
        B, T, c, j = x.shape
        # split the sequence into subsequences of length t
        t = self.clip_len
        x = x[:, T%t:]
        K = T//t
        x = x.view(B, K, t, c, j)
        x = x.reshape(B*K*t,c,j)
        x = x.permute(0, 2, 1)

        # embed the joint dimension with GCN layer
        x = self.gcn(x, edges)
        x = F.sigmoid(x)
        x = self.dropout(x)
        x = x.view(B*K, t, -1)

        # temporal transformer
        BK, t, c_ = x.shape
        x = self.pe(x)
        x = self.transformer_encoder(x)
        x = F.gelu(x)
        x = self.attention(x)
        x = self.dropout(x)
        
        clip_cls = self.mlp(x)

        pool = nn.MaxPool1d(K)
        vid_cls = clip_cls.view(B, K, -1).permute(0, 2, 1)
        vid_cls = pool(vid_cls).squeeze(dim=-1)

        return vid_cls
    


class SMNN(torch.nn.Module):
    def __init__(self, joint_in_channels: int = 4,
                 joint_hidden_channels: int = 50,
                 num_joints: int = 14,
                 clip_len: int = 50,
                 clip_overlap: int = 30,
                 num_classes: int = 2,
                 dropout: float = 0.2,
                 pool_method: Optional[str] = None) -> None:
        super().__init__()

        in_channels = joint_in_channels * num_joints * clip_len
        self.fc = nn.Linear(in_channels, joint_hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(joint_hidden_channels, 2*joint_hidden_channels)
        self.fc3 = nn.Linear(2*joint_hidden_channels, num_classes)

        self.clip_len = clip_len
        self.stride = clip_len - clip_overlap
        self.pool_method = pool_method

    def forward(self, x, **kwargs):
        t = self.clip_len
        stride = t
        x = x.unfold(1, t, stride).permute(0,1,4,2,3)
        B, K, t, c, j = x.shape
        x = x.reshape(B*K,t*c*j)

        x = self.fc(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = nn.PReLU()(x)
        clip_cls = self.fc3(x).reshape(B, K, -1).permute(0, 2, 1)

        if self.pool_method is None:
            pool = nn.AvgPool1d(K)
        else:
            pool = str_to_class(self.pool_method)(K)
        vid_cls = pool(clip_cls).squeeze(dim=-1)

        return vid_cls
    


class TimeConvNet(nn.Module):
    def __init__(self, joint_in_channels: int = 4,
                 joint_hidden_channels: int = 128,
                 num_joints: int = 14,
                 kernel_size: int = 120,
                 rel_stride: float = 0.5,
                 clip_len: int = 360,
                 clip_overlap: int = 180,
                 num_classes: int = 2,
                 dropout: float = 0.2) -> None:
        super().__init__()


        in_channels = joint_in_channels * num_joints
        conv_stride = int(kernel_size * rel_stride)
        self.tcn = nn.Conv1d(in_channels, joint_hidden_channels, kernel_size=kernel_size, stride=conv_stride)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(joint_hidden_channels, num_classes)

        self.clip_len = clip_len
        self.stride = clip_len - clip_overlap

    def forward(self, x, **kwargs):
        t = self.clip_len
        stride = t
        x = x.unfold(1, t, stride).permute(0,1,4,2,3)
        B, K, t, c, j = x.shape
        x = x.reshape(B*K,t,c*j).permute(0,2,1)

        x = self.tcn(x)
        x = F.relu(x)
        x = self.dropout(x)

        pool = nn.AvgPool1d(x.shape[-1])
        x = pool(x).squeeze(dim=-1)

        clip_cls = self.fc(x).reshape(B, K, -1).permute(0, 2, 1)

        pool = nn.AvgPool1d(K)
        vid_cls = pool(clip_cls).squeeze(dim=-1)

        return vid_cls


        


if __name__ == "__main__":
    example = torch.randn(20, 240*3, 2, 14, 14)
    cfg = {"in_params":
            {
               "joint_in_channels": 7,
                "joint_hidden_channels": 16,
                "num_encoder_layers": 1,
                "num_joints": 18,
                "clip_len": 480,
                "num_classes": 2,
                "dropout": 0.6,
                "pool_method": "scripts.FM_classification.model_utils:MaxPool"
                }
            }

    # model = TimeFormer(**cfg["in_params"])
    model = TimeConvNet()
    # model = SMNN()

    example = torch.randn(1, 4210, 4, 14)

    out = model(example)
