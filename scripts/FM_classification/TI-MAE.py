import torch
import torch.nn as nn

from omegaconf import OmegaConf
from tqdm import tqdm

import os
import sys
from model_utils import TrigonometricPositionalEmbedding, RandomMask
sys.path.append("/".join(os.path.dirname(__file__).split("/")[:-2]))
from data.dataloaders import get_dataloaders_clips
from scripts.utils.str_to_class import str_to_class


class TI_MAE(nn.Module):
    def __init__(self, in_dim: int = 7*18, enc_embed_dim: int = 512,
                 seq_len: int = 240, mask_ratio: float = 0.75,
                 enc_num_heads: int = 4, enc_num_layers: int = 2,
                 dec_embed_dim: int = 256, dec_num_heads: int = 4,
                 dec_num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()

        self.input_embedding = nn.Conv1d(in_dim, enc_embed_dim, kernel_size=3, stride=1, padding=1)
        self.enc_positional_encoding = TrigonometricPositionalEmbedding(enc_embed_dim, seq_len)
        self.num_keep = int(seq_len * (1-mask_ratio))
        self.random_masking = RandomMask(self.num_keep)

        self.layer_norm1 = nn.LayerNorm(enc_embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=enc_embed_dim, nhead=enc_num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_num_layers)
        self.layer_norm2 = nn.LayerNorm(enc_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(enc_embed_dim, 2*enc_embed_dim),
            nn.ReLU(),
            nn.Linear(2*enc_embed_dim, enc_embed_dim)
        )

        self.linear = nn.Linear(enc_embed_dim, dec_embed_dim)
        self.dec_positional_encoding = TrigonometricPositionalEmbedding(dec_embed_dim, seq_len)
        self.dec_layer_norm1 = nn.LayerNorm(dec_embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=dec_embed_dim, nhead=dec_num_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_num_layers)
        self.dec_layer_norm2 = nn.LayerNorm(dec_embed_dim)
        self.dec_mlp = nn.Sequential(
            nn.Linear(dec_embed_dim, 2*dec_embed_dim),
            nn.ReLU(),
            nn.Linear(2*dec_embed_dim, dec_embed_dim)
        )

        self.project = nn.Linear(dec_embed_dim, in_dim)

        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        B, T, C, J = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(B*C*J, 1, T)
        # x = x.view(B, C*J, T)
        x = self.input_embedding(x)
        x = self.enc_positional_encoding(x)
        x = x.permute(0, 2, 1)
        x_masked, mask, ids_restore = self.random_masking(x)

        # # encoder
        x = x_masked + self.encoder(self.layer_norm1(x_masked))
        x = self.dropout(x)
        x = x + self.mlp(self.layer_norm2(x))

        x = self.linear(x)
        # padding masked values
        x = torch.cat([x, torch.zeros((B*C*J, T-self.num_keep, x.shape[-1]))], dim=1)
        x = x.gather(1, ids_restore[:, :, None].expand(-1, -1, x.shape[-1]))
        x = x.permute(0, 2, 1)
        x = self.dec_positional_encoding(x)
        x = x.permute(0, 2, 1)

        x = x + self.decoder(self.dec_layer_norm1(x), self.dec_layer_norm1(x))
        x = self.dropout(x)

        x = x + self.dec_mlp(self.dec_layer_norm2(x))

        x = self.project(x)

        x = x.view(B, b, T, C, J)

        return x



def get_features(pose_sequence, name):
    if name == "distance_mat":
        # Reshape pose_sequence to (B, T, J, 1, C)
        pose_sequence_reshaped = pose_sequence.unsqueeze(3)
        pose_sequence_reshaped = pose_sequence_reshaped[:, :, :14, :, :2]

        # Compute absolute differences for both x and y dimensions
        abs_diff = torch.abs(pose_sequence_reshaped - pose_sequence_reshaped.permute(0, 1, 3, 2, 4))

        # shape: [B, T, C, J, J]
        features = abs_diff.permute(0, 1, 4, 2, 3)

        features = features.float()

        return features
    elif "kinematics" in name:
        # Compute differences for both x and y dimensions
        t = 1 / cfg.dataset.fps
        diff = pose_sequence[:, 1:, :, :-1] - pose_sequence[:, :-1, :, :-1]
        velocities = diff / t
        velocities = torch.concat([torch.zeros(velocities.shape[0], 1, velocities.shape[2], velocities.shape[3]), velocities], dim=1)

        # Compute diff of velocities in x and y
        diff_v = velocities[:, 1:] - velocities[:, :-1]
        accelerations = diff_v / t
        accelerations = torch.concat([torch.zeros(accelerations.shape[0], 1, accelerations.shape[2], accelerations.shape[3]), accelerations], dim=1)

        # calculate the total distance traveled by each joint in the sequence
        distances = torch.zeros(velocities.shape)
        for i in range(1, pose_sequence.shape[1]):
            distances[:, i, :, :] = distances[:, i-1, :, :] + velocities[:, i-1, :, :]*t + 0.5*accelerations[:, i-1, :, :]*t**2

        # distances = np.sqrt(distances[:, :, :, 0]**2 + distances[:, :, :, 1]**2)
        distances = torch.linalg.norm(distances, axis=-1).unsqueeze(-1)

        # shape: [B, T, J, 7]
        features = torch.concat([pose_sequence[:,:,:,:2], velocities, accelerations, distances], dim=3)
        features = features.permute(0,1,3,2)
        # shape: [B, T, 7, J]
        features = features.float()

        if name == "kinematics":
            return features
        elif name == "kinematics_pos":
            return features[:,:,:2,:]
        elif name == "kinematics_vel":
            return features[:,:,:4,:]
        elif name == "kinematics_acc":
            return features[:,:,:6,:]


def train():
    model.train()
    for epoch in range(cfg.hparams.epochs):
        for i, data in enumerate(tqdm(train_loader)):
            inputs, _ = data
            inputs = inputs
            B,b,T,J,C = inputs.shape
            inputs = inputs.view(b*B,T,J,C)
            features = get_features(inputs, "kinematics")
            features = features.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")


if __name__ == "__main__":
    cfg = OmegaConf.load("config/train_TI-MAE.yaml")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TI_MAE(**cfg.model.params)
    dataloaders = get_dataloaders_clips(cfg)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.hparams.learning_rate)
    scheduler = str_to_class(cfg.hparams.scheduler.name)(optimizer, **cfg.hparams.scheduler.params)
    model.to(device)


    train()
