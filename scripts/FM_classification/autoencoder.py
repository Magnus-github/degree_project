import torch
import torch.nn as nn

from omegaconf import OmegaConf
from tqdm import tqdm

import os
import sys
sys.path.append("/".join(os.path.dirname(__file__).split("/")[:-2]))
# print("/".join(os.path.dirname(__file__).split("/")[:-2]))
from data.dataloaders import get_dataloaders


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model=64, clip_len=240):
        super(LearnablePositionalEncoding, self).__init__()
        self.pe = nn.Parameter(torch.randn(clip_len, d_model))

    def forward(self, x):
        return x + self.pe


class SpatialAutoencoder(torch.nn.Module):
    def __init__(self, input_dim: int = 126):
        super(SpatialAutoencoder, self).__init__()

        

        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            # nn.ReLU(),
            # nn.Linear(16, 4)
        )

        self.decoder = nn.Sequential(
            # nn.Linear(4, 16),
            # nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        B, T, C, J = x.shape
        x = x.view(B*T, C*J)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(B, T, C, J)
        return x
    

def train_spatial_autoencoder():
    cfg = OmegaConf.load("config/train_AE.yaml")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpatialAutoencoder()
    model.to(device)
    model.train()
    dataloaders = get_dataloaders(cfg)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.hparams.lr)

    epochs = cfg.hparams.epochs


    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(tqdm(train_loader)):
            inputs = data
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, loss: {running_loss/len(train_loader)}')

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(tqdm(val_loader)):
                inputs = data
                inputs = inputs.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                running_val_loss += loss.item()
            print(f'Epoch {epoch+1}, val_loss: {running_val_loss/len(val_loader)}')
    print('Finished Training')
    return model

if __name__ == "__main__":
    model = train_spatial_autoencoder()
    # torch.save(model.state_dict(), "model/spatial_autoencoder.pth")
