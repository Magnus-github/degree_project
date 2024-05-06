import torch
from torch import nn
from torch.nn import functional as F

from scripts.FM_classification.model_utils import BaseVAE


class VanillaVAE(BaseVAE):
    def __init__(self,
                 in_dim: int,
                 latent_dim: int,
                 hidden_dim: int) -> None:
        super(VanillaVAE, self).__init__()

        
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(nn.Linear(in_dim, 8*hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(8*hidden_dim, 4*hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(4*hidden_dim, 2*hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(2*hidden_dim, hidden_dim),
                                     nn.ReLU(),
                                    )

        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)


        # Build Decoder

        self.decoder_input = nn.Linear(latent_dim, hidden_dim)

        self.decoder = nn.Sequential(nn.Linear(hidden_dim, 2*hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(2*hidden_dim, 4*hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(4*hidden_dim, 8*hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(8*hidden_dim, in_dim),
                                     nn.Tanh(),
                                    )


    
    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [B, N*J*C*t]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B, D]
        :return: (Tensor) [B, N*J*C*t]
        """
        result = self.decoder_input(z)
        result = self.decoder(result)        

        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        pred = self.decode(z)
        return  {'pred': pred, 'Z': z, 'distribution': [mu, log_var]}

    def loss_function(self,
                      pred: torch.Tensor,
                      input: torch.Tensor,
                      distribtion: list[torch.tensor]) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        mu = distribtion[0]
        log_var = distribtion[1]

        # kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        kld_weight = 1

        reconstruction_loss = F.mse_loss(pred, input)


        kldivergence_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = reconstruction_loss + kld_weight * kldivergence_loss
        return {'loss': loss, 'Reconstruction_Loss':reconstruction_loss.detach(), 'KLD':-kldivergence_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int) :
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x) :
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]



