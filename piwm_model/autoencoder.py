from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class PiwmConvVAE(nn.Module):
    """
    Small Lunar Lander VAE with PIWM-style physical latent slots.

    The class is generic: the first k latent dims can be supervised against physical
    state and the rest are free visual dims. The SHIPPED factored model
    (factored_clean_noaug) uses it with latent_dim=32 and the convention
      z[0:2] = (x, y)  world units, injected at inference
      z[2:4] = (cos theta, sin theta)  from the tilt reader
      z[4:]  = scene code (terrain; encoded from the lander-erased frame)
    See the README's latent-layout note.
    """

    def __init__(self, latent_dim: int = 64) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
        )
        self.feature_shape = (256, 6, 9)
        self.feature_dim = 256 * 6 * 9

        self.fc_mu = nn.Linear(self.feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.feature_dim)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_conv(x)
        h = h.reshape(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.reshape(z.size(0), *self.feature_shape)
        recon = self.decoder_conv(h)
        return F.interpolate(recon, size=(100, 150), mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


def piwm_vae_loss(
    recon: torch.Tensor,
    image: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    state: torch.Tensor,
    state_indices: Sequence[int],
    kl_weight: float,
    state_weight: float,
    kl_on_physical: bool = False,
) -> dict[str, torch.Tensor]:
    recon_loss = F.mse_loss(recon, image, reduction="mean")

    k = len(state_indices)
    if state_weight > 0.0:
        target = state[:, list(state_indices)]
        physical_mu = mu[:, :k]
        state_loss = F.mse_loss(physical_mu, target, reduction="mean")
    else:
        state_loss = torch.zeros((), device=image.device)

    if kl_on_physical or k == 0:
        kl_mu = mu
        kl_logvar = logvar
    else:
        kl_mu = mu[:, k:]
        kl_logvar = logvar[:, k:]

    kl_loss = (
        kl_divergence(kl_mu, kl_logvar)
        if kl_mu.numel() > 0
        else torch.zeros((), device=image.device)
    )
    total = recon_loss + kl_weight * kl_loss + state_weight * state_loss

    return {
        "loss": total,
        "recon_loss": recon_loss,
        "kl_loss": kl_loss,
        "state_loss": state_loss,
    }
