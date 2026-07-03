"""z_lander: a position-free APPEARANCE code for the lander (the lever for crispness + θ readability).

The factored VAE draws the lander from only 4 numbers (x, y, cosθ, sinθ) = pure pose, with zero shape
bandwidth, so the decoder paints the MEAN lander = a soft blob. z_lander adds a few latent dims carrying
the lander's APPEARANCE (silhouette / legs), encoded from the CENTERED lander crop and injected into the
TAIL of the latent (e.g. z[24:32]), so the decoder gets shape detail. Because the crop is centered, the
code carries NO position -> the haze fix and position controllability are untouched.

The tilt question (see header discussion in train_factored_vae):
- derotate=False (v1, recommended first): the crop keeps its tilt, so z_lander carries appearance+tilt and
  the decoder renders a crisp lander AT the true tilt (best for crispness + geometric-θ readability). θ then
  rides in the appearance code rather than being dial-able via z[2:4] — the v1 tradeoff.
- derotate=True: the crop is rotated upright, so z_lander is pure appearance and θ stays dial-able via
  z[2:4], but the decoder must SYNTHESIZE the rotation (which it is weak at) -> risks crisp-but-upright
  tilted landers. (Rotation convention here is unverified; verify before relying on derotate=True.)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LanderApp(nn.Module):
    """24x24 centered lander crop -> app_dim appearance code (deterministic, mirrors the θ-branch conv)."""
    def __init__(self, app_dim: int = 8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(nn.LazyLinear(64), nn.ReLU(), nn.Linear(64, app_dim))

    def forward(self, crop):
        return self.head(self.conv(crop))


def derotate_crop(crop, theta):
    """Rotate each (B,3,H,W) crop by -theta (radians) to upright via affine grid sampling, so the appearance
    code is tilt-free. theta: (B,) radians. NOTE: rotation sign/convention not yet validated against the
    lander's verified θ convention — only used when derotate_lander=True; verify before trusting it."""
    B = crop.size(0)
    cos, sin = torch.cos(theta), torch.sin(theta)        # grid_sample maps output->input, so use +θ here
    mat = torch.zeros(B, 2, 3, device=crop.device, dtype=crop.dtype)
    mat[:, 0, 0] = cos; mat[:, 0, 1] = -sin
    mat[:, 1, 0] = sin; mat[:, 1, 1] = cos
    grid = F.affine_grid(mat, list(crop.size()), align_corners=False)
    return F.grid_sample(crop, grid, align_corners=False, padding_mode="zeros")


class FeatNet(nn.Module):
    """Frozen random-init conv as a SELF-CONTAINED perceptual feature extractor — no pretrained weights, no
    download (so no torch/torchvision-version risk). Random conv features are decent perceptual descriptors
    (edges/texture, cf. deep image prior), and being FROZEN the decoder can't game them. Input (B,3,S,S) ->
    list of feature maps; the feature-matching loss is L2 between the recon-crop's and real-crop's features,
    which rewards the decoded lander having the same fine structure (legs/silhouette) as the real one."""
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.c2 = nn.Conv2d(16, 32, 3, 2, 1)
        self.c3 = nn.Conv2d(32, 32, 3, 2, 1)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        f1 = F.relu(self.c1(x)); f2 = F.relu(self.c2(f1)); f3 = F.relu(self.c3(f2))
        return [f1, f2, f3]


def crop_at(img, centers_px, size):
    """Differentiable batched ROI crop: a size×size window centered at centers_px=(cx,cy) per image
    (cx=col, cy=row). Implemented with integer-indexed SLICING (not grid_sample) so it is fully
    DETERMINISTIC on CUDA — grid_sampler_2d_backward has no deterministic kernel, which would otherwise
    break bitwise reproducibility. Centers are rounded to the nearest pixel and clamped so the window
    stays on-frame (sub-pixel centering dropped — negligible for this small readback crop); gradients
    still flow through `img`. Used to pull the DECODED lander crop at the known position (θ-swap-equiv
    readback / feature-matching)."""
    B, C, H, W = img.shape
    half = size // 2
    cx = centers_px[:, 0].round().long().clamp(half, W - (size - half))
    cy = centers_px[:, 1].round().long().clamp(half, H - (size - half))
    return torch.stack([img[i, :, int(cy[i]) - half:int(cy[i]) - half + size,
                                  int(cx[i]) - half:int(cx[i]) - half + size] for i in range(B)])


def feat_loss(featnet, a, b):
    """Perceptual feature-matching loss = sum of L2 over the frozen-conv feature maps of two crops."""
    return sum(F.mse_loss(x, y) for x, y in zip(featnet(a), featnet(b)))


def flat_sd_z(vae, branch, app):
    """Checkpoint state-dict including the appearance encoder (vae.* / branch.* / app.*)."""
    sd = {f"vae.{k}": v for k, v in vae.state_dict().items()}
    sd.update({f"branch.{k}": v for k, v in branch.state_dict().items()})
    sd.update({f"app.{k}": v for k, v in app.state_dict().items()})
    return sd
