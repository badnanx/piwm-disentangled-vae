"""Lander sprite extraction, masking, and compositing utilities.

Uses the purple lander color mask (blue channel dominant) only — not fire/flame.
This means we only erase and replace the lander body, leaving flame pixels
in the background as-is (an accepted approximation).
"""
import torch
import torch.nn.functional as F

_PURPLE_BIAS = 0.051   # 13/255 — matches lander_fully_visible in data.py
_PURPLE_MIN  = 0.10    # minimum blue value to reject dark noise


def soft_purple(img: torch.Tensor, T: float = 0.02) -> torch.Tensor:
    """Differentiable purple mask: img (B,3,H,W) in [0,1] -> (B,H,W) soft in [0,1].
    Used by the position swap-equivariance loss to score where the decoded lander lands."""
    r, g, b = img[:, 0], img[:, 1], img[:, 2]
    return (torch.sigmoid((b - r - _PURPLE_BIAS) / T)
            * torch.sigmoid((b - g - _PURPLE_BIAS) / T)
            * torch.sigmoid((b - _PURPLE_MIN) / T))


def soft_centroid(mask: torch.Tensor) -> torch.Tensor:
    """mask (B,H,W) -> (B,2) [cx, cy] in pixel coords, differentiable."""
    B, H, W = mask.shape
    ys = torch.arange(H, device=mask.device, dtype=mask.dtype).view(1, H, 1)
    xs = torch.arange(W, device=mask.device, dtype=mask.dtype).view(1, 1, W)
    s = mask.sum(dim=(1, 2)).clamp_min(1e-6)
    cx = (mask * xs).sum(dim=(1, 2)) / s
    cy = (mask * ys).sum(dim=(1, 2)) / s
    return torch.stack([cx, cy], dim=-1)


def purple_mask(img_chw: torch.Tensor) -> torch.Tensor:
    """Boolean (H, W) mask of purple lander pixels in a float [0,1] CHW image."""
    r, g, b = img_chw[0], img_chw[1], img_chw[2]
    return (b > r + _PURPLE_BIAS) & (b > g + _PURPLE_BIAS) & (b > _PURPLE_MIN)


def extract_sprite(
    img_chw: torch.Tensor,
    size: int = 32,
    pad: int = 4,
    min_pixels: int = 30,
) -> tuple[torch.Tensor, bool]:
    """
    Extract the lander sprite from a full-frame float CHW tensor.

    Finds the bounding box of purple pixels, expands it by `pad` pixels,
    zeroes non-purple pixels inside the box, then center-pads to (3, size, size).

    Returns:
        sprite  — (3, size, size) tensor, non-lander pixels are 0
        found   — False if fewer than min_pixels purple pixels detected
    """
    mask = purple_mask(img_chw)
    if int(mask.sum()) < min_pixels:
        return torch.zeros(3, size, size, dtype=img_chw.dtype, device=img_chw.device), False

    ys, xs = torch.where(mask)
    H, W = img_chw.shape[1], img_chw.shape[2]
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(H, int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(W, int(xs.max()) + pad + 1)

    crop = img_chw[:, y0:y1, x0:x1].clone()
    crop_mask = mask[y0:y1, x0:x1].float().unsqueeze(0)
    crop = crop * crop_mask

    crop_h, crop_w = y1 - y0, x1 - x0

    # Center-pad (or clip) into a size×size canvas
    sprite = torch.zeros(3, size, size, dtype=img_chw.dtype, device=img_chw.device)
    ch = min(crop_h, size)
    cw = min(crop_w, size)
    dy = max(0, (size - ch) // 2)
    dx = max(0, (size - cw) // 2)
    sprite[:, dy:dy + ch, dx:dx + cw] = crop[:, :ch, :cw]

    return sprite, True


def clean_background(img_chw: torch.Tensor) -> torch.Tensor:
    """Zero out all purple lander pixels in the full image. Returns a new tensor."""
    mask = purple_mask(img_chw)
    result = img_chw.clone()
    result[:, mask] = 0.0
    return result


def paste_sprite(
    bg_chw: torch.Tensor,
    sprite_chw: torch.Tensor,
    cx_px: float,
    cy_px: float,
    size: int = 32,
) -> torch.Tensor:
    """
    Paste sprite centered at pixel (cx_px, cy_px) on background.

    Pixels in the sprite with max channel value > 0.02 replace the background.
    Handles out-of-bounds gracefully by clipping.
    """
    _, H, W = bg_chw.shape
    half = size // 2
    x0, y0 = int(round(cx_px)) - half, int(round(cy_px)) - half
    x1, y1 = x0 + size, y0 + size

    sx0 = max(0, -x0);  sy0 = max(0, -y0)
    ix0 = max(0,  x0);  iy0 = max(0,  y0)
    ix1 = min(W,  x1);  iy1 = min(H,  y1)
    sx1 = sx0 + (ix1 - ix0)
    sy1 = sy0 + (iy1 - iy0)

    if ix1 <= ix0 or iy1 <= iy0:
        return bg_chw.clone()

    result = bg_chw.clone()
    sr = sprite_chw[:, sy0:sy1, sx0:sx1]
    # Soft mask: any channel > 0.02 is considered lander
    mask = (sr.max(dim=0).values > 0.02).float().unsqueeze(0)
    result[:, iy0:iy1, ix0:ix1] = (
        mask * sr + (1.0 - mask) * result[:, iy0:iy1, ix0:ix1]
    )
    return result
