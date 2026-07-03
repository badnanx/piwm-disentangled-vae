import glob
import os
import random
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def lander_fully_visible(img_hwc: np.ndarray, min_pixels: int = 30) -> bool:
    """Return True if the lander is fully on-screen in a uint8 HWC image.

    Uses the purple color mask (b > r+13, b > g+13 in 0-255 space).
    Fails if fewer than min_pixels purple pixels exist or if the lander
    bounding box touches any image edge (indicating clipping).
    """
    r, g, b = img_hwc[:, :, 0], img_hwc[:, :, 1], img_hwc[:, :, 2]
    mask = (b.astype(np.int16) > r.astype(np.int16) + 13) & \
           (b.astype(np.int16) > g.astype(np.int16) + 13)
    if mask.sum() < min_pixels:
        return False
    rows, cols = np.where(mask)
    H, W = img_hwc.shape[:2]
    return int(rows.min()) > 0 and int(rows.max()) < H - 1 and \
           int(cols.min()) > 0 and int(cols.max()) < W - 1


@dataclass(frozen=True)
class StateSpec:
    indices: tuple[int, ...] = (0, 1, 4)
    names: tuple[str, ...] = ("x", "y", "theta")

    @classmethod
    def from_indices(cls, indices: Iterable[int]) -> "StateSpec":
        names_by_index = {
            0: "x",
            1: "y",
            2: "vx",
            3: "vy",
            4: "theta",
            5: "omega",
            6: "left_leg",
            7: "right_leg",
        }
        idx = tuple(int(i) for i in indices)
        return cls(indices=idx, names=tuple(names_by_index.get(i, f"s{i}") for i in idx))


class LunarFrameDataset(Dataset):
    """Expose individual Lunar Lander frames from trajectory `.npz` files."""

    def __init__(
        self,
        data_dir: str,
        state_key: str = "states",
        max_files: Optional[int] = None,
        max_frames_per_file: Optional[int] = None,
        require_visible: bool = False,
        visible_min_pixels: int = 30,
        file_seed: Optional[int] = None,
    ) -> None:
        self.data_dir = data_dir
        self.state_key = state_key
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if max_files is not None:
            if file_seed is not None:
                random.Random(file_seed).shuffle(self.files)
            self.files = self.files[:max_files]
        if not self.files:
            raise ValueError(f"No .npz files found in {data_dir}")

        self.index: list[tuple[int, int]] = []
        for file_idx, path in enumerate(self.files):
            with np.load(path) as data:
                if "imgs" not in data:
                    raise KeyError(f"{path} is missing key 'imgs'")
                if state_key not in data:
                    raise KeyError(f"{path} is missing key '{state_key}'")
                imgs = data["imgs"]
                n_frames = int(imgs.shape[0])
                if max_frames_per_file is not None:
                    n_frames = min(n_frames, max_frames_per_file)
                for frame_idx in range(n_frames):
                    if require_visible and not lander_fully_visible(imgs[frame_idx], visible_min_pixels):
                        continue
                    self.index.append((file_idx, frame_idx))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        file_idx, frame_idx = self.index[idx]
        path = self.files[file_idx]

        with np.load(path) as data:
            img = data["imgs"][frame_idx].astype(np.float32) / 255.0
            state = data[self.state_key][frame_idx].astype(np.float32)
            if "acts" in data:
                acts = data["acts"]
                action = int(acts[min(frame_idx, len(acts) - 1)])
            else:
                action = 0

        return {
            "image": torch.from_numpy(img).permute(2, 0, 1),
            "state": torch.from_numpy(state),
            "action": torch.tensor(action, dtype=torch.long),
        }


class LunarTripletDataset(Dataset):
    """
    Consecutive triplets for PIWM second-order dynamics.

    Returns frames/states at t, t+1, t+2 and action at t+1, treated as the
    transition action from t+1 to t+2.
    """

    def __init__(
        self,
        data_dir: str,
        state_key: str = "states",
        max_files: Optional[int] = None,
        max_triplets_per_file: Optional[int] = None,
        require_visible: bool = False,
        visible_min_pixels: int = 30,
        file_seed: Optional[int] = None,
    ) -> None:
        self.data_dir = data_dir
        self.state_key = state_key
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if max_files is not None:
            if file_seed is not None:
                random.Random(file_seed).shuffle(self.files)
            self.files = self.files[:max_files]
        if not self.files:
            raise ValueError(f"No .npz files found in {data_dir}")

        self.index: list[tuple[int, int]] = []
        for file_idx, path in enumerate(self.files):
            with np.load(path) as data:
                if "imgs" not in data:
                    raise KeyError(f"{path} is missing key 'imgs'")
                if state_key not in data:
                    raise KeyError(f"{path} is missing key '{state_key}'")
                imgs = data["imgs"]
                n_triplets = max(0, int(imgs.shape[0]) - 2)
                if max_triplets_per_file is not None:
                    n_triplets = min(n_triplets, max_triplets_per_file)
                for t in range(n_triplets):
                    if require_visible:
                        if not all(
                            lander_fully_visible(imgs[t + i], visible_min_pixels)
                            for i in range(3)
                        ):
                            continue
                    self.index.append((file_idx, t))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        file_idx, t = self.index[idx]
        path = self.files[file_idx]

        with np.load(path) as data:
            imgs = data["imgs"][t : t + 3].astype(np.float32) / 255.0
            states = data[self.state_key][t : t + 3].astype(np.float32)
            if "acts" in data:
                acts = data["acts"]
                action = int(acts[min(t + 1, len(acts) - 1)])
            else:
                action = 0

        imgs_t = torch.from_numpy(imgs).permute(0, 3, 1, 2)
        states_t = torch.from_numpy(states)
        return {
            "image_t": imgs_t[0],
            "image_t1": imgs_t[1],
            "image_t2": imgs_t[2],
            "state_t": states_t[0],
            "state_t1": states_t[1],
            "state_t2": states_t[2],
            "action_t1": torch.tensor(action, dtype=torch.long),
        }


class LatentConditionDataset(Dataset):
    """Dataset of exported autoencoder latents and physical conditions."""

    def __init__(
        self,
        npz_path: str,
        latent_mean: Optional[np.ndarray] = None,
        latent_std: Optional[np.ndarray] = None,
        cond_mean: Optional[np.ndarray] = None,
        cond_std: Optional[np.ndarray] = None,
    ) -> None:
        data = np.load(npz_path)
        self.z = data["z"].astype(np.float32)
        self.cond = data["cond"].astype(np.float32)

        if latent_mean is None:
            latent_mean = self.z.mean(axis=0, keepdims=True)
        if latent_std is None:
            latent_std = self.z.std(axis=0, keepdims=True)
        if cond_mean is None:
            cond_mean = self.cond.mean(axis=0, keepdims=True)
        if cond_std is None:
            cond_std = self.cond.std(axis=0, keepdims=True)

        self.latent_mean = latent_mean.astype(np.float32)
        self.latent_std = np.maximum(latent_std.astype(np.float32), 1e-6)
        self.cond_mean = cond_mean.astype(np.float32)
        self.cond_std = np.maximum(cond_std.astype(np.float32), 1e-6)

        self.z_norm = ((self.z - self.latent_mean) / self.latent_std).astype(np.float32)
        self.cond_norm = ((self.cond - self.cond_mean) / self.cond_std).astype(np.float32)

    def __len__(self) -> int:
        return len(self.z)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "z_norm": torch.from_numpy(self.z_norm[idx]),
            "cond_norm": torch.from_numpy(self.cond_norm[idx]),
            "z": torch.from_numpy(self.z[idx]),
            "cond": torch.from_numpy(self.cond[idx]),
        }


def one_hot(actions: np.ndarray, num_actions: int) -> np.ndarray:
    out = np.zeros((len(actions), num_actions), dtype=np.float32)
    out[np.arange(len(actions)), actions.astype(np.int64)] = 1.0
    return out
