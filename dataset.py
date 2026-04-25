import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional


def _random_mask(width: int, accel: int, num_low_freq: int,
                 rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros(width, dtype=np.float32)
    center = width // 2
    half_lf = num_low_freq // 2
    mask[center - half_lf: center + half_lf] = 1.0

    remaining_idx = np.where(mask == 0)[0]
    num_sample = max(0, width // accel - num_low_freq)
    chosen = rng.choice(remaining_idx, size=num_sample, replace=False)
    mask[chosen] = 1.0
    return mask


def _equispaced_mask(width: int, accel: int, num_low_freq: int) -> np.ndarray:
    mask = np.zeros(width, dtype=np.float32)
    center = width // 2
    half_lf = num_low_freq // 2
    mask[center - half_lf: center + half_lf] = 1.0
    mask[::accel] = 1.0
    return mask


def ifft2c(kspace: np.ndarray) -> np.ndarray:
    return np.fft.ifftshift(
        np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2, -1)), axes=(-2, -1)),
        axes=(-2, -1)
    )


def rss(images: np.ndarray) -> np.ndarray:
    return np.sqrt((np.abs(images) ** 2).sum(axis=0))


def complex_to_2ch(img: np.ndarray) -> np.ndarray:
    return np.stack([img.real, img.imag], axis=0).astype(np.float32)


def apply_mask_to_kspace(kspace: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return kspace * mask[np.newaxis, np.newaxis, :]


class FastMRIDataset(Dataset):
    def __init__(
        self,
        file_paths: List[str],
        accel_factors: List[int] = (2, 4, 6, 8),
        num_low_freq: int = 16,
        mask_type: str = "random",
        seed: Optional[int] = 42,
        max_slices: Optional[int] = None,
    ):
        super().__init__()
        self.accel_factors = list(accel_factors)
        self.num_low_freq = num_low_freq
        self.mask_type = mask_type
        self.seed = seed

        self.samples: List[Tuple[str, int]] = []
        for fpath in file_paths:
            with h5py.File(fpath, "r") as f:
                kspace = f["kspace"]
                n_slices = kspace.shape[0]
            for s in range(n_slices):
                self.samples.append((fpath, s))
                if max_slices and len(self.samples) >= max_slices:
                    break
            if max_slices and len(self.samples) >= max_slices:
                break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        fpath, slice_idx = self.samples[idx]
        rng = np.random.default_rng(
            None if self.seed is None else self.seed + idx
        )

        with h5py.File(fpath, "r") as f:
            kspace_slice = f["kspace"][slice_idx]
            H, W = kspace_slice.shape[-2:]

        gt_images = ifft2c(kspace_slice)
        gt_rss = rss(gt_images)
        gt_max = gt_rss.max() + 1e-8
        gt_norm = (gt_rss / gt_max).astype(np.float32)

        undersampled_list = []
        for accel in self.accel_factors:
            if self.mask_type == "random":
                mask = _random_mask(W, accel, self.num_low_freq, rng)
            else:
                mask = _equispaced_mask(W, accel, self.num_low_freq)

            us_kspace = apply_mask_to_kspace(kspace_slice, mask)
            us_images = ifft2c(us_kspace)
            us_rss = rss(us_images)
            us_rss = (us_rss / gt_max).astype(np.float32)
            us_2ch = complex_to_2ch(
                us_images.mean(axis=0)
            )
            for c in range(2):
                ch_max = np.abs(us_2ch[c]).max() + 1e-8
                us_2ch[c] /= ch_max
            undersampled_list.append(torch.from_numpy(us_2ch))

        return {
            "undersampled_images": undersampled_list,
            "ground_truth": torch.from_numpy(gt_norm),
            "accel_factors": self.accel_factors,
            "slice_idx": slice_idx,
            "volume_path": fpath,
        }


def cl_collate_fn(batch):
    def resize_tensor(t, H, W):
        lead = t.shape[:-2]
        h, w = t.shape[-2], t.shape[-1]
        t = t[..., :min(h, H), :min(w, W)]
        ph = H - t.shape[-2]
        pw = W - t.shape[-1]
        if ph > 0 or pw > 0:
            pad = [0, pw, 0, ph] + [0, 0] * (len(lead))
            t = torch.nn.functional.pad(t, pad[:4])
        return t

    H = min(item["ground_truth"].shape[-2] for item in batch)
    W = min(item["ground_truth"].shape[-1] for item in batch)

    D = len(batch[0]["undersampled_images"])
    us_batch = []
    for d in range(D):
        us_batch.append(
            torch.stack([resize_tensor(item["undersampled_images"][d], H, W) for item in batch])
        )
    gt_batch = torch.stack([resize_tensor(item["ground_truth"], H, W) for item in batch])
    return us_batch, gt_batch


def get_file_list(data_dir: str, extension: str = ".h5") -> List[str]:
    paths = []
    for root, _, files in os.walk(data_dir):
        for f in sorted(files):
            if f.endswith(extension):
                paths.append(os.path.join(root, f))
    return sorted(paths)
