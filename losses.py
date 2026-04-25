import math
import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as skimage_ssim


def cl_mri_loss(
    z_list: list[torch.Tensor],
    temperature: float = 0.1,
) -> torch.Tensor:
    D = len(z_list)
    B = z_list[0].shape[0]
    device = z_list[0].device

    z_all = torch.cat(z_list, dim=0)
    N = D * B

    sim = torch.mm(z_all, z_all.T) / temperature

    pos_mask = torch.zeros(N, N, dtype=torch.bool, device=device)
    for d1 in range(D):
        for d2 in range(D):
            if d1 == d2:
                continue
            for b in range(B):
                i = d1 * B + b
                j = d2 * B + b
                pos_mask[i, j] = True

    self_mask = torch.eye(N, dtype=torch.bool, device=device)

    total_loss = torch.tensor(0.0, device=device)
    n_positives = 0

    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    exp_sim = torch.exp(sim)
    denom = (exp_sim * (~self_mask).float()).sum(dim=1)

    for i in range(N):
        pos_j = pos_mask[i].nonzero(as_tuple=True)[0]
        if len(pos_j) == 0:
            continue
        for j in pos_j:
            total_loss += -sim[i, j] + torch.log(denom[i] + 1e-8)
            n_positives += 1

    if n_positives == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return total_loss / n_positives


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.dim() == 4:
        pred_mag = torch.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2 + 1e-8)
    else:
        pred_mag = pred
    H = min(pred_mag.shape[-2], target.shape[-2])
    W = min(pred_mag.shape[-1], target.shape[-1])
    pred_mag = pred_mag[..., :H, :W]
    target   = target[..., :H, :W]
    return F.l1_loss(pred_mag, target)


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def nmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    p = _to_numpy(pred)
    t = _to_numpy(target)
    return float(np.sum((p - t) ** 2) / (np.sum(t ** 2) + 1e-8))


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    p = _to_numpy(pred)
    t = _to_numpy(target)
    mse = np.mean((p - t) ** 2)
    if mse < 1e-10:
        return 100.0
    max_val = t.max() ** 2
    return float(10 * np.log10(max_val / mse + 1e-8))


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    p = _to_numpy(pred)
    t = _to_numpy(target)
    scores = []
    if p.ndim == 2:
        p = p[np.newaxis]
        t = t[np.newaxis]
    for i in range(p.shape[0]):
        data_range = t[i].max() - t[i].min()
        if data_range < 1e-8:
            data_range = 1.0
        s = skimage_ssim(p[i], t[i], data_range=data_range)
        scores.append(s)
    return float(np.mean(scores))


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    H = min(pred.shape[-2], target.shape[-2])
    W = min(pred.shape[-1], target.shape[-1])
    pred   = pred[..., :H, :W]
    target = target[..., :H, :W]
    return {"nmse": nmse(pred, target), "psnr": psnr(pred, target), "ssim": ssim(pred, target)}


def alignment_score(z_pos_pairs: list[tuple], alpha: float = 2.0) -> float:
    dists = []
    for z, zp in z_pos_pairs:
        d = torch.norm(z - zp, dim=1, p=2).pow(alpha)
        dists.append(d.mean().item())
    return -float(np.mean(dists))


def uniformity_score(z: torch.Tensor, beta: float = 2.0) -> float:
    with torch.no_grad():
        sq_pdist = torch.pdist(z.float(), p=2).pow(2)
        return float(torch.log(torch.exp(-beta * sq_pdist).mean() + 1e-8).item())
