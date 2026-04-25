import os
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import FastMRIDataset, get_file_list
from models import ContrastiveFeatureExtractor, build_reconstruction_model
from losses import compute_metrics, alignment_score, uniformity_score

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def eval_collate_fn(batch):
    us_list = [x["undersampled_images"][0] for x in batch]
    gt_list = [x["ground_truth"] for x in batch]
    H = min(t.shape[-2] for t in us_list)
    W = min(t.shape[-1] for t in us_list)
    us = torch.stack([t[..., :H, :W] for t in us_list])
    gt = torch.stack([t[..., :H, :W] for t in gt_list])
    return us, gt


def cl_collate_fn(batch):
    n_accels = len(batch[0]["undersampled_images"])
    H = min(x["undersampled_images"][0].shape[-2] for x in batch)
    W = min(x["undersampled_images"][0].shape[-1] for x in batch)
    us_list = [
        torch.stack([x["undersampled_images"][i][..., :H, :W] for x in batch])
        for i in range(n_accels)
    ]
    gt = torch.stack([x["ground_truth"][..., :H, :W] for x in batch])
    return us_list, gt


def magnitude(pred: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2 + 1e-8)


def load_extractor(cl_ckpt_path: str, device):
    ckpt = torch.load(cl_ckpt_path, map_location="cpu", weights_only=False)
    cl_args = ckpt.get("args", {})
    ext = ContrastiveFeatureExtractor(
        in_ch=2,
        base_ch=cl_args.get("base_ch", 32),
        latent_dim=cl_args.get("latent_dim", 128),
    )
    ext.load_state_dict(ckpt["model_state"])
    ext = ext.to(device).eval()
    for p in ext.parameters():
        p.requires_grad_(False)
    return ext, cl_args


def load_recon_model(recon_ckpt_path: str, model_name: str,
                     latent_dim: int, device):
    ckpt = torch.load(recon_ckpt_path, map_location="cpu", weights_only=False)
    model = build_reconstruction_model(model_name, latent_dim)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()


@torch.no_grad()
def run_inference(extractor, recon_model, loader, device):
    nmse_list, psnr_list, ssim_list = [], [], []
    pred_images, gt_images = [], []

    for us, gt in loader:
        us, gt = us.to(device), gt.to(device)
        z    = extractor(us)
        pred = recon_model(us, z)
        pred_mag = magnitude(pred)

        for b in range(gt.shape[0]):
            m = compute_metrics(pred_mag[b], gt[b])
            nmse_list.append(m["nmse"])
            psnr_list.append(m["psnr"])
            ssim_list.append(m["ssim"])

        pred_images.append(pred_mag.cpu())
        gt_images.append(gt.cpu())

    return nmse_list, psnr_list, ssim_list, pred_images, gt_images


def exp1_reconstruction(args, extractor, cl_args, device):
    log.info("=== Experiment 1: In-distribution reconstruction ===")
    all_files  = get_file_list(args.data_dir)
    accel_list = [2, 4]
    latent_dim = cl_args.get("latent_dim", 128)
    recon_ckpt = Path(args.recon_dir) / f"{args.model}_accel4" / "best_recon.pth"
    results    = {}

    for accel in accel_list:
        ds = FastMRIDataset(all_files, accel_factors=[accel],
                            num_low_freq=args.num_low_freq,
                            mask_type="random", seed=999)
        loader = DataLoader(ds, batch_size=2, shuffle=False,
                            num_workers=0, collate_fn=eval_collate_fn)

        recon = load_recon_model(str(recon_ckpt), args.model, latent_dim, device)
        nmse_l, psnr_l, ssim_l, _, _ = run_inference(extractor, recon, loader, device)

        results[accel] = {
            "nmse":     float(np.mean(nmse_l)), "nmse_std": float(np.std(nmse_l)),
            "psnr":     float(np.mean(psnr_l)), "psnr_std": float(np.std(psnr_l)),
            "ssim":     float(np.mean(ssim_l)), "ssim_std": float(np.std(ssim_l)),
        }
        log.info(f"  Accel {accel}X  nmse={results[accel]['nmse']:.4f}  "
                 f"psnr={results[accel]['psnr']:.2f}  ssim={results[accel]['ssim']:.4f}")

    return results


def exp3_sampling_pattern(args, extractor, cl_args, device):
    log.info("=== Experiment 3: Sampling pattern robustness ===")
    all_files  = get_file_list(args.data_dir)
    latent_dim = cl_args.get("latent_dim", 128)
    recon_ckpt = Path(args.recon_dir) / f"{args.model}_accel4" / "best_recon.pth"
    recon      = load_recon_model(str(recon_ckpt), args.model, latent_dim, device)
    results    = {}

    for mask_type in ["random", "equispaced"]:
        ds = FastMRIDataset(all_files, accel_factors=[4],
                            num_low_freq=args.num_low_freq,
                            mask_type=mask_type, seed=999)
        loader = DataLoader(ds, batch_size=2, shuffle=False,
                            num_workers=0, collate_fn=eval_collate_fn)
        nmse_l, psnr_l, ssim_l, _, _ = run_inference(extractor, recon, loader, device)
        results[mask_type] = {
            "nmse": float(np.mean(nmse_l)),
            "psnr": float(np.mean(psnr_l)),
            "ssim": float(np.mean(ssim_l)),
        }
        log.info(f"  {mask_type}: nmse={results[mask_type]['nmse']:.4f}  "
                 f"psnr={results[mask_type]['psnr']:.2f}  ssim={results[mask_type]['ssim']:.4f}")
    return results


def add_gaussian_noise(kspace_img: torch.Tensor, snr_db: float) -> torch.Tensor:
    signal_power = (kspace_img ** 2).mean()
    snr_linear   = 10 ** (snr_db / 10)
    noise_power  = signal_power / snr_linear
    noise        = torch.randn_like(kspace_img) * noise_power.sqrt()
    return kspace_img + noise


def exp4_noise_robustness(args, extractor, cl_args, device):
    log.info("=== Experiment 4: Measurement noise robustness ===")
    all_files  = get_file_list(args.data_dir)
    latent_dim = cl_args.get("latent_dim", 128)
    recon_ckpt = Path(args.recon_dir) / f"{args.model}_accel4" / "best_recon.pth"
    recon      = load_recon_model(str(recon_ckpt), args.model, latent_dim, device)

    ds = FastMRIDataset(all_files, accel_factors=[4],
                        num_low_freq=args.num_low_freq,
                        mask_type="random", seed=999)

    snr_levels = [None, 40, 35, 30, 25]
    results    = {}

    for snr in snr_levels:
        label  = "baseline" if snr is None else f"{snr}dB"
        nmse_l, psnr_l, ssim_l = [], [], []
        loader = DataLoader(ds, batch_size=2, shuffle=False,
                            num_workers=0, collate_fn=eval_collate_fn)

        with torch.no_grad():
            for us, gt in loader:
                us, gt = us.to(device), gt.to(device)
                if snr is not None:
                    us = add_gaussian_noise(us, snr)
                z    = extractor(us)
                pred = recon(us, z)
                pred_mag = magnitude(pred)
                for b in range(gt.shape[0]):
                    m = compute_metrics(pred_mag[b], gt[b])
                    nmse_l.append(m["nmse"])
                    psnr_l.append(m["psnr"])
                    ssim_l.append(m["ssim"])

        results[label] = {
            "nmse": float(np.mean(nmse_l)),
            "psnr": float(np.mean(psnr_l)),
            "ssim": float(np.mean(ssim_l)),
        }
        log.info(f"  SNR={label}: nmse={results[label]['nmse']:.4f}  "
                 f"psnr={results[label]['psnr']:.2f}  ssim={results[label]['ssim']:.4f}")
    return results


def exp6_7_latent_analysis(args, extractor, cl_args, device):
    log.info("=== Experiments 6 & 7: Alignment & Uniformity ===")
    all_files     = get_file_list(args.data_dir)
    accel_factors = cl_args.get("accel_factors", [2, 4])

    ds = FastMRIDataset(all_files, accel_factors=accel_factors,
                        num_low_freq=args.num_low_freq,
                        mask_type="random", seed=999, max_slices=200)
    loader = DataLoader(ds, batch_size=4, shuffle=False,
                        num_workers=0, collate_fn=cl_collate_fn)

    all_z_pairs = []
    all_z       = []

    with torch.no_grad():
        for us_batch, _ in loader:
            z_list = [extractor(us.to(device)) for us in us_batch]
            D = len(z_list)
            for d1 in range(D - 1):
                all_z_pairs.append((z_list[d1].cpu(), z_list[d1 + 1].cpu()))
            all_z.append(torch.cat(z_list, dim=0).cpu())

    z_all = torch.cat(all_z, dim=0)
    ca    = alignment_score(all_z_pairs)
    cu    = uniformity_score(z_all)
    log.info(f"  Alignment: {ca:.4f}   Uniformity: {cu:.4f}")
    return {"alignment": ca, "uniformity": cu}


def main(args):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    extractor, cl_args = load_extractor(args.cl_ckpt, device)
    log.info(f"CL extractor loaded. latent_dim={cl_args.get('latent_dim', 128)}")

    all_results = {}

    r1  = exp1_reconstruction(args, extractor, cl_args, device)
    all_results["exp1_reconstruction"] = r1

    r3  = exp3_sampling_pattern(args, extractor, cl_args, device)
    all_results["exp3_sampling"] = r3

    r4  = exp4_noise_robustness(args, extractor, cl_args, device)
    all_results["exp4_noise"] = r4

    r67 = exp6_7_latent_analysis(args, extractor, cl_args, device)
    all_results["exp6_7_latent"] = r67

    with open(out_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"All results saved → {out_dir / 'all_results.json'}")
    log.info("Evaluation complete.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",     type=str, required=True)
    p.add_argument("--cl_ckpt",      type=str, required=True)
    p.add_argument("--recon_dir",    type=str, required=True)
    p.add_argument("--model",        type=str, default="unet",
                   choices=["unet", "d5c5", "miccan", "reconformer"])
    p.add_argument("--num_low_freq", type=int, default=16)
    p.add_argument("--out_dir",      type=str, default="./results")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)