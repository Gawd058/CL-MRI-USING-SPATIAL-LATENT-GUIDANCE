"""
train_reconstruction.py  –  Phase 2: Reconstruction Model Training
------------------------------------------------------------------
Loads the pretrained ContrastiveFeatureExtractor, extracts latent
representations, then trains a downstream DL reconstruction model G
(paper §3.2, Eq. 4).
"""

import os
import json
import argparse
import logging
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import FastMRIDataset, get_file_list, cl_collate_fn
from models import ContrastiveFeatureExtractor, build_reconstruction_model
from losses import reconstruction_loss, compute_metrics, cl_mri_loss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def recon_collate_fn(batch):
    us_list, gt_list = zip(*batch)
    H = min(t.shape[-2] for t in us_list)
    W = min(t.shape[-1] for t in us_list)
    def crop(t, H, W):
        return t[..., :H, :W]
    us = torch.stack([crop(t, H, W) for t in us_list])
    gt = torch.stack([crop(t, H, W) for t in gt_list])
    return us, gt


# ──────────────────────────────────────────────────────────────────────────────
# Single-acceleration dataset wrapper
# ──────────────────────────────────────────────────────────────────────────────

class SingleAccelDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds: FastMRIDataset, accel: int):
        self.base_ds = base_ds
        self.accel = accel
        self.accel_idx = (base_ds.dataset if hasattr(base_ds, 'dataset') else base_ds).accel_factors.index(accel)

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        item = self.base_ds[idx]
        us = item["undersampled_images"][self.accel_idx]   # (2, H, W)
        gt = item["ground_truth"]                          # (H, W)
        return us, gt


# ──────────────────────────────────────────────────────────────────────────────
# Training / validation  (w/ CL-MRI)
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(extractor, recon_model, loader, optimizer, device):
    extractor.eval()
    recon_model.train()
    total_loss = 0.0
    n_batches  = 0

    for us, gt in loader:
        us = us.to(device)   # (B, 2, H, W)
        gt = gt.to(device)   # (B, H, W)

        with torch.no_grad():
            z = extractor(us)   # (B, L)

        optimizer.zero_grad()
        pred = recon_model(us, z)   # pass both xu and z
        gt   = gt[..., :pred.shape[-2], :pred.shape[-1]]
        loss = reconstruction_loss(pred, gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(recon_model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(extractor, recon_model, loader, device):
    import numpy as np
    extractor.eval()
    recon_model.eval()
    total_loss = 0.0
    all_nmse, all_psnr, all_ssim = [], [], []
    n_batches = 0

    for us, gt in loader:
        us = us.to(device)
        gt = gt.to(device)

        z    = extractor(us)
        pred = recon_model(us, z)   # pass both xu and z

        gt   = gt[..., :pred.shape[-2], :pred.shape[-1]]
        loss = reconstruction_loss(pred, gt)
        total_loss += loss.item()
        n_batches  += 1

        pred_mag = torch.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2 + 1e-8)
        for b in range(gt.shape[0]):
            m = compute_metrics(pred_mag[b], gt[b])
            all_nmse.append(m["nmse"])
            all_psnr.append(m["psnr"])
            all_ssim.append(m["ssim"])

    return {
        "val_loss":  total_loss / max(n_batches, 1),
        "nmse":      float(np.mean(all_nmse)),
        "psnr":      float(np.mean(all_psnr)),
        "ssim":      float(np.mean(all_ssim)),
        "nmse_std":  float(np.std(all_nmse)),
        "psnr_std":  float(np.std(all_psnr)),
        "ssim_std":  float(np.std(all_ssim)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Baseline: train WITHOUT contrastive pretraining
# ──────────────────────────────────────────────────────────────────────────────

def train_baseline(args, train_ds, val_ds, device, accel):
    from models import UNet, D5C5, MICCAN, ReconFormer
    import numpy as np

    log.info("=== Training BASELINE (w/o CL-MRI) ===")

    train_single = SingleAccelDataset(train_ds, accel)
    val_single   = SingleAccelDataset(val_ds,   accel)
    train_loader = DataLoader(train_single, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0,
                              pin_memory=False,
                              collate_fn=recon_collate_fn)
    val_loader   = DataLoader(val_single,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0,
                              collate_fn=recon_collate_fn)

    name = args.model.lower()
    if name == "unet":
        model = UNet(in_ch=2)
    elif name == "d5c5":
        model = D5C5(in_ch=2)
    elif name == "miccan":
        model = MICCAN(in_ch=2)
    elif name == "reconformer":
        model = ReconFormer(in_ch=2)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    model = model.to(device)

    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    save_dir = Path(args.save_dir) / f"baseline_{name}_accel{accel}"
    save_dir.mkdir(parents=True, exist_ok=True)

    history  = []
    best_val = float("inf")
    start_epoch = 1

    resume_ckpt = save_dir / "latest_baseline.pth"
    if resume_ckpt.exists():
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", float("inf"))
        history = ckpt.get("history", [])
        log.info(f"Resuming baseline from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        # ── train ──
        model.train()
        tr_loss = 0.0
        for us, gt in train_loader:
            us, gt = us.to(device), gt.to(device)
            optimizer.zero_grad()
            pred = model(us)
            loss = reconstruction_loss(pred, gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= max(len(train_loader), 1)
        scheduler.step()

        # ── validate ──
        model.eval()
        val_loss = 0.0
        all_nmse, all_psnr, all_ssim = [], [], []
        with torch.no_grad():
            for us, gt in val_loader:
                us, gt = us.to(device), gt.to(device)
                pred = model(us)
                val_loss += reconstruction_loss(pred, gt).item()
                pred_mag  = torch.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2 + 1e-8)
                for b in range(gt.shape[0]):
                    m = compute_metrics(pred_mag[b], gt[b])
                    all_nmse.append(m["nmse"])
                    all_psnr.append(m["psnr"])
                    all_ssim.append(m["ssim"])
        val_loss /= max(len(val_loader), 1)

        val_metrics = {
            "val_loss": val_loss,
            "nmse": float(np.mean(all_nmse)),
            "psnr": float(np.mean(all_psnr)),
            "ssim": float(np.mean(all_ssim)),
        }
        history.append({"epoch": epoch, "train_loss": tr_loss, **val_metrics})

        log.info(
            f"[Baseline] Ep {epoch:3d}/{args.epochs}  "
            f"loss={tr_loss:.4f}  nmse={val_metrics['nmse']:.4f}  "
            f"psnr={val_metrics['psnr']:.2f}  ssim={val_metrics['ssim']:.4f}"
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(), "epoch": epoch},
                       save_dir / "best_baseline.pth")
            log.info("  ✓ Best baseline saved")

        torch.save({
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch":           epoch,
            "best_val":        best_val,
            "history":         history,
        }, save_dir / "latest_baseline.pth")

    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return history


# ──────────────────────────────────────────────────────────────────────────────
# CL-MRI WITHOUT Latent Guidance (Perceptual Contrastive Loss)
# ──────────────────────────────────────────────────────────────────────────────

def train_recon_with_cl_loss(args, train_ds, val_ds, extractor, device):
    """
    Trains a baseline reconstruction model using L1 + Contrastive Loss.
    Uses the frozen ContrastiveFeatureExtractor to project reconstructed images 
    into latent space to compute InfoNCE loss across accelerations in the same batch.
    """
    from models import UNet, D5C5, MICCAN, ReconFormer
    import numpy as np

    log.info("=== Training w/ Contrastive Loss (NO Latent Guidance) ===")

    # We MUST use cl_collate_fn because contrastive loss needs multiple accelerations (e.g. 2,4,6) to form positive pairs
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, collate_fn=cl_collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=cl_collate_fn)

    name = args.model.lower()
    if name == "unet":
        model = UNet(in_ch=2)
    elif name == "d5c5":
        model = D5C5(in_ch=2)
    elif name == "miccan":
        model = MICCAN(in_ch=2)
    elif name == "reconformer":
        model = ReconFormer(in_ch=2)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    model = model.to(device)

    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    extractor.eval()  # Keep extractor frozen

    save_dir = Path(args.save_dir) / f"cl_no_guidance_{name}"
    save_dir.mkdir(parents=True, exist_ok=True)

    history  = []
    best_val = float("inf")
    lambda_cl = 0.1  # Weight for contrastive loss

    for epoch in range(1, args.epochs + 1):
        # ── train ──
        model.train()
        tr_l1 = 0.0
        tr_cl = 0.0
        for us_batch, gt_batch in train_loader:
            gt = gt_batch.to(device)
            optimizer.zero_grad()

            z_list = []
            l1_loss_total = 0.0

            for us in us_batch:
                us = us.to(device)
                pred = model(us)
                
                # Crop and compute L1
                pred_cropped = pred[..., :gt.shape[-2], :gt.shape[-1]]
                l1_loss_total += reconstruction_loss(pred_cropped, gt)

                # Project reconstructed image to latent space
                z = extractor(pred)
                z_list.append(z)

            # Contrastive InfoNCE Loss
            cl_loss = cl_mri_loss(z_list, temperature=0.1)
            
            avg_l1 = l1_loss_total / len(us_batch)
            loss = avg_l1 + (lambda_cl * cl_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            tr_l1 += avg_l1.item()
            tr_cl += cl_loss.item()

        tr_l1 /= max(len(train_loader), 1)
        tr_cl /= max(len(train_loader), 1)
        scheduler.step()

        # ── validate ──
        model.eval()
        val_loss = 0.0
        all_nmse, all_psnr, all_ssim = [], [], []
        with torch.no_grad():
            for us_batch, gt_batch in val_loader:
                gt = gt_batch.to(device)
                for us in us_batch:
                    us = us.to(device)
                    pred = model(us)
                    
                    pred_cropped = pred[..., :gt.shape[-2], :gt.shape[-1]]
                    val_loss += reconstruction_loss(pred_cropped, gt).item()
                    
                    pred_mag = torch.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2 + 1e-8)
                    for b in range(gt.shape[0]):
                        m = compute_metrics(pred_mag[b], gt[b])
                        all_nmse.append(m["nmse"])
                        all_psnr.append(m["psnr"])
                        all_ssim.append(m["ssim"])
                        
        val_loss /= max(len(val_loader) * len(us_batch), 1)

        val_metrics = {
            "val_loss": val_loss,
            "nmse": float(np.mean(all_nmse)),
            "psnr": float(np.mean(all_psnr)),
            "ssim": float(np.mean(all_ssim)),
        }
        history.append({"epoch": epoch, "train_l1": tr_l1, "train_cl": tr_cl, **val_metrics})

        log.info(
            f"[CL No Guidance] Ep {epoch:3d}/{args.epochs}  "
            f"L1={tr_l1:.4f}  CL={tr_cl:.4f}  val_L1={val_metrics['val_loss']:.4f}  "
            f"psnr={val_metrics['psnr']:.2f}"
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(), "epoch": epoch},
                       save_dir / "best_cl_no_guidance.pth")
            log.info("  ✓ Best model saved")

    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return history


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Data ────────────────────────────────────────────────────────────────
    all_files = get_file_list(args.data_dir)
    if not all_files:
        raise RuntimeError(f"No .h5 files found in {args.data_dir}")
    log.info(f"Found {len(all_files)} HDF5 files")

    cl_ckpt       = torch.load(args.cl_ckpt, map_location="cpu", weights_only=False)
    cl_args        = cl_ckpt.get("args", {})
    latent_dim     = cl_args.get("latent_dim",    args.latent_dim)
    base_ch        = cl_args.get("base_ch",        32)
    accel_factors  = cl_args.get("accel_factors", [2, 4, 6, 8])
    if args.accel not in accel_factors:
        accel_factors = sorted(set(accel_factors) | {args.accel})

    full_ds = FastMRIDataset(
        file_paths=all_files,
        accel_factors=accel_factors,
        num_low_freq=args.num_low_freq,
        mask_type="random",
        seed=42,
        max_slices=args.max_slices,
    )
    val_size   = max(1, int(0.1 * len(full_ds)))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    sample = full_ds[0]
    _, H, W = sample["undersampled_images"][0].shape

    # ── CL extractor (frozen) ────────────────────────────────────────────────
    extractor = ContrastiveFeatureExtractor(in_ch=2, base_ch=base_ch, latent_dim=latent_dim)
    extractor.load_state_dict(cl_ckpt["model_state"])
    extractor = extractor.to(device)
    for p in extractor.parameters():
        p.requires_grad_(False)
    log.info(f"Loaded CL extractor from {args.cl_ckpt}")

    # ── Reconstruction model (w/ CL-MRI) ────────────────────────────────────
    if not args.baseline_only:
        recon_model = build_reconstruction_model(
            name=args.model,
            latent_dim=latent_dim,
            target_h=H,
            target_w=W,
        ).to(device)
        log.info(
            f"Reconstruction model ({args.model}): "
            f"{sum(p.numel() for p in recon_model.parameters()):,} params"
        )

        # ── Resume from checkpoint if provided ──────────────────────────────
        start_epoch = 1
        best_val    = float("inf")
        history     = []

        if args.resume:
            resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
            recon_model.load_state_dict(resume_ckpt["model_state"])
            start_epoch = resume_ckpt["epoch"] + 1
            best_val    = resume_ckpt.get("val_metrics", {}).get("val_loss", float("inf"))
            log.info(f"Resumed from epoch {resume_ckpt['epoch']} (best_val={best_val:.4f})")

            # Load existing history if available
            save_dir_tmp = Path(args.save_dir) / f"{args.model}_accel{args.accel}"
            hist_path = save_dir_tmp / "history.json"
            if hist_path.exists():
                with open(hist_path) as f:
                    history = json.load(f)

        train_single = SingleAccelDataset(train_ds, args.accel)
        val_single   = SingleAccelDataset(val_ds,   args.accel)
        train_loader = DataLoader(train_single, batch_size=args.batch_size,
                                  shuffle=True,  num_workers=args.num_workers,
                                  pin_memory=torch.cuda.is_available(),
                                  collate_fn=recon_collate_fn)
        val_loader   = DataLoader(val_single,   batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers,
                                  collate_fn=recon_collate_fn)

        optimizer = optim.RMSprop(recon_model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        save_dir = Path(args.save_dir) / f"{args.model}_accel{args.accel}"
        save_dir.mkdir(parents=True, exist_ok=True)

        log.info("=== Training w/ CL-MRI ===")
        for epoch in range(start_epoch, args.epochs + 1):
            tr_loss     = train_one_epoch(extractor, recon_model, train_loader, optimizer, device)
            val_metrics = validate(extractor, recon_model, val_loader, device)
            scheduler.step()

            history.append({"epoch": epoch, "train_loss": tr_loss, **val_metrics})
            log.info(
                f"Epoch {epoch:3d}/{args.epochs}  "
                f"train={tr_loss:.4f}  nmse={val_metrics['nmse']:.4f}±{val_metrics['nmse_std']:.4f}  "
                f"psnr={val_metrics['psnr']:.2f}  ssim={val_metrics['ssim']:.4f}"
            )

            ckpt = {
                "epoch":       epoch,
                "model_state": recon_model.state_dict(),
                "val_metrics": val_metrics,
                "args":        vars(args),
            }
            torch.save(ckpt, save_dir / "latest_recon.pth")
            if val_metrics["val_loss"] < best_val:
                best_val = val_metrics["val_loss"]
                torch.save(ckpt, save_dir / "best_recon.pth")
                log.info("  ✓ Best model saved")
            if epoch % args.save_every == 0:
                torch.save(ckpt, save_dir / f"recon_epoch_{epoch:04d}.pth")
            with open(save_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

    # ── Baseline ─────────────────────────────────────────────────────────────
    if args.train_baseline or args.baseline_only:
        train_baseline(args, train_ds, val_ds, device, args.accel)

    # ── CL-MRI WITHOUT Latent Guidance ───────────────────────────────────────
    if args.cl_loss_no_guidance:
        train_recon_with_cl_loss(args, train_ds, val_ds, extractor, device)

    log.info("Reconstruction training complete.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CL-MRI Phase 2: Reconstruction Training")
    p.add_argument("--data_dir",       type=str,   required=True)
    p.add_argument("--cl_ckpt",        type=str,   required=True)
    p.add_argument("--save_dir",       type=str,   default="./checkpoints/recon")
    p.add_argument("--model",          type=str,   default="d5c5",
                   choices=["unet", "d5c5", "miccan", "reconformer"])
    p.add_argument("--accel",          type=int,   default=8)
    p.add_argument("--num_low_freq",   type=int,   default=16)
    p.add_argument("--epochs",         type=int,   default=100)
    p.add_argument("--batch_size",     type=int,   default=4)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--latent_dim",     type=int,   default=128)
    p.add_argument("--num_workers", type=int, default=2)   # 2 = no extra workers, avoids RAM errors
    p.add_argument("--save_every",     type=int,   default=10)
    p.add_argument("--max_slices",     type=int,   default=None)
    p.add_argument("--train_baseline", action="store_true",
                   help="Also train w/o CL baseline for comparison")
    p.add_argument("--baseline_only",  action="store_true",
                   help="Skip CL recon, only train baseline")
    p.add_argument("--cl_loss_no_guidance", action="store_true",
                   help="Train reconstruction with contrastive loss, but NO latent guidance")
    p.add_argument("--resume",         type=str,   default=None,
                   help="Path to checkpoint to resume CL training from")
                   
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)