import os
import json
import argparse
import logging
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import FastMRIDataset, cl_collate_fn, get_file_list
from models import ContrastiveFeatureExtractor
from losses import cl_mri_loss, alignment_score, uniformity_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def train_one_epoch(model, loader, optimizer, device, temperature):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for us_batch, _ in loader:
        optimizer.zero_grad()

        z_list = []
        for us in us_batch:
            us = us.to(device)
            z = model(us)
            z_list.append(z)

        loss = cl_mri_loss(z_list, temperature=temperature)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, device, temperature):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    align_scores, unif_scores = [], []

    for us_batch, _ in loader:
        z_list = []
        for us in us_batch:
            z = model(us.to(device))
            z_list.append(z)

        loss = cl_mri_loss(z_list, temperature=temperature)
        total_loss += loss.item()
        n_batches += 1

        for d in range(len(z_list) - 1):
            align_scores.append((z_list[d], z_list[d + 1]))

        z_all = torch.cat(z_list, dim=0)
        unif_scores.append(uniformity_score(z_all))

    ca = alignment_score(align_scores) if align_scores else 0.0
    cu = float(sum(unif_scores) / max(len(unif_scores), 1))

    return total_loss / max(n_batches, 1), ca, cu


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    all_files = get_file_list(args.data_dir)
    if not all_files:
        raise RuntimeError(f"No .h5 files found in {args.data_dir}")
    log.info(f"Found {len(all_files)} HDF5 files")

    full_ds = FastMRIDataset(
        file_paths=all_files,
        accel_factors=args.accel_factors,
        num_low_freq=args.num_low_freq,
        mask_type="random",
        seed=42,
        max_slices=args.max_slices,
    )
    val_size = max(1, int(0.1 * len(full_ds)))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    log.info(f"Train slices: {len(train_ds)}   Val slices: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=cl_collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=cl_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    model = ContrastiveFeatureExtractor(
        in_ch=2,
        base_ch=args.base_ch,
        latent_dim=args.latent_dim,
    ).to(device)
    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "alignment": [], "uniformity": []}
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, args.temperature)
        val_loss, ca, cu = validate(model, val_loader, device, args.temperature)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["alignment"].append(ca)
        history["uniformity"].append(cu)

        log.info(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train={tr_loss:.4f}  val={val_loss:.4f}  "
            f"align={ca:.4f}  uniform={cu:.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
            "args": vars(args),
        }
        torch.save(ckpt, save_dir / "latest_cl.pth")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, save_dir / "best_cl.pth")
            log.info(f"  ✓ New best CL model saved (val_loss={best_val:.4f})")

        if epoch % args.save_every == 0:
            torch.save(ckpt, save_dir / f"cl_epoch_{epoch:04d}.pth")

        with open(save_dir / "cl_history.json", "w") as f:
            json.dump(history, f, indent=2)

    log.info("Contrastive pretraining complete.")
    return model, history


def parse_args():
    p = argparse.ArgumentParser(description="CL-MRI Phase 1: Contrastive Pretraining")
    p.add_argument("--data_dir",      type=str,   required=True,        help="Root dir of fastMRI .h5 files")
    p.add_argument("--save_dir",      type=str,   default="./checkpoints/cl")
    p.add_argument("--accel_factors", type=int,   nargs="+", default=[2, 4, 6, 8])
    p.add_argument("--num_low_freq",  type=int,   default=16)
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch_size",    type=int,   default=4)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--temperature",   type=float, default=0.1)
    p.add_argument("--latent_dim",    type=int,   default=128)
    p.add_argument("--base_ch",       type=int,   default=32)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--save_every",    type=int,   default=10)
    p.add_argument("--max_slices",    type=int,   default=None,         help="Cap total slices (debug)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
