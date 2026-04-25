import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

from dataset import FastMRIDataset, get_file_list
from models import ContrastiveFeatureExtractor, build_reconstruction_model


def recon_collate_fn(batch):
    us_list, gt_list = zip(*batch)
    H = min(t.shape[-2] for t in us_list)
    W = min(t.shape[-1] for t in us_list)
    def crop(t): return t[..., :H, :W]
    return torch.stack([crop(t) for t in us_list]), torch.stack([crop(t) for t in gt_list])


class SingleAccelDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, accel):
        self.base_ds = base_ds
        af = base_ds.accel_factors
        self.accel_idx = af.index(accel)

    def __len__(self): return len(self.base_ds)

    def __getitem__(self, idx):
        item = self.base_ds[idx]
        return item["undersampled_images"][self.accel_idx], item["ground_truth"]


def magnitude(pred):
    return torch.sqrt(pred[:, 0] ** 2 + pred[:, 1] ** 2 + 1e-8)


def normalize(img):
    mn, mx = img.min(), img.max()
    return (img - mn) / (mx - mn + 1e-8)


def load_extractor(cl_ckpt_path, device):
    ckpt = torch.load(cl_ckpt_path, map_location="cpu", weights_only=False)
    cl_args = ckpt.get("args", {})
    ext = ContrastiveFeatureExtractor(
        in_ch=2,
        base_ch=cl_args.get("base_ch", 32),
        latent_dim=cl_args.get("latent_dim", 128),
    )
    ext.load_state_dict(ckpt["model_state"])
    return ext.to(device).eval(), cl_args


def load_recon(ckpt_path, model_name, latent_dim, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_reconstruction_model(model_name, latent_dim)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    extractor, cl_args = load_extractor(args.cl_ckpt, device)
    latent_dim = cl_args.get("latent_dim", 128)

    cl_ckpt_path = Path(args.recon_dir) / f"{args.model}_accel{args.accel}" / "best_recon.pth"
    cl_model = load_recon(str(cl_ckpt_path), args.model, latent_dim, device)

    base_ckpt_path = Path(args.recon_dir) / f"baseline_{args.model}_accel{args.accel}" / "best_baseline.pth"
    from models import UNet
    baseline_model = UNet(in_ch=2, out_ch=2).to(device).eval()
    base_ckpt = torch.load(str(base_ckpt_path), map_location="cpu", weights_only=False)
    baseline_model.load_state_dict(base_ckpt["model_state"])

    all_files = get_file_list(args.data_dir)
    full_ds = FastMRIDataset(
        file_paths=all_files,
        accel_factors=[args.accel],
        num_low_freq=16,
        mask_type="random",
        seed=42,
    )
    single_ds = SingleAccelDataset(full_ds, args.accel)
    loader = DataLoader(single_ds, batch_size=1, shuffle=False,
                        collate_fn=recon_collate_fn, num_workers=0)

    samples = []
    with torch.no_grad():
        for i, (us, gt) in enumerate(loader):
            if i >= args.n_samples:
                break
            us, gt = us.to(device), gt.to(device)

            z = extractor(us)
            pred_cl = magnitude(cl_model(us, z))[0].cpu().numpy()

            pred_base = magnitude(baseline_model(us))[0].cpu().numpy()

            us_mag = magnitude(us)[0].cpu().numpy()

            gt_np = gt[0].cpu().numpy()

            samples.append((us_mag, pred_base, pred_cl, gt_np))

    n = len(samples)
    fig, axes = plt.subplots(4, n, figsize=(n * 4, 16))
    row_labels = ["Undersampled Input", "Baseline (w/o CL)", "CL-MRI (w/ CL)", "Ground Truth"]

    for col, (us_mag, pred_base, pred_cl, gt_np) in enumerate(samples):
        imgs = [us_mag, pred_base, pred_cl, gt_np]
        for row, img in enumerate(imgs):
            ax = axes[row, col] if n > 1 else axes[row]
            ax.imshow(normalize(img), cmap="gray")
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=12, labelpad=10)

    fig2, axes2 = plt.subplots(2, n, figsize=(n * 4, 8))
    for col, (_, pred_base, pred_cl, gt_np) in enumerate(samples):
        H = min(pred_base.shape[0], gt_np.shape[0])
        W = min(pred_base.shape[1], gt_np.shape[1])
        err_base = np.abs(gt_np[:H, :W] - pred_base[:H, :W])
        err_cl   = np.abs(gt_np[:H, :W] - pred_cl[:H, :W])
        ax0 = axes2[0, col] if n > 1 else axes2[0]
        ax1 = axes2[1, col] if n > 1 else axes2[1]
        ax0.imshow(err_base, cmap="hot", vmin=0, vmax=err_base.max())
        ax1.imshow(err_cl,   cmap="hot", vmin=0, vmax=err_base.max())
        ax0.axis("off")
        ax1.axis("off")
        if col == 0:
            ax0.set_ylabel("Error: Baseline", fontsize=12)
            ax1.set_ylabel("Error: CL-MRI",   fontsize=12)

    fig.suptitle(f"CL-MRI vs Baseline  |  UNet  |  {args.accel}X Acceleration", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "qualitative_comparison.png", dpi=150, bbox_inches="tight")

    fig2.suptitle("Error Maps (brighter = worse)", fontsize=14)
    fig2.tight_layout()
    fig2.savefig(out_dir / "error_maps.png", dpi=150, bbox_inches="tight")

    print(f"\n✓ Saved to {out_dir}/qualitative_comparison.png")
    print(f"✓ Saved to {out_dir}/error_maps.png")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",  type=str, required=True)
    p.add_argument("--cl_ckpt",   type=str, default="./checkpoints/cl/best_cl.pth")
    p.add_argument("--recon_dir", type=str, default="./checkpoints/recon")
    p.add_argument("--model",     type=str, default="unet")
    p.add_argument("--accel",     type=int, default=4)
    p.add_argument("--out_dir",   type=str, default="./results")
    p.add_argument("--n_samples", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
