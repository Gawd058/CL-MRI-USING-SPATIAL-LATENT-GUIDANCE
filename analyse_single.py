import h5py, os, json, argparse, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings("ignore")


def print_structure(f, indent=0):
    for key in f.keys():
        item = f[key]
        if isinstance(item, h5py.Dataset):
            print(f"{'  '*indent}📦 {key:30s}  shape={item.shape}  dtype={item.dtype}")
        elif isinstance(item, h5py.Group):
            print(f"{'  '*indent}📁 {key}/")
            print_structure(item, indent+1)


def collect_metadata(f):
    meta = {}
    meta["attrs"] = {k: (v.decode() if isinstance(v, bytes) else str(v))
                     for k, v in f.attrs.items()}

    if "kspace" in f:
        k = f["kspace"]
        meta["kspace_shape"] = list(k.shape)
        meta["kspace_dtype"] = str(k.dtype)
        meta["kspace_nbytes_gb"] = round(k.nbytes / 1e9, 2)

        if k.ndim == 5:
            meta["layout"] = "5D: (volumes, slices, coils, H, W)"
            meta["n_volumes"] = k.shape[0]
            meta["n_slices_per_vol"] = k.shape[1]
            meta["n_coils"] = k.shape[2]
            meta["H"] = k.shape[3]
            meta["W"] = k.shape[4]
        elif k.ndim == 4:
            meta["layout"] = "4D: (slices, coils, H, W)"
            meta["n_volumes"] = 1
            meta["n_slices_per_vol"] = k.shape[0]
            meta["n_coils"] = k.shape[1]
            meta["H"] = k.shape[2]
            meta["W"] = k.shape[3]
        elif k.ndim == 3:
            meta["layout"] = "3D: (slices, H, W)"
            meta["n_slices_per_vol"] = k.shape[0]
            meta["H"] = k.shape[1]
            meta["W"] = k.shape[2]

    for key in f.keys():
        if key != "kspace":
            ds = f[key]
            if isinstance(ds, h5py.Dataset):
                meta[f"extra_{key}_shape"] = list(ds.shape)
                meta[f"extra_{key}_dtype"] = str(ds.dtype)

    return meta


def stream_stats(f, n_slices_sample=50):
    k = f["kspace"]
    results = []

    if k.ndim == 5:
        n_vol, n_sl = k.shape[0], k.shape[1]
        total = n_vol * n_sl
        step  = max(1, total // n_slices_sample)
        indices = [(v, s) for v in range(n_vol) for s in range(n_sl)][::step][:n_slices_sample]

        for (v, s) in tqdm(indices, desc="Sampling slices"):
            sl = k[v, s]
            _process_slice(sl, results, label=f"vol{v}_sl{s}")

    elif k.ndim == 4:
        n_sl  = k.shape[0]
        step  = max(1, n_sl // n_slices_sample)
        idxs  = list(range(0, n_sl, step))[:n_slices_sample]

        for s in tqdm(idxs, desc="Sampling slices"):
            sl = k[s]
            _process_slice(sl, results, label=f"sl{s}")

    elif k.ndim == 3:
        n_sl  = k.shape[0]
        step  = max(1, n_sl // n_slices_sample)
        idxs  = list(range(0, n_sl, step))[:n_slices_sample]

        for s in tqdm(idxs, desc="Sampling slices"):
            sl = k[s]
            sl = sl[np.newaxis]
            _process_slice(sl, results, label=f"sl{s}")

    return results


def _process_slice(sl, results, label):
    mag = np.abs(sl)
    rss = np.sqrt((mag ** 2).sum(axis=0))

    col_energy = mag[0].sum(axis=0)
    mask = col_energy > col_energy.max() * 0.01
    n_sampled = int(mask.sum())
    W = sl.shape[-1]
    acc = round(W / n_sampled, 2) if n_sampled > 0 else None

    H, W2 = rss.shape
    signal = rss[H//4: 3*H//4, W2//4: 3*W2//4].mean()
    noise  = rss[:H//10, :W2//10].std()
    snr    = float(signal / noise) if noise > 0 else None

    results.append({
        "label":        label,
        "rss_mean":     float(rss.mean()),
        "rss_std":      float(rss.std()),
        "rss_max":      float(rss.max()),
        "ksp_mean":     float(mag.mean()),
        "ksp_std":      float(mag.std()),
        "n_sampled_cols": n_sampled,
        "acceleration": acc,
        "snr":          snr,
    })


def visualise(f, out_dir="plots", n=6):
    os.makedirs(out_dir, exist_ok=True)
    k = f["kspace"]

    if k.ndim == 5:
        total = k.shape[0] * k.shape[1]
        step  = max(1, total // n)
        slices = []
        for v in range(k.shape[0]):
            for s in range(k.shape[1]):
                slices.append((v, s))
        slices = slices[::step][:n]
        get_sl = lambda idx: k[idx[0], idx[1]]
        labels = [f"vol{v} sl{s}" for v, s in slices]
    elif k.ndim == 4:
        idxs = np.linspace(0, k.shape[0]-1, n, dtype=int)
        get_sl = lambda idx: k[idx]
        labels = [f"slice {i}" for i in idxs]
        slices = idxs
    else:
        idxs = np.linspace(0, k.shape[0]-1, n, dtype=int)
        get_sl = lambda idx: k[idx][np.newaxis]
        labels = [f"slice {i}" for i in idxs]
        slices = idxs

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1: axes = axes[np.newaxis]
    fig.suptitle("Sample Slices: RSS Image | log|k-space| | Sampling Mask", fontsize=12)

    for i, idx in enumerate(slices):
        sl  = get_sl(idx)
        mag = np.abs(sl)
        rss = np.sqrt((mag**2).sum(axis=0))

        axes[i, 0].imshow(rss, cmap="gray")
        axes[i, 0].set_title(f"RSS — {labels[i]}", fontsize=8)
        axes[i, 0].axis("off")

        lk = np.log1p(mag[0])
        axes[i, 1].imshow(lk, cmap="inferno")
        axes[i, 1].set_title("log|k-space| coil 0", fontsize=8)
        axes[i, 1].axis("off")

        col_e = mag[0].sum(axis=0)
        mask  = (col_e > col_e.max() * 0.01).astype(float)
        mask_img = np.tile(mask, (rss.shape[0], 1))
        axes[i, 2].imshow(mask_img, cmap="gray", aspect="auto")
        axes[i, 2].set_title("k-space column mask", fontsize=8)
        axes[i, 2].axis("off")

    plt.tight_layout()
    out = os.path.join(out_dir, "sample_slices.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Sample slices → {out}")


def make_plots(stats, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)
    import pandas as pd
    df = pd.DataFrame(stats)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("brain_multicoil_test_batch_2 — Dataset Analysis", fontsize=13, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(df["rss_mean"], color="#4C72B0", linewidth=0.8)
    ax.fill_between(range(len(df)),
                    df["rss_mean"] - df["rss_std"],
                    df["rss_mean"] + df["rss_std"], alpha=0.2, color="#4C72B0")
    ax.set_title("RSS Image Intensity (mean ± std)"); ax.set_xlabel("Sample index")

    ax = axes[0, 1]
    snrs = df["snr"].dropna()
    snrs.hist(bins=20, ax=ax, color="#55A868", edgecolor="white")
    ax.set_title(f"SNR Distribution  (mean={snrs.mean():.1f})")
    ax.set_xlabel("SNR")

    ax = axes[0, 2]
    accs = df["acceleration"].dropna()
    accs.hist(bins=20, ax=ax, color="#C44E52", edgecolor="white")
    ax.set_title(f"Acceleration Factor  (mean={accs.mean():.2f}x)")
    ax.set_xlabel("Acceleration")

    ax = axes[1, 0]
    ax.plot(df["ksp_mean"], color="#8172B2", linewidth=0.8)
    ax.set_title("k-space Magnitude Mean per Sample")
    ax.set_xlabel("Sample index")

    ax = axes[1, 1]
    df["n_sampled_cols"].hist(bins=20, ax=ax, color="#CCB974", edgecolor="white")
    ax.set_title("Sampled Phase-encode Lines")
    ax.set_xlabel("# Columns sampled")

    ax = axes[1, 2]
    ax.plot(df["rss_max"], color="#64B5CD", linewidth=0.8)
    ax.set_title("RSS Max Intensity per Sample")
    ax.set_xlabel("Sample index")

    plt.tight_layout()
    out = os.path.join(out_dir, "dataset_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Analysis plot  → {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",         required=True,          help="Path to HDF5 file")
    parser.add_argument("--n_sample",     type=int, default=50,   help="# slices to sample for stats")
    parser.add_argument("--n_vis",        type=int, default=6,    help="# slices to visualise")
    parser.add_argument("--plots_dir",    default="plots",        help="Output folder for plots")
    parser.add_argument("--save_report",  default="report.json",  help="JSON report output path")
    args = parser.parse_args()

    fpath = args.file
    print(f"\n{'='*60}")
    print(f"  FastMRI Single-File Analyser")
    print(f"  File : {fpath}")
    print(f"  Size : {os.path.getsize(fpath)/1e9:.2f} GB")
    print(f"{'='*60}\n")

    with h5py.File(fpath, "r") as f:
        print("── HDF5 Structure ──────────────────────────────")
        print_structure(f)
        print()

        print("── Metadata ────────────────────────────────────")
        meta = collect_metadata(f)
        for k, v in meta.items():
            print(f"  {k:<35} {v}")
        print()

        print("── Streaming slice statistics ──────────────────")
        stats = stream_stats(f, n_slices_sample=args.n_sample)
        print()

        print("── Visualising sample slices ───────────────────")
        visualise(f, out_dir=args.plots_dir, n=args.n_vis)

    snrs  = [s["snr"] for s in stats if s["snr"]]
    accs  = [s["acceleration"] for s in stats if s["acceleration"]]

    agg = {
        "snr_mean":         round(np.mean(snrs), 2),
        "snr_std":          round(np.std(snrs), 2),
        "snr_min":          round(np.min(snrs), 2),
        "snr_max":          round(np.max(snrs), 2),
        "acceleration_mean": round(np.mean(accs), 3),
        "acceleration_std":  round(np.std(accs), 3),
        "acceleration_min":  round(np.min(accs), 3),
        "acceleration_max":  round(np.max(accs), 3),
        "rss_mean_overall":  round(np.mean([s["rss_mean"] for s in stats]), 4),
        "rss_std_overall":   round(np.mean([s["rss_std"]  for s in stats]), 4),
    }

    print("\n── Aggregate Results ───────────────────────────────")
    for k, v in {**meta, **agg}.items():
        print(f"  {k:<35} {v}")

    print("\n── Generating plots ────────────────────────────────")
    make_plots(stats, out_dir=args.plots_dir)

    report = {"metadata": meta, "aggregate": agg, "per_sample": stats}
    with open(args.save_report, "w") as fp:
        json.dump(report, fp, indent=2, default=str)
    print(f"  Report         → {args.save_report}\n")


if __name__ == "__main__":
    main()