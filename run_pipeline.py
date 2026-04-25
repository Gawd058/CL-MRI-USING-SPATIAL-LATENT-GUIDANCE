import subprocess, sys, os

DATA_DIR       = "E:/ALL E FOLDERS HERE/AI LAB/multicoil_test"
SAVE_BASE      = "./checkpoints"
RESULTS_DIR    = "./results"

CL_EPOCHS      = 40
CL_BATCH_SIZE  = 2
CL_LR          = 1e-3
CL_TEMPERATURE = 0.1
ACCEL_FACTORS  = "2 4 6"
LATENT_DIM     = 128
NUM_LOW_FREQ   = 16

RECON_EPOCHS   = 40
RECON_BATCH    = 2
RECON_LR       = 1e-4
MODELS         = ["unet"]
TEST_ACCELS    = [2,4,6]


def run(cmd):
    print(f"\n>>> {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=True)
    return result


def main():
    py = sys.executable
    cl_ckpt = os.path.join(SAVE_BASE, "cl", "best_cl.pth")

    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)
    run([
        py, "evaluate.py",
        "--data_dir",  DATA_DIR,
        "--cl_ckpt",   cl_ckpt,
        "--recon_dir", os.path.join(SAVE_BASE, "recon"),
        "--model",     "unet",
        "--out_dir",   RESULTS_DIR,
    ])

    print("\n✓ Pipeline complete. Results saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()