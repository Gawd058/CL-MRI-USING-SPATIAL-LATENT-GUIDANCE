import h5py

with h5py.File("brain_multicoil_test_batch_2", "r") as f:
    print("kspace shape:", f["kspace"].shape)
    print("mask shape:", f["mask"].shape)