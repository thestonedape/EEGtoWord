# import pickle
# import h5py
# import numpy as np
# from tqdm import tqdm
# from pathlib import Path

# PKL_PATH = r"C:\Users\n1sha\Desktop\zuco_complete_18subjects.pkl"
# H5_PATH  = r"C:\Users\n1sha\Desktop\zuco_eeg_dataset.h5"

# print("Loading pickle (this may take time)...")
# with open(PKL_PATH, "rb") as f:
#     all_data = pickle.load(f)

# N = len(all_data)
# print(f"Loaded {N:,} samples")

# # ---- Pre-allocate arrays ----
# eeg_data = np.zeros((N, 105, 250), dtype=np.float32)
# sentence_id = np.zeros(N, dtype=np.int32)
# word_id = np.zeros(N, dtype=np.int32)

# words = []
# subjects = []

# # ---- Fill arrays ----
# for i, sample in tqdm(enumerate(all_data), total=N, desc="Packing data"):
#     eeg_data[i] = sample["eeg"]
#     sentence_id[i] = sample["sentence_id"]
#     word_id[i] = sample["word_id"]
#     words.append(sample["word"])
#     subjects.append(sample["subject"])

# # Convert strings to fixed-length UTF-8
# words = np.array(words, dtype=h5py.string_dtype(encoding="utf-8"))
# subjects = np.array(subjects, dtype=h5py.string_dtype(encoding="utf-8"))

# print("Writing HDF5 file...")

# with h5py.File(H5_PATH, "w") as h5:
#     h5.create_dataset(
#         "eeg",
#         data=eeg_data,
#         compression="gzip",
#         compression_opts=4,
#         chunks=True
#     )
#     h5.create_dataset("word", data=words)
#     h5.create_dataset("subject", data=subjects)
#     h5.create_dataset("sentence_id", data=sentence_id)
#     h5.create_dataset("word_id", data=word_id)

# size_gb = Path(H5_PATH).stat().st_size / 1e9
# print(f"✅ Saved HDF5 dataset: {H5_PATH}")
# print(f"📦 File size: {size_gb:.2f} GB")
