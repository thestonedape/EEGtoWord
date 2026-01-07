# Install tqdm if you don't have it
# pip install tqdm

# Run extraction
# python extract_zuco_data.py

import pickle
import pandas as pd

PKL_PATH = r"C:\Users\n1sha\Desktop\zuco_complete_18subjects.pkl"
OUT_CSV  = r"C:\Users\n1sha\Desktop\metadata.csv"

with open(PKL_PATH, "rb") as f:
    all_data = pickle.load(f)

rows = []
for i, s in enumerate(all_data):
    rows.append({
        "sample_id": i,
        "subject": s["subject"],
        "word": s["word"],
        "sentence_id": s["sentence_id"],
        "word_id": s["word_id"]
    })

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

print("metadata.csv created:", df.shape)
