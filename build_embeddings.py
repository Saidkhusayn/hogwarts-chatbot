import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# CONFIG
CSV_PATH = "data.csv"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Load local embedding model
model = SentenceTransformer(EMBED_MODEL)

# Load CSV and prepare chunks
df = pd.read_csv(CSV_PATH)
chunks = []
for _, row in df.iterrows():
    sec, sub = row["Section"], row["Sub-section"]
    text = f"Section: {sec}\nSub-section: {sub}\n{row['Content']}"
    add = row.get("Additional", "")
    if isinstance(add, str) and add.strip():
        text += "\nAdditional: " + add.strip()
    chunks.append(text)

# Generate embeddings
embs = model.encode(chunks, convert_to_numpy=True)

# Save to disk
np.save("embs.npy", embs)
with open("chunks.json", "w") as f:
    json.dump(chunks, f)

print("\u2705 Local embeddings and chunks saved.")