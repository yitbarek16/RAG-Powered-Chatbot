import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load the dataset
DATA_PATH = "/content/drive/My Drive/Data/filtered_complaints.csv"
df = pd.read_csv(DATA_PATH)

# Set up output directory
VECTOR_DIR = "vector_store"
os.makedirs(VECTOR_DIR, exist_ok=True)

# === SAMPLE a subset of the data for testing ===
df = df.sample(frac=0.05, random_state=42)  #  5% of data

# === Filter and preprocess ===
df = df[df["cleaned_narrative"].notnull()]
df["cleaned_narrative"] = df["cleaned_narrative"].astype(str)

# === Text Chunking ===
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)

chunks = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Splitting Text"):
    try:
        for chunk in text_splitter.split_text(row["cleaned_narrative"]):
            chunks.append({
                "chunk": chunk,
                "complaint_id": row["Complaint ID"],
                "product": row["Mapped_Product"]
            })
    except:
        continue

chunks_df = pd.DataFrame(chunks)
chunks_df.dropna(subset=["chunk"], inplace=True)
chunks_df.reset_index(drop=True, inplace=True)


# === Load Embedding Model ===
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embedding_dim = 384
index = faiss.IndexFlatL2(embedding_dim)

# === Embed and Index in Batches ===
batch_size = 2000
metadatas = []

for start in tqdm(range(0, len(chunks_df), batch_size), desc="Embedding & Indexing"):
    end = min(start + batch_size, len(chunks_df))
    batch = chunks_df.iloc[start:end]
    texts = batch["chunk"].tolist()

    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    index.add(embeddings)
    metadatas.extend(batch[["chunk", "complaint_id", "product"]].to_dict(orient="records"))

# === Save Outputs ===
faiss.write_index(index, os.path.join(VECTOR_DIR, "complaint_index.faiss"))

metadata_df = pd.DataFrame(metadatas)
metadata_df.to_csv(os.path.join(VECTOR_DIR, "metadata.csv"), index=False)


