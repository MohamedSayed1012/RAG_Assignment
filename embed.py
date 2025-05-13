from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
from splitter import split_documents
from loaders import load_documents

MODEL_NAME = "all-MiniLM-L6-v2"  # you can experiment with different models

def embed_and_index(folder="documents", index_path="faiss_index.bin", meta_path="metadatas.pkl"):
    # 1. Load & split
    docs = load_documents(folder)
    chunks = split_documents(docs)

    # 2. Embed
    model = SentenceTransformer(MODEL_NAME)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    # 3. Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # 4. Save index & metadata
    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump([c["metadata"] for c in chunks], f)

    print(f"Indexed {len(chunks)} chunks (dim={dim}).")

if __name__ == "__main__":
    embed_and_index()
