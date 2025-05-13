import faiss, pickle
from sentence_transformers import SentenceTransformer

INDEX_PATH = "faiss_index.bin"
META_PATH  = "metadatas.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

def semantic_search(query, top_k=5):
    # Load index & metadata
    index = faiss.read_index(INDEX_PATH)
    metas = pickle.load(open(META_PATH, "rb"))

    # Embed query
    model = SentenceTransformer(MODEL_NAME)
    q_emb = model.encode([query])

    # Search
    D, I = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        meta = metas[idx]
        results.append({
            "score": float(score),
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            # you could also show the text snippet here
        })
    return results

if __name__ == "__main__":
    q = "What is self-attention in transformers?"
    hits = semantic_search(q)
    for hit in hits:
        print(f"{hit['source']} (chunk {hit['chunk_index']}): score={hit['score']:.4f}")
