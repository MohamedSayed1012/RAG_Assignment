import json
import time
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_score, recall_score, f1_score

# Paths
INDEX_PATH = "faiss_index.bin"
META_PATH = "metadatas.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"
GT_PATH   = "ground_truth.json"

def load_ground_truth(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate():
    gt = load_ground_truth(GT_PATH)
    index = faiss.read_index(INDEX_PATH)
    metas = pickle.load(open(META_PATH, "rb"))
    embedder = SentenceTransformer(EMBED_MODEL)

    all_precisions, all_recalls, all_f1s = [], [], []

    for item in gt:
        question = item["question"]
        relevant = set((r["source"], r["chunk_index"]) for r in item["relevant_chunks"])

        # Embed and search
        q_emb = embedder.encode([question])
        D, I = index.search(q_emb, len(metas))

        # Form binary labels for top-k = len(metas)
        retrieved = [(metas[i]["source"], metas[i]["chunk_index"]) for i in I[0]]
        y_true = [1 if doc in relevant else 0 for doc in retrieved]
        y_pred = [1] * len(retrieved)

        # Compute metrics
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        all_precisions.append(p)
        all_recalls.append(r)
        all_f1s.append(f1)

        print(f"Q: {question}")
        print(f"  Precision={p:.2f}, Recall={r:.2f}, F1={f1:.2f}\n")

    # Overall averages
    print("Overall:")
    print(f"  Precision={sum(all_precisions)/len(all_precisions):.2f}")
    print(f"  Recall   ={sum(all_recalls)/len(all_recalls):.2f}")
    print(f"  F1       ={sum(all_f1s)/len(all_f1s):.2f}")

if __name__ == "__main__":
    start = time.time()
    evaluate()
    print(f"Elapsed: {time.time() - start:.2f}s")
