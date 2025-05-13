# run_experiments.py

import os
import csv
import time
from itertools import product

# External libraries
import pickle
import faiss
from sklearn.metrics import precision_score, recall_score, f1_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer

# Local modules
import embed                          # to override embed.MODEL_NAME
from embed import embed_and_index
from evaluate_retrieval import load_ground_truth
from evaluate_generation import load_test_set
from rag_pipeline import RAGPipeline

# Constants
DOCUMENT_FOLDER = "documents"
GT_PATH         = "ground_truth.json"
TEST_PATH       = "test_set.json"
DEFAULT_INDEX   = "faiss_index.bin"
DEFAULT_META    = "metadatas.pkl"

# Experiment grid
CONFIG = {
    "embed_models":       ["all-MiniLM-L6-v2", "paraphrase-MPNet-base-v2"],
    "templates":          ["GENERIC", "DEFINE"],
    "retrieval_strategies":["semantic"],  # add "mmr" when available
    "top_ks":             [3, 5, 10],
}

def eval_retrieval(pipeline, top_k):
    """Compute avg precision, recall, F1 over the ground-truth set."""
    gt = load_ground_truth(GT_PATH)
    retriever = pipeline.retriever
    index     = retriever.index
    metas     = retriever.metas
    embedder  = retriever.embedder

    precisions, recalls, f1s = [], [], []

    for item in gt:
        q = item["question"]
        relevant = {(r["source"], r["chunk_index"]) for r in item["relevant_chunks"]}

        q_emb = embedder.encode([q])
        D, I = index.search(q_emb, top_k)

        retrieved = [(metas[i]["source"], metas[i]["chunk_index"]) for i in I[0]]
        y_true    = [1 if doc in relevant else 0 for doc in retrieved]
        y_pred    = [1] * len(retrieved)

        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))

    return {
        "precision": sum(precisions) / len(precisions),
        "recall":    sum(recalls)    / len(recalls),
        "f1":        sum(f1s)        / len(f1s),
    }

def eval_generation(pipeline):
    """Compute avg ROUGE-1 and BERTScore F1 over the test set."""
    test_items = load_test_set(TEST_PATH)
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    rouge1_scores, bert_f1_scores = [], []

    for item in test_items:
        q   = item["question"]
        ref = item["reference_answer"]
        gen = pipeline.answer(q)

        scores = rouge.score(ref, gen)
        rouge1_scores.append(scores['rouge1'].fmeasure)

        _, _, F1 = bert_score([gen], [ref], lang="en", verbose=False)
        bert_f1_scores.append(F1[0].item())

    return {
        "rouge1":      sum(rouge1_scores)   / len(rouge1_scores),
        "bertscore_f1": sum(bert_f1_scores) / len(bert_f1_scores),
    }

def run():
    if "GROQ_API_KEY" not in os.environ:
        raise EnvironmentError("Please set GROQ_API_KEY before running experiments.")

    results = []

    for em, tpl, strat, k in product(
        CONFIG["embed_models"],
        CONFIG["templates"],
        CONFIG["retrieval_strategies"],
        CONFIG["top_ks"]
    ):
        # 1. Rebuild default index for this embedding model
        print(f"\n[Model: {em}] Building index → {DEFAULT_INDEX}, {DEFAULT_META}")
        embed.MODEL_NAME = em
        embed_and_index(
            folder=DOCUMENT_FOLDER,
            index_path=DEFAULT_INDEX,
            meta_path=DEFAULT_META
        )

        # 2. Initialize pipeline (uses default index & meta)
        pipeline = RAGPipeline(
            embed_model=em,
            template_name=tpl,
            strategy=strat
        )

        # 3. Evaluate
        print(f"[{em} | {tpl} | {strat} | top_k={k}] Running evaluations...")
        start = time.time()
        ret_metrics = eval_retrieval(pipeline, top_k=k)
        gen_metrics = eval_generation(pipeline)
        elapsed     = time.time() - start

        result = {
            "embed_model": em,
            "template":    tpl,
            "strategy":    strat,
            "top_k":       k,
            **ret_metrics,
            **gen_metrics,
            "time_s":      round(elapsed, 2),
        }
        results.append(result)
        print("→", result)

    # 4. Save to CSV
    with open("experiment_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\n✅ experiment_results.csv created.")

if __name__ == "__main__":
    run()
