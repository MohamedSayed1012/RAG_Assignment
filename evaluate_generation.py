import json
import time
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from rag_pipeline import RAGPipeline

TEST_SET = "test_set.json"

def load_test_set(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate():
    test_items = load_test_set(TEST_SET)
    rag = RAGPipeline()
    scorer = rouge_scorer.RougeScorer(['rouge1','rougeL'], use_stemmer=True)

    for item in test_items:
        q = item["question"]
        ref = item["reference_answer"]
        gen = rag.answer(q)

        # ROUGE
        scores = scorer.score(ref, gen)
        r1 = scores['rouge1'].fmeasure
        rL = scores['rougeL'].fmeasure

        # BERTScore
        P, R, F1 = bert_score([gen], [ref], lang="en", verbose=False)
        b_f1 = F1[0].item()

        print(f"Q: {q}")
        print(f"  ROUGE-1={r1:.2f}, ROUGE-L={rL:.2f}, BERTScore F1={b_f1:.2f}\n")

if __name__ == "__main__":
    start = time.time()
    evaluate()
    print(f"Elapsed: {time.time() - start:.2f}s")
