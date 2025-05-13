import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from loaders import load_documents
from splitter import split_documents
from prompts import GENERIC_RAG_TEMPLATE, DEFINE_RAG_TEMPLATE

class FaissRetriever:
    def __init__(self, index_path, meta_path, embed_model):
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.metas = pickle.load(f)
        self.embedder = SentenceTransformer(embed_model)

    def get_relevant(self, query, top_k=5):
        q_emb = self.embedder.encode([query])
        distances, indices = self.index.search(q_emb, top_k)
        chunks = []
        for dist, idx in zip(distances[0], indices[0]):
            meta = self.metas[idx]
            chunks.append({
                "text": None,
                "source": meta["source"],
                "chunk_index": meta["chunk_index"],
                "score": float(dist)
            })
        return chunks

def enrich_chunks(chunks, folder="documents"):
    docs = load_documents(folder)
    all_chunks = split_documents(docs)
    for res in chunks:
        for c in all_chunks:
            if (c["metadata"]["source"] == res["source"] and
                c["metadata"]["chunk_index"] == res["chunk_index"]):
                res["text"] = c["text"]
                break
    return chunks

class RAGPipeline:
    def __init__(self,
                 embed_model="all-MiniLM-L6-v2",
                 template_name="GENERIC",
                 strategy="semantic"):
        # Retriever
        self.retriever = FaissRetriever(
            index_path="faiss_index.bin",
            meta_path="metadatas.pkl",
            embed_model=embed_model
        )
        # Template
        if template_name.upper() == "GENERIC":
            self.template = GENERIC_RAG_TEMPLATE
        elif template_name.upper() == "DEFINE":
            self.template = DEFINE_RAG_TEMPLATE
        else:
            raise ValueError(f"Unknown template: {template_name}")
        # Strategy (not yet used)
        if strategy not in ("semantic", "mmr"):
            raise ValueError(f"Unknown strategy: {strategy}")
        self.strategy = strategy
        # Groq client
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("Please set GROQ_API_KEY in your environment")
        self.client = Groq(api_key=api_key)
        self.model_name = "gemma2-9b-it"

    def answer(self, question: str, top_k: int = 5) -> str:
        # 1. Retrieve
        hits = self.retriever.get_relevant(question, top_k=top_k)
        hits = enrich_chunks(hits)
        if not hits:
            return "I’m sorry, I couldn’t find any relevant context to answer that."

        # 2. Build context
        context = "\n---\n".join(h["text"] for h in hits)

        # 3. Format prompt correctly based on template vars
        vars_needed = self.template.input_variables
        if "question" in vars_needed:
            prompt = self.template.format(context=context, question=question)
        elif "term" in vars_needed:
            prompt = self.template.format(context=context, term=question)
        else:
            prompt = self.template.format(context=context, question=question)

        # 4. Call Groq
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system",  "content": "You are a helpful assistant."},
                    {"role": "user",    "content": prompt}
                ],
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating answer: {e}"

if __name__ == "__main__":
    rag = RAGPipeline()
    print("🦾 RAG Groq Assistant (type 'exit' to quit)")
    while True:
        q = input("Enter question (or 'exit'): ")
        if q.strip().lower() == "exit":
            break
        print("\n" + rag.answer(q) + "\n")
