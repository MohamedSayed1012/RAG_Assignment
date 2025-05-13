# RAGLab Project

This README provides instructions and documentation for the Retrieval-Augmented Generation (RAG) pipeline built with LangChain, Sentence-Transformers, FAISS, and Groq.

## 1. Setup Instructions

1. **Clone the Repository**

   ```powershell
   git clone <your-repo-url>
   cd rag_lab
   ```
2. **Create and Activate Virtual Environment**

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
3. **Install Dependencies**

   ```powershell
   pip install --upgrade pip
   pip install \
     langchain \
     sentence-transformers \
     faiss-cpu \
     pdfplumber \
     python-docx \
     groq \
     transformers \
     rouge-score \
     bert-score
   ```
4. **Set Environment Variable**

   ```powershell
   $env:GROQ_API_KEY = "<YOUR_GROQ_API_KEY>"
   ```
5. **Build Index**

   ```powershell
   python embed.py
   ```
6. **Run Pipeline (Interactive)**

   ```powershell
   python rag_pipeline.py
   ```
7. **Run Experiments**

   ```powershell
   python run_experiments.py
   ```

## 2. Architecture Description

The RAG system consists of the following components:

* **Loader (`loaders.py`)**: Reads PDF, DOCX, and TXT files from `documents/`, extracts text and metadata.
* **Splitter (`splitter.py`)**: Breaks documents into overlapping text chunks using LangChain's `RecursiveCharacterTextSplitter`.
* **Embedder & Indexer (`embed.py`)**: Uses Sentence-Transformers to compute embeddings and FAISS to build a vector index (`faiss_index.bin` & `metadatas.pkl`).
* **Retriever (`rag_pipeline.py`)**: `FaissRetriever` searches the index for top-k similar chunks.
* **Prompt Templates (`prompts.py`)**: Defines `GENERIC_RAG_TEMPLATE` and `DEFINE_RAG_TEMPLATE` for context injection.
* **Groq Integration (`rag_pipeline.py`)**: Wraps retrieval and calls Groq's chat completions API to generate answers.
* **Evaluation**:

  * **Retrieval Metrics**: `evaluate_retrieval.py` computes precision, recall, and F1 against `ground_truth.json`.
  * **Generation Metrics**: `evaluate_generation.py` measures ROUGE-1 and BERTScore against `test_set.json`.
  * **Experiment Driver**: `run_experiments.py` sweeps embedding models, templates, strategies, and top-k settings, logging performance into `experiment_results.csv`.

## 3. Experimental Results

| Embed Model              | Template | Strategy | Top-k | Precision | Recall | F1   | ROUGE-1 | BERTScore F1 | Time (s) |
| ------------------------ | -------- | -------- | ----- | --------- | ------ | ---- | ------- | ------------ | -------- |
| all-MiniLM-L6-v2         | GENERIC  | semantic | 3     | 0.75      | 0.60   | 0.67 | 0.48    | 0.52         | 12.3     |
| all-MiniLM-L6-v2         | GENERIC  | semantic | 5     | 0.70      | 0.65   | 0.67 | 0.50    | 0.54         | 13.1     |
| paraphrase-MPNet-base-v2 | GENERIC  | semantic | 3     | 0.80      | 0.55   | 0.65 | 0.52    | 0.56         | 18.4     |
| paraphrase-MPNet-base-v2 | GENERIC  | semantic | 5     | 0.78      | 0.60   | 0.68 | 0.54    | 0.58         | 19.2     |

*(Extend table with additional configurations as needed)*

## 4. Evaluation Metrics

* **Precision**: Fraction of retrieved chunks that are relevant.
* **Recall**: Fraction of relevant chunks that were retrieved.
* **F1-Score**: Harmonic mean of precision and recall.
* **ROUGE-1**: Overlap of unigrams between generated and reference answers.
* **BERTScore F1**: Semantic similarity score using contextual embeddings.

## 5. Analysis of Strengths & Weaknesses

* **Strengths**:

  * Modular design allows easy swapping of embedding models and retrieval strategies.
  * FAISS delivers low-latency semantic search.
  * Groq integration provides high-quality generation leveraging retrieved context.

* **Weaknesses**:

  * Mismatched dimensionality issues require rebuilding the index per model.
  * No advanced retrieval strategies (e.g., MMR) implemented yet.
  * Evaluation currently automated but lacks human judgment for generation quality.

## 6. Challenges & Solutions

* **Dimension Mismatch**: Faced FAISS assertion errors when using different embedder dimensions.
  *Solution*: Rebuild FAISS index for each embedding model within `run_experiments.py`.

* **Template KeyError**: `DEFINE_RAG_TEMPLATE` expected `term` rather than `question`.
  *Solution*: Dynamically detect template inputs and map `term=question` when needed.

* **Environment Variables**: Forgot to set `GROQ_API_KEY`.
  *Solution*: Added explicit environment-variable check in `RAGPipeline.__init__`.

---

*This file was drafted as a plain text README and can be further edited in any text editor.*
