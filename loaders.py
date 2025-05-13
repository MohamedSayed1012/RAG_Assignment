import os
import pdfplumber
from docx import Document

def load_documents(folder_path: str):
    """
    Walks `folder_path`, loads text from PDF, DOCX, and TXT files,
    returns a list of dicts: { "text": str, "metadata": {...} }.
    """
    docs = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            path = os.path.join(root, filename)
            text = ""
            try:
                if filename.lower().endswith(".pdf"):
                    with pdfplumber.open(path) as pdf:
                        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                elif filename.lower().endswith(".docx"):
                    doc = Document(path)
                    text = "\n".join(p.text for p in doc.paragraphs)
                elif filename.lower().endswith(".txt"):
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()
                else:
                    print(f"Skipping unsupported file: {filename}")
                    continue

                docs.append({
                    "text": text,
                    "metadata": {
                        "source": filename,
                        "path": path,
                    }
                })
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return docs

if __name__ == "__main__":
    loaded = load_documents("documents")
    print(f"Loaded {len(loaded)} documents.")
