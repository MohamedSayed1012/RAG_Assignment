from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    """
    docs: list of {"text": str, "metadata": {...}}
    returns: list of {"text": str, "metadata": {...}}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = []
    for doc in docs:
        pieces = splitter.split_text(doc["text"])
        for i, piece in enumerate(pieces):
            chunks.append({
                "text": piece,
                "metadata": {
                    **doc["metadata"],
                    "chunk_index": i,
                }
            })
    return chunks

if __name__ == "__main__":
    from loaders import load_documents
    docs = load_documents("documents")
    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")
