from langchain_core.prompts import PromptTemplate

GENERIC_RAG_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an expert assistant. Use the following context to answer the question.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer concisely and completely."
    )
)

DEFINE_RAG_TEMPLATE = PromptTemplate(
    input_variables=["context", "term"],
    template=(
        "Using the context below, provide a clear definition of “{term}”:\n\n"
        "Context:\n{context}\n\n"
        "Definition:"
    )
)
