from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
# Directory where core.py exists.
CURRENT_DIR = Path(__file__).resolve().parent

# Point to exactly the same database used by ingestion.py.
CHROMA_DIR = CURRENT_DIR / "chroma_db_nomic"


# ---------------------------------------------------------
# Environment setup
# ---------------------------------------------------------
load_dotenv()


# ---------------------------------------------------------
# Embedding model
# ---------------------------------------------------------
# IMPORTANT:
# Must be the same embedding model used during ingestion.
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
)


# ---------------------------------------------------------
# Vector store
# ---------------------------------------------------------
# Opens the Chroma database created by ingestion.py.
vectorstore = Chroma(
    persist_directory=str(CHROMA_DIR),
    embedding_function=embeddings,
)


# ---------------------------------------------------------
# Chat model
# ---------------------------------------------------------
# Qwen reads the retrieved context
# and generates the final answer.
model = ChatOllama(
    model="qwen3:4b",
)


# ---------------------------------------------------------
# Run RAG pipeline
# ---------------------------------------------------------
def run_llm(query: str) -> Dict[str, Any]:
    """
    Retrieve relevant documents first,
    then use them to generate the answer.
    """

    # -----------------------------------------------------
    # 1. Create retriever
    # -----------------------------------------------------
    # Return the 4 most relevant document chunks.
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    # -----------------------------------------------------
    # 2. Retrieve relevant chunks
    # -----------------------------------------------------
    # The query is embedded with nomic-embed-text,
    # then compared with the vectors stored in Chroma.
    retrieved_docs = retriever.invoke(query)

    print(
        f"📄 RETRIEVED {len(retrieved_docs)} DOCUMENTS"
    )

    # -----------------------------------------------------
    # 3. Convert Documents into text
    # -----------------------------------------------------
    # Qwen receives readable text instead of
    # raw LangChain Document objects.
    context = "\n\n".join(
        (
            f"Source: {doc.metadata.get('source', 'Unknown')}\n"
            f"Content: {doc.page_content}"
        )
        for doc in retrieved_docs
    )

    # -----------------------------------------------------
    # 4. Build prompt
    # -----------------------------------------------------
    prompt = f"""
You are a helpful AI assistant that answers questions
about LangChain documentation.

Answer ONLY using the retrieved context below.

Do not invent facts, APIs, examples, or links.
Cite source URLs when possible.

If the retrieved context does not contain enough information,
say that the documentation does not provide enough information.

Retrieved context:
{context}

User question:
{query}
"""

    # -----------------------------------------------------
    # 5. Generate answer
    # -----------------------------------------------------
    response = model.invoke(prompt)

    # -----------------------------------------------------
    # 6. Return answer + retrieved Documents
    # -----------------------------------------------------
    return {
        "answer": response.content,
        "context": retrieved_docs,
    }


# ---------------------------------------------------------
# Program entry point
# ---------------------------------------------------------
if __name__ == "__main__":

    result = run_llm(
        query="What are deep agents?"
    )

    print(result)


"""
===========================================================
RAG / RETRIEVAL FLOW
===========================================================

User Question
      ↓
run_llm(query)
      ↓
Retriever
      ↓
nomic-embed-text
      ↓
Query Vector
      ↓
Chroma
      ↓
Compare Query Vector
with stored Document Vectors
      ↓
Top 4 Relevant Chunks
      ↓
retrieved_docs
      ↓
Convert Documents → Context Text
      ↓
Context + User Question
      ↓
Qwen3:4b
      ↓
Final Answer


===========================================================
CONNECTION WITH ingestion.py
===========================================================

INGESTION:

Website
   ↓
Tavily
   ↓
Documents
   ↓
Chunks
   ↓
nomic-embed-text
   ↓
Document Vectors
   ↓
Chroma
   ↓
chroma_db_nomic


RETRIEVAL:

User Question
   ↓
nomic-embed-text
   ↓
Query Vector
   ↓
same chroma_db_nomic
   ↓
Most Similar Chunks
   ↓
Qwen3:4b
   ↓
Answer

===========================================================
"""