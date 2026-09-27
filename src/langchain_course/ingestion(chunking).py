import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

CURRENT_DIR = Path(__file__).parent

if __name__ == "__main__":
    print("Ingesting...")

    file_path = CURRENT_DIR / "mediumblog1.txt"

    print(f"Loading file from: {file_path}")
    print(f"File exists: {file_path.exists()}")

    loader = UnstructuredLoader(
        file_path=str(file_path),
        chunking_strategy="basic",
        max_characters=1000000,
    )

    document = loader.load()

    print("Splitting...")

    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )

    texts = text_splitter.split_documents(document)

    print(f"Created {len(texts)} chunks")

    # Local embeddings through Ollama
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    print("Ingesting...")

    PineconeVectorStore.from_documents(
        texts,
        embeddings,
        index_name=os.environ["INDEX_NAME"]
    )

    print("Finished")


"""
mediumblog1.txt
       ↓
UnstructuredLoader
       ↓
Document
       ↓
CharacterTextSplitter
       ↓
1000-character chunks
       ↓
OllamaEmbeddings
nomic-embed-text
       ↓
768-dimensional vectors
       ↓
Pinecone
"""