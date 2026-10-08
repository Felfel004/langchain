import asyncio
import os
import ssl
from pathlib import Path
from typing import List

import certifi
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_tavily import TavilyCrawl

from logger import (
    Colors,
    log_error,
    log_header,
    log_info,
    log_success,
    log_warning,
)


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
# Directory where ingestion.py exists.
CURRENT_DIR = Path(__file__).resolve().parent

# Always save Chroma inside the mini-rag folder.
CHROMA_DIR = CURRENT_DIR / "chroma_db_nomic"


# ---------------------------------------------------------
# Environment setup
# ---------------------------------------------------------
load_dotenv()

# Use certifi certificates for HTTPS requests.
ssl_context = ssl.create_default_context(cafile=certifi.where())

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


# ---------------------------------------------------------
# Embedding model
# ---------------------------------------------------------
# Converts text into vectors locally using Ollama.
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
)


# ---------------------------------------------------------
# Vector store
# ---------------------------------------------------------
# Stores document text, vectors, and metadata locally.
vectorstore = Chroma(
    persist_directory=str(CHROMA_DIR),
    embedding_function=embeddings,
)


# ---------------------------------------------------------
# Tavily crawler
# ---------------------------------------------------------
# Crawls website pages and extracts their content.
tavily_crawl = TavilyCrawl()


# ---------------------------------------------------------
# Index documents into Chroma
# ---------------------------------------------------------
async def index_documents_async(
    documents: List[Document],
    batch_size: int = 50,
):
    """
    Divide documents into batches,
    embed them, and store them in Chroma.
    """

    log_header("VECTOR STORAGE PHASE")

    log_info(
        f"📚 Preparing to index {len(documents)} documents",
        Colors.DARKCYAN,
    )

    # Example:
    # 1200 documents + batch_size=500
    #
    # Batch 1 → 500
    # Batch 2 → 500
    # Batch 3 → 200
    batches = [
        documents[i : i + batch_size]
        for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 Created {len(batches)} batches "
        f"of up to {batch_size} documents"
    )

    async def add_batch(
        batch: List[Document],
        batch_num: int,
    ):
        try:
            # Chroma extracts page_content,
            # creates embeddings using nomic-embed-text,
            # then stores text + vector + metadata.
            await vectorstore.aadd_documents(batch)

            log_success(
                f"Added batch {batch_num}/{len(batches)} "
                f"({len(batch)} documents)"
            )

            return True

        except Exception as e:
            log_error(
                f"Failed batch {batch_num}: {e}"
            )

            return False

    # Create one async task for each batch.
    tasks = [
        add_batch(batch, i + 1)
        for i, batch in enumerate(batches)
    ]

    # Run the batch tasks concurrently.
    results = await asyncio.gather(
        *tasks,
        return_exceptions=True,
    )

    successful = sum(
        result is True
        for result in results
    )

    if successful == len(batches):
        log_success(
            f"All batches indexed successfully "
            f"({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"Indexed {successful}/{len(batches)} "
            f"batches successfully"
        )


# ---------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------
async def main():

    log_header("DOCUMENTATION INGESTION PIPELINE")

    # -----------------------------------------------------
    # 1. Crawl website
    # -----------------------------------------------------
    log_info(
        "🗺️ Crawling LangChain documentation",
        Colors.PURPLE,
    )

    res = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com/",
            "max_depth": 2,
            "extract_depth": "advanced",
        }
    )

    # -----------------------------------------------------
    # 2. Convert Tavily results → Documents
    # -----------------------------------------------------
    all_docs = []

    for item in res["results"]:

        url = item.get("url", "Unknown source")
        raw_content = item.get("raw_content")

        # Skip pages where Tavily extracted no valid text.
        if not isinstance(raw_content, str) or not raw_content.strip():
            log_warning(
                f"Skipping {url}: no valid text extracted"
            )
            continue

        log_info(
            f"Successfully crawled {url}"
        )

        all_docs.append(
            Document(
                page_content=raw_content,
                metadata={
                    "source": url,
                },
            )
        )

    if not all_docs:
        log_error(
            "No valid documents were extracted."
        )
        return

    # -----------------------------------------------------
    # 3. Split Documents into chunks
    # -----------------------------------------------------
    log_header("DOCUMENT CHUNKING PHASE")

    # chunk_size=4000:
    # Try to keep each chunk around 4000 characters.
    #
    # chunk_overlap=200:
    # Neighboring chunks share some text to preserve context.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200,
    )

    splitted_docs = text_splitter.split_documents(
        all_docs
    )

    log_success(
        f"Created {len(splitted_docs)} chunks "
        f"from {len(all_docs)} documents"
    )

    # -----------------------------------------------------
    # 4. Create embeddings and store chunks
    # -----------------------------------------------------
    await index_documents_async(
        splitted_docs,
        batch_size=500,
    )

    # -----------------------------------------------------
    # Finished
    # -----------------------------------------------------
    log_header("PIPELINE COMPLETE")

    log_success(
        "🎉 Documentation ingestion completed!"
    )

    log_info(
        f"Web pages converted to Documents: {len(all_docs)}"
    )

    log_info(
        f"Document chunks created: {len(splitted_docs)}"
    )


# ---------------------------------------------------------
# Program entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(main())


"""
===========================================================
INGESTION FLOW
===========================================================

Website
   ↓
Tavily Crawl
   ↓
Webpage Text
   ↓
LangChain Documents
   ↓
RecursiveCharacterTextSplitter
   ↓
Document Chunks
   ↓
Batch Documents
   ↓
OllamaEmbeddings
   ↓
nomic-embed-text
   ↓
Document Vectors
   ↓
Chroma
   ↓
chroma_db_nomic

===========================================================
"""