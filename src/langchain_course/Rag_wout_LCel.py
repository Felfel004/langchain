import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("Initializing components...")


# ---------------------------------------------------------
# 1. EMBEDDING MODEL
# ---------------------------------------------------------
# Converts text into numerical vectors.
# IMPORTANT: use the SAME model used when storing documents in Pinecone.
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# ---------------------------------------------------------
# 2. LLM
# ---------------------------------------------------------
# Generates the final natural-language answer.
llm = ChatOllama(
    model="qwen3:4b"
)


# ---------------------------------------------------------
# 3. CONNECT TO PINECONE
# ---------------------------------------------------------
# Connects LangChain to the existing Pinecone index.
#
# index_name:
#   tells LangChain WHICH Pinecone index to use.
#
# embedding:
#   tells LangChain HOW to convert the user's query
#   into a vector before searching Pinecone.
vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"],
    embedding=embeddings
)


# Turn the vector store into a retriever.
# k=3 means return the 3 most relevant chunks.
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# ---------------------------------------------------------
# 4. PROMPT TEMPLATE
# ---------------------------------------------------------
# {context}  -> retrieved document chunks
# {question} -> original user question
prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
)


def format_docs(docs):
    """Combine retrieved Documents into one context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def retrieval_chain_without_lcel(query: str):

    # -----------------------------------------------------
    # STEP 1: RETRIEVE DOCUMENTS
    # -----------------------------------------------------
    # invoke() means:
    # "Run this LangChain component with this input."
    #
    # Here:
    # query -> embedding -> Pinecone search -> top 3 Documents
    docs = retriever.invoke(query)

    # Convert the 3 Document objects into one text string.
    context = format_docs(docs)

    # -----------------------------------------------------
    # STEP 2: BUILD THE PROMPT
    # -----------------------------------------------------
    # Replaces:
    # {context}  with retrieved text
    # {question} with the user's query
    #
    # It DOES NOT call the LLM.
    # It only prepares the messages.
    messages = prompt_template.format_messages(
        context=context,
        question=query
    )

    # -----------------------------------------------------
    # STEP 3: CALL THE LLM
    # -----------------------------------------------------
    # invoke() here means:
    # "Run the LLM using these messages."
    response = llm.invoke(messages)

    return response.content


if __name__ == "__main__":

    query = "what is Pinecone in machine learning?"

    # -----------------------------------------------------
    # OPTION 0: NO RAG
    # -----------------------------------------------------
    # Question -> LLM -> Answer
    # No Pinecone and no retrieved documents.
    print("\n===== RAW LLM: NO RAG =====")

    result_raw = llm.invoke([
        HumanMessage(content=query)
    ])

    print(result_raw.content)


    # -----------------------------------------------------
    # OPTION 1: RAG
    # -----------------------------------------------------
    # Question
    #   -> embedding
    #   -> Pinecone
    #   -> retrieve 3 chunks
    #   -> build prompt
    #   -> LLM
    #   -> answer
    print("\n===== RAG: WITHOUT LCEL =====")

    result_rag = retrieval_chain_without_lcel(query)

    print(result_rag)

"""
User query
   ↓
retriever.invoke(query)
   ↓
embedding model converts query → vector
   ↓
Pinecone searches stored vectors
   ↓
returns top 3 relevant Documents
   ↓
format_docs(docs)
   ↓
combines document text into one context string
   ↓
prompt_template.format_messages(...)
   ↓
fills {context} + {question}
   ↓
llm.invoke(messages)
   ↓
LLM generates final answer
   ↓
response.content
"""