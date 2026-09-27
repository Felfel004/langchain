import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()

print("Initializing components...")


# ============================================================
# 1. Initialize embeddings and LLM
# ============================================================

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()


# ============================================================
# 2. Connect to Pinecone
# ============================================================

vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"],
    embedding=embeddings,
)


# Retrieve the top 3 most relevant documents
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# ============================================================
# 3. Prompt template
# ============================================================

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
)


# ============================================================
# 4. Format retrieved documents
# ============================================================

def format_docs(docs):
    """
    Converts:

    [
        Document(page_content="Document 1"),
        Document(page_content="Document 2"),
        Document(page_content="Document 3")
    ]

    into:

    "Document 1

    Document 2

    Document 3"
    """

    return "\n\n".join(doc.page_content for doc in docs)


# ============================================================
# 5. LCEL RAG Chain
# ============================================================

def create_retrieval_chain():
    #RunnablePassthrough.assign() keeps the original input fields unchanged and adds a new field to
    #them. Here, it keeps the original question and creates a new field called context. The context value
    #is produced by taking the question, retrieving the relevant documents, and formatting those
    #documents into one string. The output therefore contains both question and context.
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=(
                itemgetter("question")
                | retriever
                | format_docs
            )
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    # ========================================================
    # LCEL EXPLANATION
    # ========================================================
    #
    # The chain expects an input like:
    #
    # {
    #     "question": "what is Pinecone?"
    # }
    #
    #
    # --------------------------------------------------------
    # 1. itemgetter("question")
    # --------------------------------------------------------
    #
    # Extracts the value of "question" from the input dictionary.
    #
    # Input:
    #
    # {
    #     "question": "what is Pinecone?"
    # }
    #
    # Output:
    #
    # "what is Pinecone?"
    #
    #
    # --------------------------------------------------------
    # 2. | retriever
    # --------------------------------------------------------
    #
    # The extracted question is passed to the retriever.
    #
    # question
    #    ↓
    # embedding model
    #    ↓
    # query vector
    #    ↓
    # Pinecone similarity search
    #    ↓
    # top 3 Documents
    #
    #
    # --------------------------------------------------------
    # 3. | format_docs
    # --------------------------------------------------------
    #
    # The retriever returns:
    #
    # [
    #     Document(...),
    #     Document(...),
    #     Document(...)
    # ]
    #
    # format_docs() combines their page_content
    # into one string.
    #
    # List[Document]
    #       ↓
    # format_docs()
    #       ↓
    # context string
    #
    #
    # --------------------------------------------------------
    # 4. RunnablePassthrough.assign(...)
    # --------------------------------------------------------
    #
    # Keeps the original input dictionary
    # and adds the new "context" value.
    #
    # Before:
    #
    # {
    #     "question": "what is Pinecone?"
    # }
    #
    # After:
    #
    # {
    #     "question": "what is Pinecone?",
    #     "context": "retrieved information..."
    # }
    #
    #
    # --------------------------------------------------------
    # 5. | prompt_template
    # --------------------------------------------------------
    #
    # Uses:
    #
    # {question}
    # {context}
    #
    # to create the final prompt.
    #
    #
    # --------------------------------------------------------
    # 6. | llm
    # --------------------------------------------------------
    #
    # Sends the formatted prompt to the LLM.
    #
    # Output:
    #
    # AIMessage(
    #     content="..."
    # )
    #
    #
    # --------------------------------------------------------
    # 7. | StrOutputParser()
    # --------------------------------------------------------
    #
    # Converts:
    #
    # AIMessage(content="...")
    #
    # into:
    #
    # "..."
    #
    # A normal Python string.
    #
    #
    # ========================================================
    # FULL FLOW
    # ========================================================
    #
    # {"question": query}
    #        ↓
    # itemgetter("question")
    #        ↓
    # question string
    #        ↓
    # retriever
    #        ↓
    # top 3 Documents
    #        ↓
    # format_docs()
    #        ↓
    # context string
    #        ↓
    # RunnablePassthrough.assign()
    #        ↓
    # {
    #   "question": query,
    #   "context": context
    # }
    #        ↓
    # prompt_template
    #        ↓
    # llm
    #        ↓
    # AIMessage
    #        ↓
    # StrOutputParser()
    #        ↓
    # final answer string
    #

    return retrieval_chain


# ============================================================
# 6. Run the LCEL chain
# ============================================================

if __name__ == "__main__":

    query = "what is Pinecone in machine learning?"

    chain = create_retrieval_chain()

    result = chain.invoke(
        {
            "question": query
        }
    )

    print("\nAnswer:")
    print(result)