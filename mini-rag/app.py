import streamlit as st

from core import run_llm


# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="LangChain RAG Assistant",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 LangChain RAG Assistant")

st.write(
    "Ask questions about the LangChain documentation "
    "stored in the local Chroma vector database."
)


# ---------------------------------------------------------
# User input
# ---------------------------------------------------------
query = st.text_input(
    "Ask a question",
    placeholder="Example: What are deep agents?",
)


# ---------------------------------------------------------
# Run RAG pipeline
# ---------------------------------------------------------
if st.button("Ask"):

    if not query.strip():
        st.warning("Please enter a question.")

    else:

        # Show a loading spinner while the RAG pipeline runs.
        with st.spinner("Searching documentation..."):

            try:
                result = run_llm(query)

                answer = result["answer"]
                context_docs = result["context"]

            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()


        # -------------------------------------------------
        # Display final answer
        # -------------------------------------------------
        st.subheader("Answer")

        st.markdown(answer)


        # -------------------------------------------------
        # Display retrieved context
        # -------------------------------------------------
        st.subheader("Retrieved Sources")

        if not context_docs:
            st.info("No context documents were returned.")

        else:

            for i, doc in enumerate(context_docs, start=1):

                source = doc.metadata.get(
                    "source",
                    "Unknown source",
                )

                # Each retrieved chunk can be expanded
                # so we can inspect what the RAG system used.
                with st.expander(
                    f"Source {i}: {source}"
                ):

                    st.write(
                        doc.page_content
                    )