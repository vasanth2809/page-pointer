import streamlit as st
from modules.rag_pipeline import ask_rag

st.set_page_config(page_title="Exam RAG Assistant", layout="wide")

st.title("📚 Exam Preparation RAG Assistant")
st.write("Ask questions using your uploaded textbook PDFs.")

query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching textbooks..."):
            res = ask_rag(query)

        st.subheader("🧠 Answer")
        st.write(res["answer"])

        st.subheader("📊 Confidence Score")
        st.progress(res["confidence"])
        st.text(f"Confidence: {res['confidence']}")

        st.subheader("📎 Citations")
        for c in res["citations"]:
            st.write(f"- {c['source']} (page {c['page']})")

        with st.expander("📖 Retrieved Chunks"):
            for i, chunk in enumerate(res["chunks_used"], start=1):
                st.markdown(f"**Chunk {i}:**")
                st.write(chunk)
                st.markdown("---")
