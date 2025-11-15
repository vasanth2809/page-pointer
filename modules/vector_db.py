import os
from langchain_community.vectorstores import Chroma
from modules.embeddings import get_embeddings_model
from modules.data_loader import load_pdfs
from modules.text_splitter import split_documents

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "vectorstore")


def get_vectorstore():
    """Builds or loads vectorstore."""
    embeddings = get_embeddings_model()

    # If vectorstore already exists → load it
    if os.path.exists(os.path.join(PERSIST_DIR, "chroma.sqlite")):
        return Chroma(
            embedding_function=embeddings,
            collection_name="textbook_rag",
            persist_directory=PERSIST_DIR
        )

    # Else → build vectorstore from PDFs
    docs = load_pdfs(os.path.join(BASE_DIR, "data"))
    chunks = split_documents(docs)

    os.makedirs(PERSIST_DIR, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="textbook_rag",
        persist_directory=PERSIST_DIR
    )

    return vectorstore
    