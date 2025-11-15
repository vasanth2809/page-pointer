import os
from langchain_community.document_loaders import PyPDFLoader

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# for loading pdfs
def load_pdfs(data_dir: str = DATA_DIR):
    """Loads all PDFs from the data folder."""
    docs = []
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(data_dir, filename)
            loader = PyPDFLoader(filepath)
            loaded_docs = loader.load()
            docs.extend(loaded_docs)
    return docs

# for loading text files
def load_texts(data_dir: str = DATA_DIR):
    """Loads all text files from the data folder."""
    docs = []
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                docs.append({"content": content, "source": filepath})
    return docs

# for loading csv files
def load_csvs(data_dir: str = DATA_DIR):
    """Loads all CSV files from the data folder."""
    import pandas as pd
    docs = []
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(".csv"):
            filepath = os.path.join(data_dir, filename)
            df = pd.read_csv(filepath)
            for _, row in df.iterrows():
                docs.append({"content": row.to_dict(), "source": filepath})
    return docs

# for loading all documents
def load_all_documents(data_dir: str = DATA_DIR):
    """Loads all supported document types from the data folder."""
    docs = []
    docs.extend(load_pdfs(data_dir))
    docs.extend(load_texts(data_dir))
    docs.extend(load_csvs(data_dir))
    return docs