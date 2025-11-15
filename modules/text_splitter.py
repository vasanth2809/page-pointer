from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(docs, chunk_size=800, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_documents(docs)
