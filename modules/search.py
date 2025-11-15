import numpy as np

def retrieve_with_confidence(vectorstore, query: str, k: int = 5):
    """Retrieve relevant docs and compute confidence."""
    
    docs_and_scores = vectorstore.similarity_search_with_relevance_scores(query, k=k)

    if not docs_and_scores:
        return [], 0.0

    docs = [d for d, s in docs_and_scores]
    scores = np.array([s for d, s in docs_and_scores])

    # Confidence = average of top 3 scores
    top_scores = np.sort(scores)[-3:] if len(scores) >= 3 else scores
    confidence = float(top_scores.mean())

    return docs, round(confidence, 3)


def format_docs(docs):
    blocks = []
    for d in docs:
        source = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        text = d.page_content
        blocks.append(f"[Source: {source}, Page: {page}]\n{text}")
    return "\n\n".join(blocks)
