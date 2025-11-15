from modules.vector_db import get_vectorstore
from modules.search import retrieve_with_confidence, format_docs


def build_prompt(query: str, context: str):
    return f"""
You are an exam-preparation assistant. Use ONLY the given context.

Question:
{query}

Context:
{context}

Rules:
- Do NOT hallucinate.
- Cite facts like [Source: <filename>, Page: <page>].
- Give clear, structured explanations for students.
- End with a 'Summary' section.
"""


def ask_rag(query: str):
    vectorstore = get_vectorstore()

    docs, confidence = retrieve_with_confidence(vectorstore, query, k=5)

    if not docs:
        return {
            "answer": "No relevant information found in textbooks.",
            "citations": [],
            "confidence": 0.0,
            "chunks_used": [],
        }

    context = format_docs(docs)
    prompt = build_prompt(query, context)

    # Import the LLM call lazily so missing API keys or LLM init errors
    # don't break module import (which would cause ImportError at app startup).
    try:
        from modules.llm import call_gemini
    except Exception as e:
        return {
            "answer": f"LLM unavailable: {e}",
            "citations": [],
            "confidence": 0.0,
            "chunks_used": [d.page_content for d in docs],
        }

    try:
        answer = call_gemini(prompt)
    except Exception as e:
        return {
            "answer": f"Error calling LLM: {e}",
            "citations": [],
            "confidence": confidence,
            "chunks_used": [d.page_content for d in docs],
        }

    citations = []
    for d in docs:
        citations.append({
            "source": d.metadata.get("source", "unknown"),
            "page": d.metadata.get("page", "?")
        })

    return {
        "answer": answer,
        "citations": citations,
        "confidence": confidence,
        "chunks_used": [d.page_content for d in docs]
    }
