from ai_service import db
from ai_service.embedder import embed_text
from ai_service.ollama_client import chat_with_ollama
from .exceptions import AIServiceError


def upload_code(code: str):
    embedding = embed_text(code)
    db.add_chunks([code], [embedding])


def answer_question(user_question: str, top_k: int = 3):
    question_embedding = embed_text(user_question)
    results = db.query_chunks(question_embedding, number_of_results=top_k)
    unique_snippets = list(dict.fromkeys(results["documents"][0]))
    relevant_code = "\n---\n".join(unique_snippets)
    prompt = f"Code context:\n{relevant_code}\nQuestion: {user_question}\nExplain how the code works."
    answer = chat_with_ollama(prompt)
    print("User question:", user_question)
    print("\nMost relevant code snippet(s):\n", relevant_code)
    print("\nLLM answer:\n", answer)


def main() -> None:
    code = """
def add(a, b, c):
    return a + b - c
"""
    try:
        # Upload code once
        upload_code(code)

        # Clear all documents in the collection so only the current code is present
        # collection.delete(ids=collection.peek()["ids"])

        # User asks multiple questions
        questions = [
            "How does the sum work here?",
            "What happens if I pass strings to this function?",
        ]
        for q in questions:
            answer_question(q)
            print("\n" + "=" * 40 + "\n")
    except AIServiceError as e:
        print(f"AI Service error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
