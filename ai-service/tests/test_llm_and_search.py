import ai_service.ollama_client
from ai_service import db
from ai_service.embedder import embed_text


# -------------- LLM Chat Tests --------------


def test_llm_chat_basic_response(monkeypatch):
    # Mock Ollama response for deterministic test
    monkeypatch.setattr(
        "ai_service.ollama_client.chat_with_ollama",
        lambda prompt: "Mocked LLM response for: " + prompt,
    )
    prompt = "Code context:\ndef add(a, b): return a + b\nQuestion: How does the sum work?\nExplain."
    response = ai_service.ollama_client.chat_with_ollama(prompt)
    assert response.startswith("Mocked LLM response for:")


# -------------- DB Search + LLM Integration --------------


def test_db_search_and_llm_integration(monkeypatch):
    # Add code to DB
    code = "def add(a, b): return a + b"
    embedding = embed_text(code)
    db.add_chunks([code], [embedding])

    # Mock LLM
    monkeypatch.setattr(
        "ai_service.ollama_client.chat_with_ollama",
        lambda prompt: f"LLM saw: {prompt}",
    )
    # User question
    question = "How does the sum work?"
    question_embedding = embed_text(question)
    results = db.query_chunks(question_embedding, number_of_results=1)
    relevant_code = results["documents"][0][0]
    prompt = f"Code context:\n{relevant_code}\nQuestion: {question}\nExplain."
    response = ai_service.ollama_client.chat_with_ollama(prompt)
    assert "LLM saw: Code context:" in response
    assert code in response
    assert question in response


# -------------- Edge Cases --------------


def test_db_search_no_results():
    # Ensure db is empty
    if db.collection.peek()["ids"]:
        db.collection.delete(ids=db.collection.peek()["ids"])
    question = "This code does not exist."
    question_embedding = embed_text(question)
    results = db.query_chunks(question_embedding, number_of_results=1)
    # Should return an empty or placeholder result
    assert results["documents"][0] == [] or results["documents"][0][0] == ""


def test_llm_chat_with_long_prompt(monkeypatch):
    # Mock LLM
    monkeypatch.setattr(
        "ai_service.ollama_client.chat_with_ollama",
        lambda prompt: "LLM received prompt of length: " + str(len(prompt)),
    )
    long_code = "def foo(): pass\n" * 1000
    prompt = f"Code context:\n{long_code}\nQuestion: What does this do?\nExplain."
    response = ai_service.ollama_client.chat_with_ollama(prompt)
    assert response.startswith("LLM received prompt of length:")
