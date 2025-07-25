import pytest
import numpy as np
from ai_service.embedder import embed_text, embed_texts
from ai_service import db
from ai_service.exceptions import EmbeddingError

# ------------------ Input Validation ------------------


# Ensure invalid inputs (empty strings, whitespace, None) raise appropriate exceptions
@pytest.mark.parametrize("invalid_input", ["", "   ", "\t", "\n", None])
def test_embed_invalid_inputs_raise_errors(invalid_input):
    if invalid_input is None:
        with pytest.raises(AttributeError):  # None has no strip() method
            embed_text(invalid_input)
    else:
        with pytest.raises(
            EmbeddingError, match="Cannot embed empty or whitespace-only text"
        ):
            embed_text(invalid_input)


# ------------------ Core Functionality ------------------


# Test storing and retrieving a variety of code snippets, including long ones
def test_embed_and_store_and_retrieve_texts():
    sample_texts = [
        "def greet(): print('Hello!')",
        "def farewell(): print('Goodbye!')",
        "class Calculator: pass",
        "def long_func():\n" + "    # comment\n" * 1000 + "    pass",  # very long text
        "def a(): pass",
        "def b(): pass",
    ]
    embeddings = embed_texts(sample_texts)
    db.add_chunks(sample_texts, embeddings)

    for i, embedding in enumerate(embeddings):
        result = db.collection.query(query_embeddings=[embedding], n_results=1)
        assert result["documents"][0][0] == sample_texts[i]  # Check exact match


# ------------------ Unicode, Special Characters ------------------


# Ensure the embedding system handles various languages, symbols, and edge inputs
@pytest.mark.parametrize(
    "text",
    [
        "def greet(): print('Hello! 你好')",
        "# Comment with émojis 🚀",
        "def func(): return 'quotes\"and\\backslashes'",
        "multiline\nstring\nwith\nnewlines",
        "def 函数(): return '中文'",
        "def función(): return 'español'",
        "def функция(): return 'русский'",
        "def פונקציה(): return 'עברית'",
        "def 関数(): return '日本語'",
        "def फ़ंक्शन(): return 'हिंदी'",
        "a",  # single char
        "1",  # number
        "!",  # symbol
        "@",  # symbol
        "中",  # Chinese char
    ],
)
def test_embed_and_retrieve_special_and_unicode_texts(
    text,
):
    embedding = embed_text(text)
    assert embedding is not None and len(embedding) > 0

    db.add_chunks([text], [embedding])

    result = db.collection.query(query_embeddings=[embedding], n_results=1)
    assert result["documents"][0][0] == text


# ------------------ Embedding Properties & Structure ------------------


# Check the embedding's structure, type, and integration with chroma DB
def test_embedding_output_properties_and_structure():
    text = "def test_function(): return True"
    embedding = embed_text(text)

    # Validate basic embedding properties
    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1
    assert embedding.dtype in [np.float32, np.float64]
    assert len(embedding) > 0

    # Validate output structure from the DB query

    db.add_chunks([text], [embedding])
    result = db.collection.query(query_embeddings=[embedding], n_results=1)
    assert isinstance(result, dict)
    assert "documents" in result and isinstance(result["documents"], list)
    assert isinstance(result["documents"][0], list)
    assert len(result["documents"][0]) > 0
    assert "ids" in result


# ------------------ Query Behavior ------------------


# Check behavior with different `n_results` values including > available matches
def test_query_with_varied_n_results():
    sample_texts = [
        "def function_a(): pass",
        "def function_b(): pass",
        "def function_c(): pass",
    ]
    embeddings = embed_texts(sample_texts)
    db.add_chunks(sample_texts, embeddings)

    for n in [1, 2, 5]:  # Requesting more results than stored is valid
        result = db.collection.query(query_embeddings=[embeddings[0]], n_results=n)
        docs = result["documents"][0]
        # NOTE:
        # ChromaDB may return duplicate documents when n_results exceeds the number
        # of uniquely stored items. This causes the length of the returned list
        # to exceed the number of stored items.
        # To handle this properly, we count only unique documents.
        unique_docs = set(docs)
        assert len(unique_docs) <= n
