import pytest
import chromadb
from dotenv import load_dotenv
import os
from ai_service.exceptions import NotFound


def create_db_test_collection(collection_name: str):
    load_dotenv()
    chroma_path = os.getenv("CHROMA_STORE_PATH")
    if not chroma_path:
        raise NotFound("Missing CHROMA_STORE_PATH environment variable")
    client = chromadb.PersistentClient(path=chroma_path)
    return client.get_or_create_collection(collection_name)


@pytest.fixture(scope="module")
def db_test_collection(request):
    collection_name = f"{request.module.__name__}"
    return create_db_test_collection(collection_name)


@pytest.fixture(autouse=True)
def patch_and_clean_db_collection(monkeypatch, db_test_collection):
    from ai_service import db  # Make sure this matches your code

    monkeypatch.setattr(db, "collection", db_test_collection)

    yield

    # Clean up after each test
    ids = db_test_collection.peek()["ids"]
    if ids:
        db_test_collection.delete(ids=ids)
