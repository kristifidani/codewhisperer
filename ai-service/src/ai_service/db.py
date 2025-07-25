# --- Handle ChromaDB NumPy compatibility ---
# ChromaDB may fail with newer NumPy versions >2.0 that removed np.float_
# This ensures backward compatibility by aliasing float_ to float64
import numpy as np

if not hasattr(np, "float_"):
    np.float_ = np.float64


import uuid
import chromadb
import os
from dotenv import load_dotenv
from .exceptions import DatabaseError, NotFound, InvalidParam

load_dotenv()

# Set up ChromaDB client and persistent collection
chroma_path = os.getenv("CHROMA_STORE_PATH")
if not chroma_path:
    raise NotFound("Missing CHROMA_STORE_PATH environment variable")

client = chromadb.PersistentClient(path=chroma_path)
collection = client.get_or_create_collection("code_chunks")


def add_chunks(chunks: list[str], embeddings: list[list[float]]) -> None:
    """
    Add new code chunks and their embeddings to ChromaDB.

    Args:
        chunks: Code or text chunks to store.
        embeddings: Corresponding vector embeddings.

    Raises:
        DatabaseError: If database operation fails.
    """

    try:
        ids = [f"chunk-{uuid.uuid4()}" for _ in range(len(chunks))]
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
        )
    except Exception as e:
        raise DatabaseError(f"Failed to add chunks: {e}") from e


def query_chunks(text_embedding: list[float], number_of_results: int = 5) -> dict:
    """
    Query ChromaDB for most similar documents.

    Args:
        text_embedding: Vector embedding of a user query.
        number_of_results: Number of results to return (1-100).

    Returns:
        Dict containing 'documents', 'distances', 'metadatas', and 'ids'.

    Raises:
        DatabaseError: If the query fails.
        InvalidParam: If parameters are invalid.
    """
    if text_embedding is None or len(text_embedding) == 0:
        raise InvalidParam("Query embedding is empty.")
    if (
        not isinstance(number_of_results, int)
        or number_of_results < 1
        or number_of_results > 100
    ):
        raise InvalidParam("number_of_results must be an integer between 1 and 100")

    try:
        return collection.query(
            query_embeddings=[text_embedding], n_results=number_of_results
        )
    except Exception as e:
        raise DatabaseError(f"Failed to query chunks: {e}") from e
