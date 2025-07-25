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
from .exceptions import DatabaseError, NotFound

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
