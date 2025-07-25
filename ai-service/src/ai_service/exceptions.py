class AIServiceError(Exception):
    """Base exception for all AI service errors."""

    pass


class EmbeddingError(AIServiceError):
    """Base exception for embedding errors."""

    EMPTY_INPUT_MESSAGE = "Cannot embed empty or whitespace-only texts"


class LLMQueryError(AIServiceError):
    """Raised when querying the LLM fails."""

    pass


class DatabaseError(AIServiceError):
    """Raised when database operations fail."""

    pass


class NotFound(AIServiceError):
    """Raised when something is not found."""

    pass


class InvalidParam(AIServiceError):
    """Raised when something is invalid."""

    pass
