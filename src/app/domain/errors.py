class AppError(Exception):
    """Base class for errors safe to map to a stable public contract."""

    code = "application_error"
    public_message = "The operation could not be completed."


class ConfigurationError(AppError):
    code = "configuration_error"
    public_message = "The application configuration is invalid."


class ModelUnavailableError(AppError):
    code = "model_unavailable"
    public_message = "The local language model is unavailable."


class ModelTimeoutError(AppError):
    code = "model_timeout"
    public_message = "The local language model did not respond in time."


class UnsupportedDocumentTypeError(AppError):
    code = "unsupported_document_type"
    public_message = "The document type is not supported."


class DocumentTooLargeError(AppError):
    code = "document_too_large"
    public_message = "The document exceeds the configured size limit."


class PdfEncryptedError(AppError):
    code = "pdf_encrypted"
    public_message = "Encrypted PDF documents are not supported."


class PdfCorruptedError(AppError):
    code = "pdf_corrupted"
    public_message = "The PDF document is invalid or corrupted."


class NoExtractableTextError(AppError):
    code = "document_contains_no_extractable_text"
    public_message = "The document contains no extractable text."


class UnsafeUrlError(AppError):
    code = "unsafe_url"
    public_message = "The URL is not allowed."


class RemoteDocumentFetchError(AppError):
    code = "remote_document_fetch_failed"
    public_message = "The remote document could not be fetched."


class DocumentParsingError(AppError):
    code = "document_parsing_failed"
    public_message = "The document could not be parsed."


class EmbeddingError(AppError):
    code = "embedding_failed"
    public_message = "Text embedding failed."


class VectorStoreError(AppError):
    code = "vector_store_failed"
    public_message = "The vector store operation failed."


class CollectionCompatibilityError(AppError):
    code = "collection_incompatible"
    public_message = (
        "The collection is incompatible with the configured embeddings. "
        "Use another collection or rebuild the index."
    )


class DocumentNotFoundError(AppError):
    code = "document_not_found"
    public_message = "The document was not found."
