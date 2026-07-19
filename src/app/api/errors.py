from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.domain.errors import (
    AppError,
    CollectionCompatibilityError,
    DocumentNotFoundError,
    DocumentParsingError,
    DocumentTooLargeError,
    EmbeddingError,
    ModelTimeoutError,
    ModelUnavailableError,
    NoExtractableTextError,
    PdfCorruptedError,
    PdfEncryptedError,
    RemoteDocumentFetchError,
    UnsafeUrlError,
    UnsupportedDocumentTypeError,
    VectorStoreError,
)

ERROR_STATUS: dict[type[AppError], int] = {
    CollectionCompatibilityError: status.HTTP_409_CONFLICT,
    DocumentNotFoundError: status.HTTP_404_NOT_FOUND,
    DocumentParsingError: status.HTTP_400_BAD_REQUEST,
    DocumentTooLargeError: status.HTTP_413_CONTENT_TOO_LARGE,
    EmbeddingError: status.HTTP_503_SERVICE_UNAVAILABLE,
    ModelTimeoutError: status.HTTP_504_GATEWAY_TIMEOUT,
    ModelUnavailableError: status.HTTP_503_SERVICE_UNAVAILABLE,
    NoExtractableTextError: status.HTTP_400_BAD_REQUEST,
    PdfCorruptedError: status.HTTP_400_BAD_REQUEST,
    PdfEncryptedError: status.HTTP_400_BAD_REQUEST,
    RemoteDocumentFetchError: status.HTTP_502_BAD_GATEWAY,
    UnsafeUrlError: status.HTTP_400_BAD_REQUEST,
    UnsupportedDocumentTypeError: status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    VectorStoreError: status.HTTP_503_SERVICE_UNAVAILABLE,
}


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", "unknown"))


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=ERROR_STATUS.get(type(exc), status.HTTP_400_BAD_REQUEST),
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.public_message,
                    "request_id": _request_id(request),
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request,
        _exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "The request payload is invalid.",
                    "request_id": _request_id(request),
                }
            },
        )
