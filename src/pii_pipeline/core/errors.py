"""Pipeline errors. Each carries a stable `code` and the HTTP `status_code` the API returns for it.

Messages never contain document text.
"""
from __future__ import annotations


class PIIError(Exception):
    code = "pii_error"
    status_code = 500

    def __init__(self, message: str, *, code: str | None = None, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code

    def to_dict(self) -> dict:
        return {"detail": self.message, "code": self.code}


class ConfigError(PIIError, ValueError):
    """The configuration is invalid. Also a ValueError, so callers can catch either."""
    code = "invalid_config"
    status_code = 500

    def __init__(self, message: str, **kw):
        if not message.startswith("invalid config"):
            message = f"invalid config: {message}"
        super().__init__(message, **kw)


class ModelNotAvailableError(PIIError):
    """A configured model or its optional dependency cannot be loaded."""
    code = "model_not_available"
    status_code = 503


class TextTooLargeError(PIIError):
    code = "text_too_large"
    status_code = 413


class InvalidTextError(PIIError):
    """The text cannot be processed, e.g. it contains a lone UTF-16 surrogate."""
    code = "invalid_text"
    status_code = 422


class UnsupportedLanguageError(PIIError):
    code = "unsupported_language"
    status_code = 422


class ChecksumMismatchError(PIIError):
    """A detection result was applied to a different text from the one it was computed on."""
    code = "checksum_mismatch"
    status_code = 409
