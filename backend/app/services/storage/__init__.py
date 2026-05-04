"""Storage adapters for original image bytes."""

from .local import LocalStorage, StoredObject

__all__ = ["LocalStorage", "StoredObject"]
