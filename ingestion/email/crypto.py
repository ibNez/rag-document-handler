from __future__ import annotations

"""Utility functions for encrypting and decrypting sensitive email data."""

import os
from cryptography.fernet import Fernet

_KEY_ENV_VAR = "EMAIL_ENCRYPTION_KEY"


def _get_fernet() -> Fernet:
    """Return a :class:`Fernet` instance from the configured key.

    The key is sourced from the ``EMAIL_ENCRYPTION_KEY`` environment variable
    so it can be managed outside of version control.
    """
    key = os.environ.get(_KEY_ENV_VAR)
    if not key:
        raise RuntimeError(f"{_KEY_ENV_VAR} is not set")
    return Fernet(key.encode())


def encrypt(value: str) -> str:
    """Encrypt ``value`` using Fernet symmetric encryption."""
    fernet = _get_fernet()
    return fernet.encrypt(value.encode()).decode()


def decrypt(token: str) -> str:
    """Decrypt a Fernet ``token`` back into plaintext."""
    fernet = _get_fernet()
    return fernet.decrypt(token.encode()).decode()
