"""
Multimodal Context Store

Manages incoming context as inherently multimodal data.
A context is a dict with optional keys: "text", "image", "audio".
  - "text":  a string
  - "image": a file path (str) to an image
  - "audio": a file path (str) to an audio file
Any combination of these keys is valid, including a single modality.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def validate_context(ctx: dict) -> dict:
    """Validate and normalise a raw context dict.

    Accepted keys: ``text``, ``image``, ``audio``.
    At least one must be present.  Unknown keys are silently dropped.

    Returns a clean copy containing only the recognised keys.

    Raises ``ValueError`` when no valid modality is present.
    """
    VALID_KEYS = {"text", "image", "audio"}
    cleaned = {k: v for k, v in ctx.items() if k in VALID_KEYS and v}
    if not cleaned:
        raise ValueError(
            "Context must contain at least one of: 'text', 'image', 'audio'. "
            f"Got keys: {list(ctx.keys())}"
        )
    return cleaned


# ---------------------------------------------------------------------------
# Context Store
# ---------------------------------------------------------------------------

class ContextStore:
    """Thread-safe store for the latest multimodal context.

    The planner polls ``get()``; context is pushed via ``push()``.
    Each push increments an internal version id so the planner can detect
    changes cheaply.
    """

    def __init__(self) -> None:
        self._context: dict = {}
        self._version: int = 0

    # -- write -----------------------------------------------------------------

    def push(self, raw_context: dict) -> None:
        """Accept a new multimodal context dict.

        The dict is validated and stored with a bumped version id.
        """
        cleaned = validate_context(raw_context)
        self._version += 1
        self._context = {**cleaned, "_version": self._version}
        logger.info("[context] Stored context v%d  keys=%s", self._version, list(cleaned.keys()))

    # -- read ------------------------------------------------------------------

    def get(self) -> dict:
        """Return a *copy* of the current context (empty dict if nothing yet)."""
        return deepcopy(self._context)

    @property
    def version(self) -> int:
        return self._version
