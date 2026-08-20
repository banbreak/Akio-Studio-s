"""Exception hierarchy for Akio Studio.

Every module raises subclasses of :class:`AkioStudioError` so callers can
catch studio failures without swallowing unrelated bugs.
"""

from __future__ import annotations


class AkioStudioError(Exception):
    """Base class for all Akio Studio errors."""


class MetricsError(AkioStudioError):
    """Raised when platform metrics are malformed or cannot be evaluated."""


class WebhookDispatchError(MetricsError):
    """Raised when a greenlight webhook POST fails or is rejected."""


class LoreGraphError(AkioStudioError):
    """Raised for lore knowledge-graph consistency failures."""


class EntityNotFoundError(LoreGraphError):
    """Raised when a relation references an entity that was never added."""


class PoolCoordinatorError(AkioStudioError):
    """Raised for local model-pool orchestration failures."""


class OllamaUnavailableError(PoolCoordinatorError):
    """Raised when neither the Ollama HTTP API nor the CLI is reachable."""


class MemoryBudgetExceededError(PoolCoordinatorError):
    """Raised when loading a stage would overflow the unified-memory budget."""


class ActorExportError(AkioStudioError):
    """Raised when a ``.synthetic_actor`` bundle cannot be assembled."""


class MissingAssetError(ActorExportError):
    """Raised when a required actor asset path does not exist."""


class FileTreeError(AkioStudioError):
    """Raised when the production directory tree cannot be created or written."""


class CloudRenderError(AkioStudioError):
    """Base class for remote (cloud GPU) rendering failures."""


class CloudAuthError(CloudRenderError):
    """Raised when the cloud endpoint or its credential is missing or rejected."""


class CloudJobFailedError(CloudRenderError):
    """Raised when a cloud render job reaches a terminal failure state."""


class CloudSubmitAmbiguousError(CloudRenderError):
    """Raised when a submission failed in flight and may still have started.

    Distinct from a plain :class:`CloudRenderError` because the remedy is
    different: the job may be running and billing right now, so the operator
    must reconcile against the provider console rather than simply resubmit.
    """


class CloudTimeoutError(CloudRenderError):
    """Raised when a cloud render job outlives its wall-clock budget."""


class CloudBudgetExceededError(CloudRenderError):
    """Raised when a cloud render job's reported cost exceeds the budget."""
