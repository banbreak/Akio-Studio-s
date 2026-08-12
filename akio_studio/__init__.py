"""Akio Studio — a data-driven, audience-reinforced local anime studio engine.

The package is organized as six cooperating modules:

- :mod:`akio_studio.metrics_engine` — social metrics ingestion, R2E scoring,
  greenlight validation, and NLP fan-theory mining.
- :mod:`akio_studio.lore_graph_agent` — the multi-agent writer's room backed by
  a persistent lore knowledge graph.
- :mod:`akio_studio.pool_coordinator` — local model pool coordination for the
  Apple Silicon unified-memory budget, plus DPO latent feedback logging.
- :mod:`akio_studio.actor_sdk_exporter` — packaging of validated characters
  into portable ``.synthetic_actor`` bundles.
- :mod:`akio_studio.file_tree_manager` — production directory layout and
  atomic metadata persistence.
- ``build_mac_app.sh`` — the native macOS application bundler (shell, not
  importable).
"""

from akio_studio.config import StudioConfig

__all__ = ["StudioConfig", "__version__"]
__version__ = "1.0.0"
