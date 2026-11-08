"""HariAI by K R Hari Prajwal: local conversation and experimental typed decisions."""

from hari_ai.cluster.coordinator import Cluster
from hari_ai.model.config import HariNanoConfig
from hari_ai.model.transformer import HariNanoModel
from hari_ai.runtime.chat import ChatEngine
from hari_ai.runtime.engine import HariAI
from hari_ai.search.engine import SearchEngine

__version__ = "0.2.0a1"
__author__ = "K R Hari Prajwal"

__all__ = [
    "HariAI",
    "Cluster",
    "ChatEngine",
    "SearchEngine",
    "HariNanoModel",
    "HariNanoConfig",
    "__version__",
    "__author__",
]
