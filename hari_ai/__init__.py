"""HariAI — Built by Hari. Open to everyone.

Motto: Small enough to run anywhere. Intelligent enough to think. Open enough for everyone.
"""

from hari_ai.cluster.coordinator import Cluster
from hari_ai.model.config import HariNanoConfig
from hari_ai.model.transformer import HariNanoModel
from hari_ai.runtime.chat import ChatEngine
from hari_ai.runtime.engine import HariAI
from hari_ai.search.engine import SearchEngine

__version__ = "0.1.0"
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
