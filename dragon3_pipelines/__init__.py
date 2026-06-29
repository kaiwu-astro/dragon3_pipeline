"""
Dragon3 Pipelines - N-body simulation data analysis and visualization toolkit
"""

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

if TYPE_CHECKING:
    from dragon3_pipelines.__main__ import SimulationPlotter, main


def __getattr__(name: str) -> Any:
    """Lazily expose CLI entry points without importing __main__ during -m startup."""
    if name in {"main", "SimulationPlotter"}:
        from dragon3_pipelines.__main__ import SimulationPlotter, main

        exported = {"main": main, "SimulationPlotter": SimulationPlotter}
        globals().update(exported)
        return exported[name]
    if name == "BinaryStellarTypeExtractor":
        from dragon3_pipelines.analysis import BinaryStellarTypeExtractor

        globals()[name] = BinaryStellarTypeExtractor
        return BinaryStellarTypeExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Public API
__all__ = [
    "__version__",
    "main",
    "SimulationPlotter",
    "BinaryStellarTypeExtractor",
]
