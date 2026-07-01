"""Visualization tools for creating plots and figures"""

from dragon3_pipelines.visualization.base import (
    BaseVisualizer,
    BaseHDF5Visualizer,
    HDF5Visualizer,
    BaseContinousFileVisualizer,
    set_mpl_fonts,
    add_grid,
)
from dragon3_pipelines.visualization.single_star import SingleStarVisualizer
from dragon3_pipelines.visualization.binary_star import BinaryStarVisualizer
from dragon3_pipelines.visualization.lagrangian import LagrVisualizer
from dragon3_pipelines.visualization.galactic_orbit import GalacticOrbitVisualizer
from dragon3_pipelines.visualization.collision import CollCoalVisualizer
from dragon3_pipelines.visualization.particle_history import ParticleHistoryVisualizer
from dragon3_pipelines.visualization.purge import PlotPurger, PurgeResult

__all__ = [
    "BaseVisualizer",
    "BaseHDF5Visualizer",
    "HDF5Visualizer",
    "BaseContinousFileVisualizer",
    "SingleStarVisualizer",
    "BinaryStarVisualizer",
    "LagrVisualizer",
    "GalacticOrbitVisualizer",
    "CollCoalVisualizer",
    "ParticleHistoryVisualizer",
    "PlotPurger",
    "PurgeResult",
    "set_mpl_fonts",
    "add_grid",
]
