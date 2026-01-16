#!/usr/bin/env python3
"""
Backward compatibility wrapper for plot_nbody.py

This module maintains backward compatibility by importing from the new
dragon3_pipelines package structure. All functionality has been migrated
to dragon3_pipelines.

For new code, please use:
    from dragon3_pipelines import main, SimulationPlotter
"""
import sys

if sys.version_info < (3, 11):
    raise RuntimeError("Need python >= 3.11")

import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"

# Import main function and SimulationPlotter from the new package
from dragon3_pipelines.__main__ import main, SimulationPlotter

# Re-export for backward compatibility
__all__ = ['main', 'SimulationPlotter']

if __name__ == "__main__":
    main()
