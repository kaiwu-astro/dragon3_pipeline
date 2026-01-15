#!/usr/bin/env python3
"""
Backward compatibility wrapper for nbody_tools.py

This module maintains backward compatibility by re-exporting functions and classes
from the new dragon3_pipelines package structure.

For new code, please import directly from dragon3_pipelines submodules:
    from dragon3_pipelines.utils import save, read, get_output
    from dragon3_pipelines.io import HDF5FileProcessor, LagrFileProcessor
    from dragon3_pipelines.visualization import HDF5Visualizer
    etc.
"""
import warnings

# Issue a deprecation warning when this module is imported
warnings.warn(
    "nbody_tools.py is deprecated. Please import from dragon3_pipelines submodules instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export utilities from dragon3_pipelines.utils
from dragon3_pipelines.utils import (
    save,
    read,
    get_output,
    can_convert_to_float,
    log_time,
    BlackbodyColorConverter,
)

# Re-export I/O functions and classes from dragon3_pipelines.io
from dragon3_pipelines.io import (
    HDF5FileProcessor,
    LagrFileProcessor,
    Coll13FileProcessor,
    Coal24FileProcessor,
    get_scale_dict,
    get_scale_dict_from_hdf5_df,
    load_snapshot_data,
    dataframes_from_hdf5_file,
    merge_multiple_hdf5_dataframes,
    decode_bytes_columns_inplace,
    tau_gw,
    load_GWTC_catalog,
    get_valueStr_of_namelist_key,
    read_bwdat,
    read_coll_13,
    read_coal_24,
    make_l7header,
    read_lagr_7,
    l7df_to_physical_units,
    transform_l7df_to_sns_friendly,
)

# Re-export analysis functions from dragon3_pipelines.analysis
from dragon3_pipelines.analysis import (
    ParticleTracker,
)

# Re-export visualization classes from dragon3_pipelines.visualization
from dragon3_pipelines.visualization import (
    BaseVisualizer,
    BaseHDF5Visualizer,
    HDF5Visualizer,
    BaseContinousFileVisualizer,
    SingleStarVisualizer,
    BinaryStarVisualizer,
    LagrVisualizer,
    CollCoalVisualizer,
    set_mpl_fonts,
    add_grid,
)

# Re-export config from dragon3_pipelines.config
from dragon3_pipelines.config import (
    ConfigManager,
    load_config,
)

# Standard library and third-party imports that were in original nbody_tools.py
import datetime
import gzip
import io
import os
import pickle as pk
import re
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from functools import lru_cache, wraps
from glob import glob
from subprocess import Popen, call, run

import astropy
import astropy.constants as constants
import astropy.units as u
from astropy.units.quantity import Quantity
import h5py
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import colour
from pandas.api.types import is_list_like
from scipy.interpolate import interp1d
from scipy.io import FortranFile
from tqdm.auto import tqdm

# Constants
pc_to_AU = constants.pc.to(u.AU).value

# For backwards compatibility, try to import colour colorimetry
try:
    from colour.colorimetry import SpectralDistribution, msds_to_XYZ, planck_law
    from colour.models import XYZ_to_sRGB
except ImportError:
    pass

# Export all re-exported symbols
__all__ = [
    # Utils
    'save',
    'read',
    'get_output',
    'can_convert_to_float',
    'log_time',
    'BlackbodyColorConverter',
    # I/O
    'HDF5FileProcessor',
    'LagrFileProcessor',
    'Coll13FileProcessor',
    'Coal24FileProcessor',
    'get_scale_dict',
    'get_scale_dict_from_hdf5_df',
    'load_snapshot_data',
    'dataframes_from_hdf5_file',
    'merge_multiple_hdf5_dataframes',
    'decode_bytes_columns_inplace',
    'tau_gw',
    'load_GWTC_catalog',
    'get_valueStr_of_namelist_key',
    'read_bwdat',
    'read_coll_13',
    'read_coal_24',
    'make_l7header',
    'read_lagr_7',
    'l7df_to_physical_units',
    'transform_l7df_to_sns_friendly',
    # Analysis
    'ParticleTracker',
    # Visualization
    'BaseVisualizer',
    'BaseHDF5Visualizer',
    'HDF5Visualizer',
    'BaseContinousFileVisualizer',
    'SingleStarVisualizer',
    'BinaryStarVisualizer',
    'LagrVisualizer',
    'CollCoalVisualizer',
    'set_mpl_fonts',
    'add_grid',
    # Config
    'ConfigManager',
    'load_config',
    # Constants
    'pc_to_AU',
]
