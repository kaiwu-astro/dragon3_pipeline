"""I/O module for reading and processing dragon3 simulation files"""

from dragon3_pipelines.io.hdf5_reader import HDF5FileProcessor
from dragon3_pipelines.io.lagr_reader import LagrFileProcessor
from dragon3_pipelines.io.collision_reader import Coll13FileProcessor, Coal24FileProcessor
from dragon3_pipelines.io.text_parsers import (
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
    read_bdat,
    read_coll_13,
    read_coal_24,
    make_l7header,
    read_lagr_7,
    l7df_to_physical_units,
    transform_l7df_to_sns_friendly,
)

__all__ = [
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
    'read_bdat',
    'read_coll_13',
    'read_coal_24',
    'make_l7header',
    'read_lagr_7',
    'l7df_to_physical_units',
    'transform_l7df_to_sns_friendly',
]
