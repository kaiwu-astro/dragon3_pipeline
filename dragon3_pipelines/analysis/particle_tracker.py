"""Particle tracking functionality for following individual particles through simulation"""

import logging
import multiprocessing
import os
import time
from glob import glob
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from dragon3_pipelines.io import HDF5FileProcessor
from dragon3_pipelines.utils import log_time

logger = logging.getLogger(__name__)


class ParticleTracker:
    """Track individual particles through time in simulation data"""
    
    def __init__(self, config_manager: Any):
        """
        Initialize ParticleTracker
        
        Args:
            config_manager: Configuration manager with simulation paths and settings
        """
        self.config = config_manager
        self.hdf5_file_processor = HDF5FileProcessor(config_manager)

    @log_time(logger)
    def get_particle_df_from_snap(self, df_dict: Dict[str, pd.DataFrame], 
                                   particle_name: int) -> pd.DataFrame:
        """
        Track a specific particle's evolution through time
        
        Args:
            df_dict: Dictionary containing 'singles', 'binaries', 'scalars' DataFrames 
                    (obtained from HDF5FileProcessor.read_file)
            particle_name: Particle Name to track (e.g. 94820)
        
        Returns:
            particle_history_df: DataFrame containing all time points for this particle
        """
        single_df_all = df_dict['singles']
        binary_df_all = df_dict['binaries']
        
        # Extract records for this particle from single star data
        single_particle_df = single_df_all[single_df_all['Name'] == particle_name].copy()
        
        # Extract records from binary data
        # Check if particle appears as member 1 or member 2 of a binary
        binary_as_1 = binary_df_all[binary_df_all['Bin Name1'] == particle_name].copy()
        binary_as_2 = binary_df_all[binary_df_all['Bin Name2'] == particle_name].copy()
        if not binary_as_1.empty:
            binary_as_1['state'] = 'binary'
            binary_as_1['companion_name'] = binary_as_1['Bin Name2']
            _binary = binary_as_1
        elif not binary_as_2.empty:
            binary_as_2['state'] = 'binary'
            binary_as_2['companion_name'] = binary_as_2['Bin Name1']
            _binary = binary_as_2
        
        if (not binary_as_1.empty) and (not binary_as_2.empty):
            # Particle appears as both Bin Name1 and Bin Name2 - this is an error
            try:
                scalars_df = df_dict.get('scalars', None)
                if scalars_df is None:
                    scalar_ttot = None
                elif 'TTOT' in scalars_df.columns:
                    scalar_ttot = np.array(scalars_df['TTOT'].unique())
                else:
                    scalar_ttot = np.array(scalars_df.index.unique())
            except Exception as e:
                logger.warning(f"Warning: Failed to extract TTOT from df_dict['scalars']: {e}")
                scalar_ttot = None

            t1 = np.array(binary_as_1['TTOT'].unique()) if 'TTOT' in binary_as_1.columns else None
            t2 = np.array(binary_as_2['TTOT'].unique()) if 'TTOT' in binary_as_2.columns else None
            logger.error(
                f"Assertion failed: particle {particle_name} appears as both Bin Name1 and Bin Name2. "
                f"binary_as_1 TTOT={t1}, binary_as_2 TTOT={t2}, scalars TTOT={scalar_ttot}"
            )
            raise AssertionError(
                f"particle {particle_name} appears as both Bin Name1 and Bin Name2 at TTOT={scalar_ttot}"
            )
        
        # Merge all records
        if binary_as_1.empty and binary_as_2.empty:
            single_particle_df['state'] = 'single'
            single_particle_df['companion_name'] = np.nan
            particle_history_df = single_particle_df
        else:
            # Particle is in a binary
            particle_history_df = single_particle_df.merge(
                    _binary,
                    on='TTOT',
                    how='outer',                 
                    suffixes=('', '_from_binary')  # Columns with same name from single vs binary get suffix
                )
        
        if not particle_history_df.empty:
            particle_history_df = particle_history_df.sort_values('TTOT').reset_index(drop=True)
            return particle_history_df
        else:
            logger.warning(f"Warning: Particle {particle_name} not found at any of TTOT: {df_dict['scalars']['TTOT'].unique()}")
            return pd.DataFrame()

    @log_time(logger)
    def get_particle_summary(self, particle_history_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary information about a particle's evolution history
        
        Args:
            particle_history_df: DataFrame returned by track_particle
        
        Returns:
            summary_dict: Dictionary containing key information about particle evolution
        """
        if particle_history_df.empty:
            return {}
        
        summary = {
            'particle_name': particle_history_df['Name'].iloc[0] if 'Name' in particle_history_df.columns else None,
            'total_snapshots': len(particle_history_df),
            'time_range_myr': (particle_history_df['Time[Myr]'].min(), particle_history_df['Time[Myr]'].max()),
            'single_count': len(particle_history_df[particle_history_df['state'] == 'single']),
            'binary_count': len(particle_history_df[particle_history_df['state'].str.contains('binary', na=False)]),
            'initial_mass': particle_history_df['M'].iloc[0] if 'M' in particle_history_df.columns else None,
            'final_mass': particle_history_df['M'].iloc[-1] if 'M' in particle_history_df.columns else None,
            'stellar_types': particle_history_df['KW'].unique().tolist() if 'KW' in particle_history_df.columns else [],
        }
        
        return summary

    def _process_single_snap_for_particle(self, args: Tuple[str, int, str]) -> pd.DataFrame:
        """
        Worker function for parallel processing
        
        Args:
            args: Tuple of (hdf5_path, particle_name, simu_name)
            
        Returns:
            DataFrame with particle data from this snapshot
        """
        hdf5_path, particle_name, simu_name = args
        
        try:
            df_dict = self.hdf5_file_processor.read_file(hdf5_path, simu_name)
            particle_df = self.get_particle_df_from_snap(df_dict, particle_name)
            return particle_df
        except Exception as e:
            logger.error(f"Error processing {hdf5_path} for particle {particle_name}: {type(e).__name__}: {e}")
            return pd.DataFrame()

    @log_time(logger)
    def get_particle_new_df_all(self, simu_name: str, particle_name: int, 
                               update: bool = True) -> pd.DataFrame:
        """
        Get complete evolution history of a particle throughout the simulation
        Uses caching and parallel processing
        
        Args:
            simu_name: Name of the simulation
            particle_name: Particle Name to track
            update: If True, process new snapshots; if False, only return cached data
            
        Returns:
            DataFrame containing complete particle evolution history
        """
        cache_base = self.config.particle_df_cache_dir_of[simu_name]
        os.makedirs(cache_base, exist_ok=True)
        
        all_feather_path = os.path.join(cache_base, f"{particle_name}_ALL.df.feather")
        
        # 1. Read existing aggregated cache
        old_df_all = pd.DataFrame()
        particle_skip_until = -1.0
        
        if os.path.exists(all_feather_path):
            try:
                old_df_all = pd.read_feather(all_feather_path)
                if not old_df_all.empty and 'TTOT' in old_df_all.columns:
                    particle_skip_until = old_df_all['TTOT'].max()
                    logger.info(f"Loaded existing particle df for {particle_name}, records: {len(old_df_all)}, max TTOT: {particle_skip_until}")
            except Exception as e:
                logger.warning(f"Failed to read aggregated cache {all_feather_path}: {e}")

        if not update:
            return old_df_all
        
        # 2. Get and filter file list
        # 获取所有快照文件
        hdf5_snap_files = sorted(
            glob(self.config.pathof[simu_name] + '/**/*.h5part'), 
            key=lambda fn: self.hdf5_file_processor.get_hdf5_name_time(fn)
        )
        WAIT_SNAPSHOT_AGE_HOUR = 24
        cutoff = time.time() - WAIT_SNAPSHOT_AGE_HOUR * 3600
        hdf5_snap_files = [
            fn for fn in hdf5_snap_files
            if os.path.getmtime(fn) <= cutoff
        ]

        files_to_process = [f for f in hdf5_snap_files if self.hdf5_file_processor.get_hdf5_name_time(f) > particle_skip_until]
        
        if not files_to_process:
            logger.info(f"No new snapshots to process for particle {particle_name}")
            return old_df_all

        logger.info(f"Found {len(files_to_process)} new snapshots to process for particle {particle_name}: {files_to_process[0]} ... {files_to_process[-1]}")

        # 3. Prepare task arguments
        tasks = []
        for fpath in files_to_process:
            tasks.append((fpath, particle_name, simu_name))
            

        # 4. Parallel processing
        new_dfs = []
        consecutive_missing_count = 0
        MISSING_THRESHOLD = 5  # Particle may be merged/ejected, stop searching after threshold

        ctx = multiprocessing.get_context('forkserver')
        with ctx.Pool(
            processes=self.config.processes_count, 
            maxtasksperchild=self.config.tasks_per_child
        ) as pool:
            # imap returns result in order of input
            iterator = pool.imap(self._process_single_snap_for_particle, tasks)
            tqdm_iterator = tqdm(iterator, total=len(tasks), desc=f"Tracking {particle_name} in {simu_name}")

            try:
                for res_df in tqdm_iterator:
                    if res_df is not None and not res_df.empty:
                        new_dfs.append(res_df)
                        consecutive_missing_count = 0  # Reset counter
                    else:
                        consecutive_missing_count += 1
                    
                    if consecutive_missing_count >= MISSING_THRESHOLD:
                        logger.info(f"Particle {particle_name} missing for {consecutive_missing_count} consecutive snapshots. Stopping search early and starting merging results")
                        pool.terminate()  # Force terminate process pool
                        break
            except Exception as e:
                logger.warning(f"Process pool interrupted or error occurred: {e}")
        
        # 5. Merge and save
        if new_dfs:
            new_df_concat = pd.concat(new_dfs, ignore_index=True)
            if not old_df_all.empty:
                new_df_all = pd.concat([old_df_all, new_df_concat], ignore_index=True)
            else:
                new_df_all = new_df_concat
            
            # Deduplicate and sort
            if 'TTOT' in new_df_all.columns:
                new_df_all = new_df_all.sort_values('TTOT').drop_duplicates(subset=['TTOT'], keep='last').reset_index(drop=True)
            
            # Save aggregated result
            try:
                new_df_all.to_feather(all_feather_path)
                logger.info(f"Updated aggregated cache for {particle_name} at {all_feather_path}")
            except Exception as e:
                logger.error(f"Failed to save aggregated cache: {e}")
            
            return new_df_all
        else:
            return old_df_all
