"""Particle tracking functionality for following individual particles through simulation"""

import gc
import logging
import multiprocessing
import os
import time
from glob import glob
from typing import Any, Dict, Optional, Tuple, Union

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
    def get_particle_df_from_hdf5_file(
        self, 
        df_dict: Dict[str, pd.DataFrame], 
        particle_name: Union[int, str],
        hdf5_file_path: Optional[str] = None,
        simu_name: Optional[str] = None,
        save_cache: bool = False
    ) -> Union[pd.DataFrame, Dict[int, pd.DataFrame]]:
        """
        Track particle(s) evolution through snapshots in an HDF5 file
        
        Args:
            df_dict: Dictionary containing 'singles', 'binaries', 'scalars' DataFrames 
                    (obtained from HDF5FileProcessor.read_file)
                    Note: This dict contains data for MULTIPLE snapshots from ONE HDF5 file
            particle_name: Particle Name to track (e.g. 94820) or 'all' to process all particles
            hdf5_file_path: Path to HDF5 file (required when save_cache=True)
            simu_name: Simulation name (required when save_cache=True)
            save_cache: If True, save individual particle DataFrames to cache files
        
        Returns:
            If particle_name is int: DataFrame containing all time points for this particle
            If particle_name is 'all': Dict mapping particle_name -> DataFrame
        """
        # Handle 'all' particles mode
        if particle_name == 'all':
            if save_cache and (hdf5_file_path is None or simu_name is None):
                raise ValueError("hdf5_file_path and simu_name are required when save_cache=True")
            
            single_df_all = df_dict['singles']
            if single_df_all.empty or 'Name' not in single_df_all.columns:
                logger.warning("No particles found in singles DataFrame")
                return {}
            
            all_particle_names = single_df_all['Name'].unique()
            result_dict = {}
            
            for pname in all_particle_names:
                particle_df = self._get_single_particle_df(df_dict, pname)
                
                if save_cache and not particle_df.empty:
                    self._save_particle_cache(particle_df, pname, hdf5_file_path, simu_name)
                
                result_dict[pname] = particle_df
            
            return result_dict
        
        # Single particle mode
        return self._get_single_particle_df(df_dict, particle_name)
    
    def _get_single_particle_df(self, df_dict: Dict[str, pd.DataFrame], 
                                particle_name: int) -> pd.DataFrame:
        """
        Extract a single particle's data from df_dict
        
        Args:
            df_dict: Dictionary containing 'singles', 'binaries', 'scalars' DataFrames
            particle_name: Particle Name to track
        
        Returns:
            DataFrame containing all time points for this particle in the HDF5 file
        """
        single_df_all = df_dict['singles']
        binary_df_all = df_dict['binaries']
        
        # Extract records for this particle from single star data
        single_particle_df = single_df_all[single_df_all['Name'] == particle_name].copy()
        
        # Extract records from binary data
        # Check if particle appears as member 1 or member 2 of a binary
        # Note: this is necessary due to a star may sometimes be Bin 1 and sometimes Bin 2
        #       even in the same hdf5 file
        binary_as_1 = binary_df_all[binary_df_all['Bin Name1'] == particle_name].copy()
        binary_as_1['state'] = 'binary'
        binary_as_1['companion_name'] = binary_as_1['Bin Name2']
        binary_as_2 = binary_df_all[binary_df_all['Bin Name2'] == particle_name].copy()
        binary_as_2['state'] = 'binary'
        binary_as_2['companion_name'] = binary_as_2['Bin Name1']

        _binary = pd.concat([binary_as_1, binary_as_2], ignore_index=True)

        if not _binary['TTOT'].is_unique:
            logger.warning(f"Warning: Particle {particle_name} is found in both components at TTOT = {_binary['TTOT'][_binary['TTOT'].duplicated()].unique()}")
            _binary = _binary.drop_duplicates(subset=['TTOT'], keep='first')
        
        # Merge all records
        if binary_as_1.empty and binary_as_2.empty:
            single_particle_df['state'] = 'single'
            single_particle_df['companion_name'] = np.nan
            particle_df = single_particle_df
        else:
            # Particle is in a binary
            particle_df = single_particle_df.merge(
                    _binary,
                    on='TTOT',
                    how='outer',                 
                    suffixes=('', '_from_binary')  # Columns with same name from single vs binary get suffix
                )
        
        if not particle_df.empty:
            particle_df = particle_df.sort_values('TTOT').reset_index(drop=True)
            return particle_df
        else:
            logger.debug(f"Particle {particle_name} not found at any of TTOT: {df_dict['scalars']['TTOT'].unique()}")
            return pd.DataFrame()
    
    def _save_particle_cache(self, particle_df: pd.DataFrame, particle_name: int,
                            hdf5_file_path: str, simu_name: str) -> None:
        """
        Save particle DataFrame to cache file
        
        Args:
            particle_df: DataFrame for a single particle from one HDF5 file
            particle_name: Particle name
            hdf5_file_path: Path to the HDF5 file
            simu_name: Simulation name
        """
        # Get HDF5 file time from filename
        hdf5_time = self.hdf5_file_processor.get_hdf5_file_time_from_filename(hdf5_file_path)
        
        # Create particle-specific subdirectory
        cache_base = self.config.particle_df_cache_dir_of[simu_name]
        particle_cache_dir = os.path.join(cache_base, str(particle_name))
        os.makedirs(particle_cache_dir, exist_ok=True)
        
        # Save to feather file with naming: {hdf5_time}.{particle_name}.df.feather
        cache_file = os.path.join(particle_cache_dir, f"{hdf5_time:.6f}.{particle_name}.df.feather")
        
        try:
            particle_df.to_feather(cache_file)
            logger.debug(f"Saved particle {particle_name} cache to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save particle {particle_name} cache: {e}")

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

    def _process_single_hdf5_for_particle(self, args: Tuple[str, int, str]) -> pd.DataFrame:
        """
        Worker function for parallel processing of HDF5 files
        
        Args:
            args: Tuple of (hdf5_file_path, particle_name, simu_name)
            
        Returns:
            DataFrame with particle data from all snapshots in this HDF5 file
        """
        hdf5_file_path, particle_name, simu_name = args
        
        try:
            df_dict = self.hdf5_file_processor.read_file(hdf5_file_path, simu_name)
            particle_df = self.get_particle_df_from_hdf5_file(df_dict, particle_name)
            return particle_df
        except Exception as e:
            logger.error(f"Error processing {hdf5_file_path} for particle {particle_name}: {type(e).__name__}: {e}")
            return pd.DataFrame()

    @log_time(logger)
    def update_one_particle_history_df(self, simu_name: str, particle_name: int, 
                                        update: bool = True) -> pd.DataFrame:
        """
        Get complete evolution history of a particle throughout the simulation
        Uses caching and parallel processing
        
        Args:
            simu_name: Name of the simulation
            particle_name: Particle Name to track
            update: If True, process new HDF5 files; if False, only return cached data
            
        Returns:
            DataFrame containing complete particle evolution history
        """
        cache_base = self.config.particle_df_cache_dir_of[simu_name]
        os.makedirs(cache_base, exist_ok=True)
        
        all_feather_path = os.path.join(cache_base, f"{particle_name}_ALL.df.feather")
        
        # 1. Read existing aggregated cache
        old_particle_history_df = pd.DataFrame()
        particle_skip_until = -1.0
        
        if os.path.exists(all_feather_path):
            try:
                old_particle_history_df = pd.read_feather(all_feather_path)
                if not old_particle_history_df.empty and 'TTOT' in old_particle_history_df.columns:
                    particle_skip_until = old_particle_history_df['TTOT'].max()
                    logger.info(f"Loaded existing particle df for {particle_name}, records: {len(old_particle_history_df)}, max TTOT: {particle_skip_until}")
            except Exception as e:
                logger.warning(f"Failed to read aggregated cache {all_feather_path}: {e}")

        if not update:
            return old_particle_history_df
        
        # 2. Get and filter file list
        # 获取所有HDF5文件
        hdf5_files = sorted(
            glob(self.config.pathof[simu_name] + '/**/*.h5part'), 
            key=lambda fn: self.hdf5_file_processor.get_hdf5_file_time_from_filename(fn)
        )
        WAIT_HDF5_FILE_AGE_HOUR = 24
        cutoff = time.time() - WAIT_HDF5_FILE_AGE_HOUR * 3600
        hdf5_files = [
            fn for fn in hdf5_files
            if os.path.getmtime(fn) <= cutoff
        ]

        files_to_process = [f for f in hdf5_files if self.hdf5_file_processor.get_hdf5_file_time_from_filename(f) > particle_skip_until]
        
        if not files_to_process:
            logger.info(f"No new HDF5 files to process for particle {particle_name}")
            return old_particle_history_df

        logger.info(f"Found {len(files_to_process)} new HDF5 files to process for particle {particle_name}: {files_to_process[0]} ... {files_to_process[-1]}")

        # 3. Prepare task arguments
        tasks = []
        for fpath in files_to_process:
            tasks.append((fpath, particle_name, simu_name))
            

        # 4. Parallel processing
        new_particle_dfs = []
        consecutive_missing_count = 0
        MISSING_THRESHOLD = 5  # Particle may be merged/ejected, stop searching after threshold

        ctx = multiprocessing.get_context('forkserver')
        with ctx.Pool(
            processes=self.config.processes_count, 
            maxtasksperchild=self.config.tasks_per_child
        ) as pool:
            # imap returns result in order of input
            iterator = pool.imap(self._process_single_hdf5_for_particle, tasks)
            tqdm_iterator = tqdm(iterator, total=len(tasks), desc=f"Tracking {particle_name} in {simu_name}")

            try:
                for particle_df in tqdm_iterator:
                    if particle_df is not None and not particle_df.empty:
                        new_particle_dfs.append(particle_df)
                        consecutive_missing_count = 0  # Reset counter
                    else:
                        consecutive_missing_count += 1
                    
                    if consecutive_missing_count >= MISSING_THRESHOLD:
                        logger.info(f"Particle {particle_name} missing for {consecutive_missing_count} consecutive HDF5 files. Stopping search early and starting merging results")
                        pool.terminate()  # Force terminate process pool
                        break
            except Exception as e:
                logger.warning(f"Process pool interrupted or error occurred: {e}")
        
        # 5. Merge and save
        if new_particle_dfs:
            new_particle_df_concat = pd.concat(new_particle_dfs, ignore_index=True)
            if not old_particle_history_df.empty:
                new_particle_history_df = pd.concat([old_particle_history_df, new_particle_df_concat], ignore_index=True)
            else:
                new_particle_history_df = new_particle_df_concat
            
            # Deduplicate and sort
            if 'TTOT' in new_particle_history_df.columns:
                new_particle_history_df = new_particle_history_df.sort_values('TTOT').drop_duplicates(subset=['TTOT'], keep='last').reset_index(drop=True)
            
            # Save aggregated result
            try:
                new_particle_history_df.to_feather(all_feather_path)
                logger.info(f"Updated aggregated cache for {particle_name} at {all_feather_path}")
            except Exception as e:
                logger.error(f"Failed to save aggregated cache: {e}")
            
            return new_particle_history_df
        else:
            return old_particle_history_df

    @log_time(logger)
    def update_all_particle_history_df(self, simu_name: str) -> None:
        """
        Get and save particle dataframes for all particles in the simulation
        
        Args: 
            simu_name: Name of the simulation
        """
        # 1. Get all HDF5 files
        hdf5_files = sorted(
            glob(self.config.pathof[simu_name] + '/**/*.h5part'), 
            key=lambda fn: self.hdf5_file_processor.get_hdf5_file_time_from_filename(fn)
        )
        if not hdf5_files:
            logger.error(f"No HDF5 files found for simulation {simu_name}")
            return
        
        # 2. Find the t=0 HDF5 file (first file)
        t0_hdf5_file = hdf5_files[0]
        
        # 3. Read t=0 HDF5 file and get all particle names
        try:
            df_dict = self.hdf5_file_processor.read_file(t0_hdf5_file, simu_name)
            single_df = df_dict['singles']
            
            if single_df.empty or 'Name' not in single_df.columns:
                logger.error(f"No particles found in initial HDF5 file {t0_hdf5_file}")
                return
            
            all_particle_names = single_df['Name'].unique()
            logger.info(f"Found {len(all_particle_names)} particles in simulation {simu_name}")
            
        except Exception as e:
            logger.error(f"Failed to read initial HDF5 file {t0_hdf5_file}: {type(e).__name__}: {e}")
            return
        
        # 4. Process each particle
        successful_count = 0
        failed_count = 0
        
        for particle_name in tqdm(all_particle_names, desc=f"Saving all particles in {simu_name}"):
            try:
                particle_history_df = None
                particle_history_df = self.update_one_particle_history_df(simu_name, particle_name, update=True)
                
                if not particle_history_df.empty:
                    successful_count += 1
                else:
                    logger.warning(f"Empty dataframe returned for particle {particle_name}")
                    failed_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to process particle {particle_name}: {type(e).__name__}: {e}")
                failed_count += 1
            
            finally:
                gc.collect()
        
        logger.info(f"Completed saving particles in {simu_name}: {successful_count} successful, {failed_count} failed")
    
    # Backward compatibility aliases
    def get_particle_df_all(self, simu_name: str, particle_name: int, 
                           update: bool = True) -> pd.DataFrame:
        """
        Backward compatibility alias for update_one_particle_history_df
        
        .. deprecated:: 0.2.0
            Use :meth:`update_one_particle_history_df` instead.
        """
        return self.update_one_particle_history_df(simu_name, particle_name, update)
    
    def save_every_particle_df_of_sim(self, simu_name: str) -> None:
        """
        Backward compatibility alias for update_all_particle_history_df
        
        .. deprecated:: 0.2.0
            Use :meth:`update_all_particle_history_df` instead.
        """
        return self.update_all_particle_history_df(simu_name)
