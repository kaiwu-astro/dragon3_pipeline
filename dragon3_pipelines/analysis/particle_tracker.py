"""Particle tracking functionality for following individual particles through simulation"""

import gc
import logging
import multiprocessing
import os
from typing import Any, Dict, Optional, Tuple, Union, Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from glob import glob

try:
    import lz4
except ImportError:
    pass

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
        df_dict: Optional[Dict[str, pd.DataFrame]] = None,
        particle_name: Union[int, Iterable[int]] = "all",
        hdf5_file_path: Optional[str] = None,
        simu_name: Optional[str] = None,
        save_cache: bool = False,
    ) -> Union[pd.DataFrame, Dict[int, pd.DataFrame]]:
        """
        Track particle(s) evolution through snapshots in an HDF5 file

        Args:
            df_dict: Dictionary containing 'singles', 'binaries', 'scalars' DataFrames
                    (obtained from HDF5FileProcessor.read_file)
            particle_name: Particle Name to track (int) or list-like of ints
            hdf5_file_path: Path to HDF5 file (required when df_dict is None or save_cache=True)
            simu_name: Simulation name (required when df_dict is None or save_cache=True)
            save_cache: If True, save individual particle DataFrames to cache files

        Returns:
            If particle_name is int: DataFrame containing all time points for this particle
            If particle_name is list-like: Dict mapping particle_name -> DataFrame
        """
        if isinstance(particle_name, (int, np.integer)):
            assert df_dict is not None, "df_dict is required for single-particle mode"
            return self._get_single_particle_df(df_dict, int(particle_name))

        if df_dict is None:
            if hdf5_file_path is None or simu_name is None:
                raise ValueError("hdf5_file_path and simu_name are required when df_dict is None")
            df_dict = self.hdf5_file_processor.read_file(hdf5_file_path, simu_name)

        if save_cache and (hdf5_file_path is None or simu_name is None):
            raise ValueError("hdf5_file_path and simu_name are required when save_cache=True")

        try:
            particle_names = [int(p) for p in particle_name]  # list-like
        except TypeError:
            raise ValueError("particle_name must be int or list-like of ints")

        if not particle_names:
            return {}

        result_dict = {}
        for pname in tqdm(
            particle_names,
            desc=(
                f"Writing cache for particles in {os.path.basename(hdf5_file_path)}"
            ),
        ):
            particle_df = self._get_single_particle_df(df_dict, int(pname))
            if save_cache and not particle_df.empty:
                assert hdf5_file_path is not None
                assert simu_name is not None
                self._save_particle_cache(particle_df, int(pname), hdf5_file_path, simu_name)
            result_dict[int(pname)] = particle_df

        return result_dict

    def _get_single_particle_df(
        self, df_dict: Dict[str, pd.DataFrame], particle_name: int
    ) -> pd.DataFrame:
        """
        Extract a single particle's data from df_dict

        Args:
            df_dict: Dictionary containing 'singles', 'binaries', 'scalars' DataFrames
            particle_name: Particle Name to track

        Returns:
            DataFrame containing all time points for this particle in the HDF5 file
        """
        single_df_all = df_dict["singles"]
        binary_df_all = df_dict["binaries"]

        # Extract records for this particle from single star data
        single_particle_df = single_df_all[single_df_all["Name"] == particle_name].copy()

        # Extract records from binary data
        # Check if particle appears as member 1 or member 2 of a binary
        # Note: this is necessary due to a star may sometimes be Bin 1 and sometimes Bin 2
        #       even in the same hdf5 file
        binary_as_1 = binary_df_all[binary_df_all["Bin Name1"] == particle_name].copy()
        binary_as_1["state"] = "binary"
        binary_as_1["companion_name"] = binary_as_1["Bin Name2"]
        binary_as_2 = binary_df_all[binary_df_all["Bin Name2"] == particle_name].copy()
        binary_as_2["state"] = "binary"
        binary_as_2["companion_name"] = binary_as_2["Bin Name1"]

        _binary = pd.concat([binary_as_1, binary_as_2], ignore_index=True)

        if not _binary["TTOT"].is_unique:
            logger.warning(
                f"Warning: Particle {particle_name} is found in both components at TTOT = {_binary['TTOT'][_binary['TTOT'].duplicated()].unique()}"
            )
            _binary = _binary.drop_duplicates(subset=["TTOT"], keep="first")

        # Merge all records
        if binary_as_1.empty and binary_as_2.empty:
            single_particle_df["state"] = "single"
            single_particle_df["companion_name"] = np.nan
            particle_df = single_particle_df
        else:
            # Particle is in a binary
            particle_df = single_particle_df.merge(
                _binary,
                on="TTOT",
                how="outer",
                suffixes=(
                    "",
                    "_from_binary",
                ),  # Columns with same name from single vs binary get suffix
            )

        if not particle_df.empty:
            particle_df = particle_df.sort_values("TTOT").reset_index(drop=True)
            return particle_df
        else:
            logger.debug(
                f"Particle {particle_name} not found at any of TTOT: {df_dict['scalars']['TTOT'].unique()}"
            )
            return pd.DataFrame()

    def _save_particle_cache(
        self, particle_df: pd.DataFrame, particle_name: int, hdf5_file_path: str, simu_name: str
    ) -> None:
        """
        Save particle DataFrame to cache file (feather)

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

    def _read_progress_file(self, simu_name: str) -> float:
        """
        Read the progress file to get the last processed HDF5 time

        Args:
            simu_name: Simulation name

        Returns:
            Last processed HDF5 time (or -1.0 if no progress file exists)
        """
        cache_base = self.config.particle_df_cache_dir_of[simu_name]
        progress_file = os.path.join(cache_base, "_progress.txt")

        if not os.path.exists(progress_file):
            return -1.0

        try:
            with open(progress_file, "r") as f:
                content = f.read().strip()
                if content:
                    return float(content)
        except Exception as e:
            logger.warning(f"Failed to read progress file {progress_file}: {e}")

        return -1.0

    def _write_progress_file(self, simu_name: str, hdf5_time: float) -> None:
        """
        Write the progress file with the last processed HDF5 time

        Args:
            simu_name: Simulation name
            hdf5_time: HDF5 file time to record
        """
        cache_base = self.config.particle_df_cache_dir_of[simu_name]
        os.makedirs(cache_base, exist_ok=True)
        progress_file = os.path.join(cache_base, "_progress.txt")

        try:
            with open(progress_file, "w") as f:
                f.write(f"{hdf5_time}\n")
            logger.debug(f"Updated progress file to {hdf5_time}")
        except Exception as e:
            logger.warning(f"Failed to write progress file {progress_file}: {e}")

    def _merge_and_cleanup_particle_cache(self, simu_name: str) -> None:
        """
        Merge temporary particle cache files into consolidated history files

        This function:
        1. Iterates through all particle subdirectories
        2. Reads all temporary feather files for each particle
        3. Merges them into a single {particle_name}_history_until_{max_ttot:.2f}.df.feather file
        4. Deletes old merged files and temporary files

        Args:
            simu_name: Simulation name
        """
        cache_base = self.config.particle_df_cache_dir_of[simu_name]

        if not os.path.exists(cache_base):
            logger.warning(f"Cache directory does not exist: {cache_base}")
            return

        # Find all particle subdirectories
        particle_dirs = [
            d
            for d in os.listdir(cache_base)
            if os.path.isdir(os.path.join(cache_base, d)) and d.isdigit()
        ]

        if not particle_dirs:
            logger.info("No particle subdirectories found to merge")
            return

        logger.info(f"Merging cache for {len(particle_dirs)} particles in {simu_name}")

        for particle_name_str in tqdm(particle_dirs, desc="Merging particle caches"):
            particle_name = int(particle_name_str)
            particle_dir = os.path.join(cache_base, particle_name_str)

            try:
                # Find all temporary feather files (format: {time}.{particle_name}.df.feather)
                temp_files = sorted(glob(os.path.join(particle_dir, "*.df.feather")))

                # Skip already-merged history files
                temp_files = [
                    f
                    for f in temp_files
                    if not os.path.basename(f).startswith(f"{particle_name}_history_until_")
                ]

                if not temp_files:
                    logger.debug(f"No temporary files to merge for particle {particle_name}")
                    continue

                # Read all temporary files
                particle_dfs = []
                for temp_file in temp_files:
                    try:
                        df = pd.read_feather(temp_file)
                        if not df.empty:
                            particle_dfs.append(df)
                    except Exception as e:
                        logger.warning(f"Failed to read {temp_file}: {e}")

                if not particle_dfs:
                    logger.warning(f"No valid data found for particle {particle_name}")
                    continue

                # Merge all DataFrames
                merged_df = pd.concat(particle_dfs, ignore_index=True)

                # Sort and deduplicate by TTOT
                if "TTOT" in merged_df.columns:
                    merged_df = (
                        merged_df.sort_values("TTOT")
                        .drop_duplicates(subset=["TTOT"], keep="last")
                        .reset_index(drop=True)
                    )
                    max_ttot = merged_df["TTOT"].max()
                else:
                    logger.warning(f"No TTOT column found for particle {particle_name}")
                    continue

                # Check if there's an existing merged file
                old_merged_files = glob(
                    os.path.join(particle_dir, f"{particle_name}_history_until_*.df.feather")
                )

                # If there's an old merged file, read it and merge with new data
                if old_merged_files:
                    for old_file in old_merged_files:
                        try:
                            old_df = pd.read_feather(old_file)
                            if not old_df.empty:
                                merged_df = pd.concat([old_df, merged_df], ignore_index=True)
                                if "TTOT" in merged_df.columns:
                                    merged_df = (
                                        merged_df.sort_values("TTOT")
                                        .drop_duplicates(subset=["TTOT"], keep="last")
                                        .reset_index(drop=True)
                                    )
                                    max_ttot = merged_df["TTOT"].max()
                        except Exception as e:
                            logger.warning(f"Failed to read old merged file {old_file}: {e}")

                # Save new merged file
                new_merged_file = os.path.join(
                    particle_dir, f"{particle_name}_history_until_{max_ttot:.2f}.df.feather"
                )
                merged_df.to_feather(new_merged_file)
                logger.debug(
                    f"Saved merged history for particle {particle_name} to {new_merged_file}"
                )

                # Delete old merged files (with different max_ttot)
                for old_file in old_merged_files:
                    if old_file != new_merged_file:
                        try:
                            os.remove(old_file)
                            logger.debug(f"Deleted old merged file: {old_file}")
                        except Exception as e:
                            logger.warning(f"Failed to delete old merged file {old_file}: {e}")

                # Delete temporary files
                for temp_file in temp_files:
                    try:
                        os.remove(temp_file)
                        logger.debug(f"Deleted temporary file: {temp_file}")
                    except Exception as e:
                        logger.warning(f"Failed to delete temporary file {temp_file}: {e}")

            except Exception as e:
                logger.error(f"Failed to merge cache for particle {particle_name}: {e}")

        logger.info(f"Completed merging particle caches for {simu_name}")

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
            "particle_name": (
                particle_history_df["Name"].iloc[0]
                if "Name" in particle_history_df.columns
                else None
            ),
            "total_snapshots": len(particle_history_df),
            "time_range_myr": (
                particle_history_df["Time[Myr]"].min(),
                particle_history_df["Time[Myr]"].max(),
            ),
            "single_count": len(particle_history_df[particle_history_df["state"] == "single"]),
            "binary_count": len(
                particle_history_df[particle_history_df["state"].str.contains("binary", na=False)]
            ),
            "initial_mass": (
                particle_history_df["M"].iloc[0] if "M" in particle_history_df.columns else None
            ),
            "final_mass": (
                particle_history_df["M"].iloc[-1] if "M" in particle_history_df.columns else None
            ),
            "stellar_types": (
                particle_history_df["KW"].unique().tolist()
                if "KW" in particle_history_df.columns
                else []
            ),
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
            logger.error(
                f"Error processing {hdf5_file_path} for particle {particle_name}: {type(e).__name__}: {e}"
            )
            return pd.DataFrame()

    @log_time(logger)
    def update_one_particle_history_df(
        self, simu_name: str, particle_name: int, update: bool = True
    ) -> pd.DataFrame:
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

        # Try to read from new merged cache format first
        particle_dir = os.path.join(cache_base, str(particle_name))
        old_particle_history_df = pd.DataFrame()
        particle_skip_until = -1.0

        # 1a. Check for new merged cache format: {particle_name}_history_until_*.df.feather
        merged_cache_files = []
        if os.path.exists(particle_dir):
            merged_cache_files = sorted(
                glob(os.path.join(particle_dir, f"{particle_name}_history_until_*.df.feather")),
                reverse=True,  # Get the latest one first
            )

        if merged_cache_files:
            # Use the latest merged cache file
            merged_cache_path = merged_cache_files[0]
            try:
                old_particle_history_df = pd.read_feather(merged_cache_path)
                if not old_particle_history_df.empty and "TTOT" in old_particle_history_df.columns:
                    particle_skip_until = old_particle_history_df["TTOT"].max()
                    logger.info(
                        f"Loaded merged cache for particle {particle_name} from {merged_cache_path}, records: {len(old_particle_history_df)}, max TTOT: {particle_skip_until}"
                    )
            except Exception as e:
                logger.warning(f"Failed to read merged cache {merged_cache_path}: {e}")

        if not update:
            return old_particle_history_df

        # 2. Get and filter file list
        hdf5_files = self.hdf5_file_processor.get_all_hdf5_paths(simu_name)

        files_to_process = [
            f
            for f in hdf5_files
            if self.hdf5_file_processor.get_hdf5_file_time_from_filename(f) > particle_skip_until
        ]

        if not files_to_process:
            logger.info(f"No new HDF5 files to process for particle {particle_name}")
            return old_particle_history_df

        logger.info(
            f"Found {len(files_to_process)} new HDF5 files to process for particle {particle_name}: {files_to_process[0]} ... {files_to_process[-1]}"
        )

        # 3. Prepare task arguments
        tasks = []
        for fpath in files_to_process:
            tasks.append((fpath, particle_name, simu_name))

        # 4. Parallel processing
        new_particle_dfs = []
        consecutive_missing_count = 0
        MISSING_THRESHOLD = 5  # Particle may be merged/ejected, stop searching after threshold

        ctx = multiprocessing.get_context("forkserver")
        with ctx.Pool(
            processes=self.config.processes_count, maxtasksperchild=self.config.tasks_per_child
        ) as pool:
            # imap returns result in order of input
            iterator = pool.imap(self._process_single_hdf5_for_particle, tasks)
            tqdm_iterator = tqdm(
                iterator, total=len(tasks), desc=f"Tracking {particle_name} in {simu_name}"
            )

            try:
                for particle_df in tqdm_iterator:
                    if particle_df is not None and not particle_df.empty:
                        new_particle_dfs.append(particle_df)
                        consecutive_missing_count = 0  # Reset counter
                    else:
                        consecutive_missing_count += 1

                    if consecutive_missing_count >= MISSING_THRESHOLD:
                        logger.info(
                            f"Particle {particle_name} missing for {consecutive_missing_count} consecutive HDF5 files. Stopping search early and starting merging results"
                        )
                        pool.terminate()  # Force terminate process pool
                        break
            except Exception as e:
                logger.warning(f"Process pool interrupted or error occurred: {e}")

        # 5. Merge and save
        if new_particle_dfs:
            new_particle_df_concat = pd.concat(new_particle_dfs, ignore_index=True)
            if not old_particle_history_df.empty:
                new_particle_history_df = pd.concat(
                    [old_particle_history_df, new_particle_df_concat], ignore_index=True
                )
            else:
                new_particle_history_df = new_particle_df_concat

            # Deduplicate and sort
            if "TTOT" in new_particle_history_df.columns:
                new_particle_history_df = (
                    new_particle_history_df.sort_values("TTOT")
                    .drop_duplicates(subset=["TTOT"], keep="last")
                    .reset_index(drop=True)
                )
                max_ttot = new_particle_history_df["TTOT"].max()
            else:
                max_ttot = 0.0

            # Save to new merged cache format in particle subdirectory
            os.makedirs(particle_dir, exist_ok=True)
            merged_cache_path = os.path.join(
                particle_dir, f"{particle_name}_history_until_{max_ttot:.2f}.df.feather"
            )

            try:
                new_particle_history_df.to_feather(merged_cache_path)
                logger.info(f"Updated merged cache for {particle_name} at {merged_cache_path}")

                # Clean up old merged files with different max_ttot
                old_merged_files = glob(
                    os.path.join(particle_dir, f"{particle_name}_history_until_*.df.feather")
                )
                for old_file in old_merged_files:
                    if old_file != merged_cache_path:
                        try:
                            os.remove(old_file)
                            logger.debug(f"Deleted old merged cache: {old_file}")
                        except Exception as e:
                            logger.warning(f"Failed to delete old merged cache {old_file}: {e}")

            except Exception as e:
                logger.error(f"Failed to save merged cache: {e}")

            return new_particle_history_df
        else:
            return old_particle_history_df

    @log_time(logger)
    def update_multiple_particle_history_df(
        self, simu_name: str, particle_names: Iterable[int], merge_interval: int = 3
    ) -> None:
        """
        Process specified particles in the simulation using HDF5-centric iteration

        This method iterates through HDF5 files for efficiency:
        1. Reads the progress file to skip already-processed files
        2. For each unprocessed HDF5 file, extracts data for specified particles
        3. Saves individual particle cache files
        4. Periodically merges temporary cache files to avoid inode limits
        5. Updates progress file after each HDF5 file

        Args:
            simu_name: Name of the simulation
            particle_names: List-like of particle names (ints)
            merge_interval: Number of HDF5 files to process before merging caches (default: 10)
        """
        particle_names = [int(p) for p in particle_names]
        if not particle_names:
            logger.info("No particle_names provided, nothing to process")
            return

        # 1. Get all HDF5 files
        hdf5_files = self.hdf5_file_processor.get_all_hdf5_paths(simu_name, wait_age_hour=24)

        # 2. Read progress file to skip already-processed files
        last_processed_time = self._read_progress_file(simu_name)
        files_to_process = [
            f
            for f in hdf5_files
            if self.hdf5_file_processor.get_hdf5_file_time_from_filename(f) > last_processed_time
        ]

        if not files_to_process:
            logger.info(f"No new HDF5 files to process for simulation {simu_name}")
            logger.info(f"Last processed time: {last_processed_time}")
            return

        logger.info(
            f"Found {len(files_to_process)} HDF5 files to process for simulation {simu_name}"
        )
        logger.info(f"Processing files from {files_to_process[0]} to {files_to_process[-1]}")

        # 3. Process HDF5 files one by one
        files_processed_since_merge = 0

        for i, hdf5_file_path in enumerate(
            tqdm(files_to_process, desc=f"Processing HDF5 files in {simu_name}")
        ):
            try:
                # Read HDF5 file
                df_dict = self.hdf5_file_processor.read_file(hdf5_file_path, simu_name)

                _ = self.get_particle_df_from_hdf5_file(
                    df_dict,
                    particle_name=particle_names,
                    hdf5_file_path=hdf5_file_path,
                    simu_name=simu_name,
                    save_cache=True,
                )

                # Update progress file
                hdf5_time = self.hdf5_file_processor.get_hdf5_file_time_from_filename(
                    hdf5_file_path
                )
                self._write_progress_file(simu_name, hdf5_time)

                files_processed_since_merge += 1

                # Periodically merge caches to avoid too many small files
                if files_processed_since_merge >= merge_interval:
                    logger.info(f"Processed {files_processed_since_merge} files, starting merge...")
                    self._merge_and_cleanup_particle_cache(simu_name)
                    files_processed_since_merge = 0
                    gc.collect()

            except Exception as e:
                logger.error(
                    f"Failed to process HDF5 file {hdf5_file_path}: {type(e).__name__}: {e}"
                )
                continue

        # Final merge of any remaining temporary files
        if files_processed_since_merge > 0:
            logger.info("Performing final merge of particle caches...")
            self._merge_and_cleanup_particle_cache(simu_name)

        logger.info(f"Completed processing all HDF5 files for simulation {simu_name}")

    # Backward compatibility aliases
    def get_particle_df_all(
        self, simu_name: str, particle_name: int, update: bool = True
    ) -> pd.DataFrame:
        """
        Backward compatibility alias for update_one_particle_history_df

        .. deprecated:: 0.2.0
            Use :meth:`update_one_particle_history_df` instead.
        """
        return self.update_one_particle_history_df(simu_name, particle_name, update)

    def save_every_particle_df_of_sim(self, simu_name: str, particle_names: Iterable[int]) -> None:
        """
        Backward compatibility alias for update_multiple_particle_history_df

        .. deprecated:: 0.2.0
            Use :meth:`update_multiple_particle_history_df` instead.
        """
        return self.update_multiple_particle_history_df(simu_name, particle_names)
