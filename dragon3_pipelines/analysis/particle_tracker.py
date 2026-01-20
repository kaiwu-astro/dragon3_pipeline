"""Particle tracking functionality for following individual particles through simulation"""

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
        particle_name: Union[int, Iterable[int], str] = "all",
        hdf5_file_path: Optional[str] = None,
        simu_name: Optional[str] = None,
        save_cache: bool = False,
    ) -> Union[pd.DataFrame, Dict[int, pd.DataFrame]]:
        """
        Track particle(s) evolution through snapshots in an HDF5 file

        Args:
            df_dict: Dictionary containing 'singles', 'binaries', 'scalars' DataFrames
                    (obtained from HDF5FileProcessor.read_file)
            particle_name: Particle Name to track. Can be:
                - int: track a single particle
                - list-like of ints: track multiple specific particles
                - "all": track all particles found in df_dict
            hdf5_file_path: Path to HDF5 file (required when df_dict is None or save_cache=True)
            simu_name: Simulation name (required when df_dict is None or save_cache=True)
            save_cache: If True, save individual particle DataFrames to cache files

        Returns:
            If particle_name is int: DataFrame containing all time points for this particle
            If particle_name is "all" or list-like: Dict mapping particle_name -> DataFrame
        """
        if isinstance(particle_name, (int, np.integer)):
            assert df_dict is not None, "df_dict is required for single-particle mode"
            return self._get_one_particle_df(df_dict, int(particle_name))

        if df_dict is None:
            if hdf5_file_path is None or simu_name is None:
                raise ValueError("hdf5_file_path and simu_name are required when df_dict is None")
            df_dict = self.hdf5_file_processor.read_file(hdf5_file_path, simu_name)

        if save_cache and (hdf5_file_path is None or simu_name is None):
            raise ValueError("hdf5_file_path and simu_name are required when save_cache=True")

        # Handle special case: particle_name == "all" means process all particles
        if particle_name == "all":
            if "singles" not in df_dict or "Name" not in df_dict["singles"].columns:
                raise ValueError(
                    "df_dict must contain 'singles' DataFrame with 'Name' column when using particle_name='all'"
                )
            particle_names = [int(name) for name in df_dict["singles"]["Name"].unique()]
        else:
            try:
                particle_names = [int(p) for p in particle_name]  # list-like
            except TypeError:
                raise ValueError("particle_name must be int, list-like of ints, or 'all'")

        if not particle_names:
            return {}

        result_dict = {}
        for pname in tqdm(
            particle_names,
            desc=(
                f"Getting particle df from {os.path.basename(hdf5_file_path)}"
            ),
            leave=False
        ):
            particle_df = self._get_one_particle_df(df_dict, int(pname))
            if save_cache and not particle_df.empty:
                assert hdf5_file_path is not None
                assert simu_name is not None
                self._save_particle_cache(particle_df, int(pname), hdf5_file_path, simu_name)
            result_dict[int(pname)] = particle_df

        return result_dict

    @log_time(logger)
    def update_multiple_particle_history_df(
        self, simu_name: str, particle_names: Iterable[int], n_cache_tol: Optional[int] = None
    ) -> None:
        """
        Process specified particles in the simulation using HDF5-centric batch iteration.

        Args:
            simu_name: Name of the simulation
            particle_names: List-like of particle names (ints)
            n_cache_tol: Maximum number of cache files before triggering merge.
                        If None, calculates as inode_limit // len(particle_names) - 5
        """
        particle_names = [int(p) for p in particle_names]
        if not particle_names:
            logger.info("No particle_names provided, nothing to process")
            return

        # Calculate n_cache_tol if not provided
        if n_cache_tol is None:
            n_cache_tol = self.config.inode_limit // max(1, len(particle_names)) - 5
            logger.info(
                f"Calculated n_cache_tol = {n_cache_tol} from inode_limit and particle count"
            )

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

        # 3. Estimate batch size based on memory cap
        sample_path = files_to_process[0]
        try:
            sample_df_dict = self.hdf5_file_processor.read_file(sample_path, simu_name)
        except Exception as e:
            logger.error(f"Failed to read sample HDF5 file {sample_path}: {e}")
            return

        per_file_bytes = 0
        for df in sample_df_dict.values():
            if isinstance(df, pd.DataFrame):
                per_file_bytes += int(df.memory_usage(deep=True).sum())

        mem_cap_bytes = self.config.mem_cap_bytes
        if per_file_bytes <= 0:
            batch_size = 1
        else:
            batch_size = max(1, int(mem_cap_bytes // per_file_bytes))

        batch_size = min(batch_size, len(files_to_process))
        logger.info(
            f"Memory cap: {mem_cap_bytes / 1024**3:.2f} GB, "
            f"per-file estimate: {per_file_bytes / 1024**3:.4f} GB, "
            f"batch_size: {batch_size}"
        )

        # 4. Batch process HDF5 files
        in_mem_particle_dfs: Dict[int, list] = {}
        in_mem_time_start: Optional[float] = None
        last_batch_end: Optional[float] = None

        for start in tqdm(
            range(0, len(files_to_process), batch_size),
            desc=f"Processing all HDF5 batches {simu_name}",
        ):
            batch_files = files_to_process[start : start + batch_size]
            batch_particle_dfs: Dict[int, list] = {}

            ctx = multiprocessing.get_context("forkserver")
            tasks = [(fpath, simu_name, particle_names) for fpath in batch_files]

            with ctx.Pool(
                processes=min(batch_size, self.config.processes_count), maxtasksperchild=self.config.tasks_per_child
            ) as pool:
                iterator = pool.imap(self._process_one_hdf5_file_for_particles_wrapper_mp, tasks)
                for _, result_dict in tqdm(
                    iterator, total=len(tasks), desc=f"Processing HDF5 batch in {simu_name}"
                ):
                    if not result_dict:
                        continue
                    for pname, pdf in result_dict.items():
                        if pdf is not None and not pdf.empty:
                            batch_particle_dfs.setdefault(int(pname), []).append(pdf)

            # 5. Accumulate and persist per particle (on condition)
            # Get time range for this batch
            t_batch_start = self.hdf5_file_processor.get_hdf5_file_time_from_filename(batch_files[0])
            t_batch_end = self.hdf5_file_processor.get_hdf5_file_time_from_filename(batch_files[-1])
            last_batch_end = t_batch_end

            # Decide whether to keep in memory or accumulate
            def _estimate_in_mem_bytes(pdict: Dict[int, list]) -> int:
                for _, dfs in pdict.items():
                    if dfs:
                        return int(dfs[0].memory_usage(deep=True).sum()) * sum(
                            len(v) for v in pdict.values()
                        )
                return 0

            # Merge dict
            tentative_in_mem = {}
            if in_mem_particle_dfs:
                tentative_in_mem = {k: list(v) for k, v in in_mem_particle_dfs.items()}
            for pname, dfs in batch_particle_dfs.items():
                if dfs:
                    tentative_in_mem.setdefault(int(pname), []).extend(dfs)

            est_bytes = _estimate_in_mem_bytes(tentative_in_mem)
            if est_bytes <= self.config.mem_cap_bytes / 4:
                if not in_mem_particle_dfs:
                    in_mem_time_start = t_batch_start
                in_mem_particle_dfs = tentative_in_mem
                continue
            else: # write to file (mem threshold exceeded)
                t_start = in_mem_time_start if in_mem_time_start is not None else t_batch_start
                t_end = t_batch_end

                tasks = [
                    (
                        simu_name,
                        int(pname),
                        pd.concat(dfs, ignore_index=True),
                        t_start,
                        t_end,
                        n_cache_tol,
                    )
                    for pname, dfs in tentative_in_mem.items()
                    if dfs
                ]

                cache_base = self.config.particle_df_cache_dir_of[simu_name]
                particle_dir_0 = os.path.join(cache_base, str(particle_names[0]))
                os.makedirs(particle_dir_0, exist_ok=True)
                cache_file_count_0 = len(glob(
                    os.path.join(particle_dir_0, f"{particle_names[0]}_df_*.df.feather")
                ))

                if tasks:
                    with ctx.Pool(
                        processes=self.config.processes_count,
                        maxtasksperchild=self.config.tasks_per_child,
                    ) as pool:
                        iterator = pool.imap(self._accumulate_particle_df_wrapper_mp, tasks)
                        for _ in tqdm(
                            iterator,
                            total=len(tasks),
                            desc=f"Writing particle caches in {simu_name}" if cache_file_count_0 < n_cache_tol else
                            f"Accumulating particle caches in {simu_name}",
                        ):
                            pass

                in_mem_particle_dfs = {}
                in_mem_time_start = None
                # Update progress after accumulate
                self._write_progress_file(simu_name, t_end)

        # After looping over all hdf5s, flush remaining in-memory data
        if in_mem_particle_dfs and last_batch_end is not None:
            t_start = in_mem_time_start if in_mem_time_start is not None else last_batch_end
            t_end = last_batch_end

            tasks = [
                (
                    simu_name,
                    int(pname),
                    pd.concat(dfs, ignore_index=True),
                    t_start,
                    t_end,
                    1, # n_cache_tol set to 1 to force merge at the end
                )
                for pname, dfs in in_mem_particle_dfs.items()
                if dfs
            ]

            if tasks:
                with ctx.Pool(
                    processes=self.config.processes_count,
                    maxtasksperchild=self.config.tasks_per_child,
                ) as pool:
                    iterator = pool.imap(self._accumulate_particle_df_wrapper_mp, tasks)
                    for _ in tqdm(
                        iterator,
                        total=len(tasks),
                        desc=f"Accumulating particle caches in {simu_name}",
                    ):
                        pass

            self._write_progress_file(simu_name, t_end)

        logger.info(f"Completed processing all HDF5 files for simulation {simu_name}")

    @log_time(logger)
    def save_every_particle_history_of_sim(
        self,
        simu_name: str,
    ) -> None:
        """
        Call update_multiple_particle_history_df, process for all particles.
        CAUTION: this is likely to drain you disk INODE. Only use for small star clusters.
        Args:
            simu_name: Name of the simulation
        """
        # get all particle names from the first hdf5 file
        first_hdf5_path = self.hdf5_file_processor.get_all_hdf5_paths(simu_name)[0]
        df_dict = self.hdf5_file_processor.read_file(first_hdf5_path, simu_name)
        single_df_all = df_dict["singles"]
        particle_names = single_df_all["Name"].unique().tolist()
        return self.update_multiple_particle_history_df(simu_name, particle_names)

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
            iterator = pool.imap(self._process_one_dfdict_for_particle_wrapper_mp, tasks)
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

    def _get_one_particle_df(
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

    def _get_one_particle_df_wrapper_mp(
        self, args: Tuple[Dict[str, pd.DataFrame], int]
    ) -> Tuple[int, pd.DataFrame]:
        df_dict, particle_name = args
        return particle_name, self._get_one_particle_df(df_dict, particle_name)

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

    def _process_one_dfdict_for_particle_wrapper_mp(self, args: Tuple[str, int, str]) -> pd.DataFrame:
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

    def _accumulate_particle_df(
        self,
        simu_name: str,
        particle_name: int,
        new_particle_df: pd.DataFrame,
        t_start: float,
        t_end: float,
        n_cache_tol: int,
    ) -> None:
        """
        Accumulate particle DataFrame using inode-based merge strategy.

        If cache file count < n_cache_tol: write individual feather file
        If cache file count >= n_cache_tol: merge all files and cleanup

        Args:
            simu_name: Simulation name
            particle_name: Particle name
            new_particle_df: Newly accumulated DataFrame for this particle
            t_start: Start time of the batch being processed
            t_end: End time of the batch being processed
            n_cache_tol: Threshold for number of cache files before merging
        """
        if new_particle_df is None or new_particle_df.empty:
            return

        cache_base = self.config.particle_df_cache_dir_of[simu_name]
        particle_dir = os.path.join(cache_base, str(particle_name))
        os.makedirs(particle_dir, exist_ok=True)

        # Count existing cache files (exclude _history_until_ files)
        individual_cache_files = glob(
            os.path.join(particle_dir, f"{particle_name}_df_*.df.feather")
        )
        merged_cache_files = sorted(
            glob(os.path.join(particle_dir, f"{particle_name}_history_until_*.df.feather")),
            reverse=True,
        )

        cache_file_count = len(individual_cache_files)

        if cache_file_count < n_cache_tol:
            # Strategy 1: Just write a new feather file
            new_cache_file = os.path.join(
                particle_dir, f"{particle_name}_df_{t_start:.6f}_to_{t_end:.6f}.df.feather"
            )
            try:
                new_particle_df.to_feather(new_cache_file)
                logger.debug(
                    f"Saved individual cache for particle {particle_name}: {new_cache_file} "
                    f"(count: {cache_file_count + 1}/{n_cache_tol})"
                )
            except Exception as e:
                logger.error(f"Failed to save individual cache for particle {particle_name}: {e}")
        else:
            # Strategy 2: Merge all caches
            logger.info(
                f"Cache file count ({cache_file_count}) >= threshold ({n_cache_tol}) "
                f"for particle {particle_name}. Triggering merge."
            )

            # Collect all DataFrames to merge
            dfs_to_merge = [new_particle_df]

            # Read all individual cache files
            for cache_file in individual_cache_files:
                try:
                    df = pd.read_feather(cache_file)
                    if not df.empty:
                        dfs_to_merge.append(df)
                except Exception as e:
                    logger.warning(f"Failed to read cache file {cache_file}: {e}")

            # Read existing merged file if present
            if merged_cache_files:
                try:
                    old_merged_df = pd.read_feather(merged_cache_files[0])
                    if not old_merged_df.empty:
                        dfs_to_merge.append(old_merged_df)
                except Exception as e:
                    logger.warning(f"Failed to read merged cache {merged_cache_files[0]}: {e}")

            # Merge all DataFrames
            if dfs_to_merge:
                merged_df = pd.concat(dfs_to_merge, ignore_index=True)

                # Sort, deduplicate
                if "TTOT" in merged_df.columns:
                    merged_df = (
                        merged_df.sort_values("TTOT")
                        .drop_duplicates(subset=["TTOT"], keep="last")
                        .reset_index(drop=True)
                    )
                    max_ttot = merged_df["TTOT"].max()
                else:
                    max_ttot = 0.0

                # Save merged file
                new_merged_file = os.path.join(
                    particle_dir, f"{particle_name}_history_until_{max_ttot:.2f}.df.feather"
                )

                try:
                    merged_df.to_feather(new_merged_file)
                    logger.info(
                        f"Merged {len(dfs_to_merge)} caches for particle {particle_name} "
                        f"into {new_merged_file}"
                    )

                    # Clean up individual cache files
                    for cache_file in individual_cache_files:
                        try:
                            os.remove(cache_file)
                        except Exception as e:
                            logger.warning(f"Failed to delete cache file {cache_file}: {e}")

                    # Clean up old merged files
                    for old_file in merged_cache_files:
                        if old_file != new_merged_file:
                            try:
                                os.remove(old_file)
                            except Exception as e:
                                logger.warning(f"Failed to delete old merged cache {old_file}: {e}")

                except Exception as e:
                    logger.error(f"Failed to save merged cache for particle {particle_name}: {e}")

    def _accumulate_particle_df_wrapper_mp(
        self, args: Tuple[str, int, pd.DataFrame, float, float, int]
    ) -> None:
        simu_name, particle_name, new_particle_df, t_start, t_end, n_cache_tol = args
        self._accumulate_particle_df(
            simu_name, particle_name, new_particle_df, t_start, t_end, n_cache_tol
        )

    def _merge_one_particle_cache(self, args: Tuple[str, int]) -> Tuple[int, bool]:
        """
        Worker function to merge cache files for a single particle.

        Args:
            args: Tuple of (simu_name, particle_name)

        Returns:
            Tuple of (particle_name, success_flag)
        """
        simu_name, particle_name = args

        try:
            cache_base = self.config.particle_df_cache_dir_of[simu_name]
            particle_dir = os.path.join(cache_base, str(particle_name))

            if not os.path.exists(particle_dir):
                return particle_name, False

            # Get all individual cache files
            individual_cache_files = glob(
                os.path.join(particle_dir, f"{particle_name}_df_*.df.feather")
            )

            if not individual_cache_files:
                return particle_name, True  # Nothing to merge

            merged_cache_files = sorted(
                glob(os.path.join(particle_dir, f"{particle_name}_history_until_*.df.feather")),
                reverse=True,
            )

            # Collect all DataFrames to merge
            dfs_to_merge = []

            # Read all individual cache files
            for cache_file in individual_cache_files:
                try:
                    df = pd.read_feather(cache_file)
                    if not df.empty:
                        dfs_to_merge.append(df)
                except Exception as e:
                    logger.warning(f"Failed to read cache file {cache_file}: {e}")

            # Read existing merged file if present
            if merged_cache_files:
                try:
                    old_merged_df = pd.read_feather(merged_cache_files[0])
                    if not old_merged_df.empty:
                        dfs_to_merge.append(old_merged_df)
                except Exception as e:
                    logger.warning(f"Failed to read merged cache {merged_cache_files[0]}: {e}")

            if not dfs_to_merge:
                return particle_name, False

            # Merge all DataFrames
            merged_df = pd.concat(dfs_to_merge, ignore_index=True)

            # Sort, deduplicate
            if "TTOT" in merged_df.columns:
                merged_df = (
                    merged_df.sort_values("TTOT")
                    .drop_duplicates(subset=["TTOT"], keep="last")
                    .reset_index(drop=True)
                )
                max_ttot = merged_df["TTOT"].max()
            else:
                max_ttot = 0.0

            # Save merged file
            new_merged_file = os.path.join(
                particle_dir, f"{particle_name}_history_until_{max_ttot:.2f}.df.feather"
            )

            merged_df.to_feather(new_merged_file)
            logger.info(
                f"Merged {len(dfs_to_merge)} caches for particle {particle_name} "
                f"into {new_merged_file}"
            )

            # Clean up individual cache files
            for cache_file in individual_cache_files:
                try:
                    os.remove(cache_file)
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {cache_file}: {e}")

            # Clean up old merged files
            for old_file in merged_cache_files:
                if old_file != new_merged_file:
                    try:
                        os.remove(old_file)
                    except Exception as e:
                        logger.warning(f"Failed to delete old merged cache {old_file}: {e}")

            return particle_name, True

        except Exception as e:
            logger.error(f"Failed to merge cache for particle {particle_name}: {e}")
            return particle_name, False

    def _merge_particle_caches_memory_aware(
        self, simu_name: str, particle_names: Iterable[int]
    ) -> None:
        """
        Merge cache files for multiple particles with dynamic memory management.

        First processes one particle to estimate memory usage, then dynamically
        adjusts parallel process count to stay within memory limits.

        Args:
            simu_name: Simulation name
            particle_names: List of particle names to merge caches for
        """
        particle_names = [int(p) for p in particle_names]
        if not particle_names:
            logger.info("No particles to merge")
            return

        logger.info(f"Starting memory-aware cache merge for {len(particle_names)} particles")

        # Process first particle to estimate memory usage
        first_particle = particle_names[0]
        logger.info(f"Processing first particle {first_particle} to estimate memory usage")

        _, success = self._merge_one_particle_cache((simu_name, first_particle))

        if not success:
            logger.warning(f"Failed to process first particle {first_particle}, continuing anyway")
            per_particle_bytes = 1024**3  # Assume 1GB default
        else:
            # Estimate memory usage from the merged file
            cache_base = self.config.particle_df_cache_dir_of[simu_name]
            particle_dir = os.path.join(cache_base, str(first_particle))
            merged_files = glob(
                os.path.join(particle_dir, f"{first_particle}_history_until_*.df.feather")
            )

            if merged_files:
                try:
                    test_df = pd.read_feather(merged_files[0])
                    per_particle_bytes = int(test_df.memory_usage(deep=True).sum())
                    logger.info(
                        f"Estimated per-particle memory: {per_particle_bytes / 1024**3:.4f} GB"
                    )
                except Exception as e:
                    logger.warning(f"Failed to estimate memory usage: {e}")
                    per_particle_bytes = 1024**3
            else:
                per_particle_bytes = 1024**3

        # Calculate max parallel processes based on memory
        if per_particle_bytes > 0:
            max_parallel = max(1, self.config.mem_cap_bytes // per_particle_bytes)
        else:
            max_parallel = self.config.processes_count

        # Adjust processes to not exceed config limit
        processes = min(max_parallel, self.config.processes_count, len(particle_names) - 1)

        logger.info(
            f"Memory cap: {self.config.mem_cap_bytes / 1024**3:.2f} GB, "
            f"per-particle: {per_particle_bytes / 1024**3:.4f} GB, "
            f"using {processes} parallel processes"
        )

        # Process remaining particles
        remaining_particles = particle_names[1:]
        if remaining_particles:
            ctx = multiprocessing.get_context("forkserver")
            tasks = [(simu_name, pname) for pname in remaining_particles]

            with ctx.Pool(
                processes=processes, maxtasksperchild=self.config.tasks_per_child
            ) as pool:
                results = list(
                    tqdm(
                        pool.imap(self._merge_one_particle_cache, tasks),
                        total=len(tasks),
                        desc=f"Merging particle caches for {simu_name}",
                    )
                )

            success_count = sum(1 for _, success in results if success)
            logger.info(
                f"Completed merging: {success_count}/{len(remaining_particles)} particles succeeded"
            )

        logger.info(f"Memory-aware cache merge completed for {simu_name}")

    def _process_one_hdf5_file_for_particles_wrapper_mp(
        self, args: Tuple[str, str, Iterable[int]]
    ) -> Tuple[str, Dict[int, pd.DataFrame]]:
        hdf5_file_path, simu_name, particle_names = args
        try:
            df_dict = self.hdf5_file_processor.read_file(hdf5_file_path, simu_name)
            result_dict = self.get_particle_df_from_hdf5_file(
                df_dict,
                particle_name=particle_names,
                hdf5_file_path=hdf5_file_path,
                simu_name=simu_name,
                save_cache=False,
            )
            return hdf5_file_path, result_dict
        except Exception as e:
            logger.error(f"Failed to process HDF5 file {hdf5_file_path}: {type(e).__name__}: {e}")
            return hdf5_file_path, {}
