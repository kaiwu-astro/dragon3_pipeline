"""Particle tracking functionality for following individual particles through simulation"""

import logging
import multiprocessing
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Iterable

import numpy as np
import pandas as pd
from rich.progress import Progress, track
from glob import glob

try:
    import lz4
except ImportError:
    pass

from dragon3_pipelines.io import HDF5FileProcessor
from dragon3_pipelines.utils import log_time

logger = logging.getLogger(__name__)


@dataclass
class HDF5ParticleTask:
    """Task parameters for processing particles from an HDF5 file.

    Attributes:
        hdf5_file_path: Path to the HDF5 file
        simu_name: Simulation name
        particle_names: List of particle names to process
        progress_dict: Dictionary mapping particle_name -> last processed time
    """

    hdf5_file_path: str
    simu_name: str
    particle_names: List[int]
    progress_dict: Dict[int, float]


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
            hdf5_file_path: Path to HDF5 file (required when df_dict is None)
            simu_name: Simulation name (required when df_dict is None)

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
        # desc = (
        #     f"Getting particle df from {os.path.basename(hdf5_file_path)}"
        #     if hdf5_file_path is not None
        #     else "Getting particle df"
        # )
        # for pname in track(
        #     particle_names,
        #     description=desc,
        # ):
        for pname in particle_names:
            particle_df = self._get_one_particle_df(df_dict, int(pname))
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

        # 2. Build progress dict from individual particle history files
        progress_dict = self._build_progress_dict(simu_name, particle_names)
        min_progress_time = min(progress_dict.values()) if progress_dict else -1.0
        files_to_process = [
            f
            for f in hdf5_files
            if self.hdf5_file_processor.get_hdf5_file_time_from_filename(f) > min_progress_time
        ]

        if not files_to_process:
            logger.info(f"No new HDF5 files to process for simulation {simu_name}")
            logger.info(f"Min progress time across particles: {min_progress_time}")
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
            batch_size = max(1, int((mem_cap_bytes * 3 / 4) // per_file_bytes)) # 留1/4缓存particle_df，避免疯狂写盘

        batch_size = min(batch_size, len(files_to_process), self.config.processes_count)
        logger.info(
            f"Memory cap: {mem_cap_bytes / 1024**3:.2f} GB, "
            f"per-file estimate: {per_file_bytes / 1024**3:.4f} GB, "
            f"batch_size: {batch_size}"
        )

        # 4. Batch process HDF5 files
        in_mem_particle_dfs: Dict[int, list] = {}
        in_mem_time_start: Optional[float] = None
        last_batch_end: Optional[float] = None

        for start in track(
            range(0, len(files_to_process), batch_size),
            description=f"Processing all HDF5 batches {simu_name}",
        ):
            batch_files = files_to_process[start : start + batch_size]
            batch_particle_dfs: Dict[int, list] = {}

            ctx = multiprocessing.get_context("forkserver")
            tasks = [
                HDF5ParticleTask(
                    hdf5_file_path=fpath,
                    simu_name=simu_name,
                    particle_names=particle_names,
                    progress_dict=progress_dict,
                )
                for fpath in batch_files
            ]

            with ctx.Pool(
                processes=min(batch_size, self.config.processes_count),
                maxtasksperchild=self.config.tasks_per_child,
            ) as pool:
                iterator = pool.imap(self._process_one_hdf5_file_for_particles_wrapper_mp, tasks)
                with Progress() as progress:
                    task_id = progress.add_task(
                        f"Processing HDF5 batch number {start // batch_size + 1} in {simu_name}",
                        total=len(tasks),
                    )
                    for _, result_dict in iterator:
                        progress.advance(task_id)
                        if not result_dict:
                            continue
                        for pname, pdf in result_dict.items():
                            if pdf is not None and not pdf.empty:
                                batch_particle_dfs.setdefault(int(pname), []).append(pdf)

            # 5. Accumulate and persist per particle (on condition)
            # Get time range for this batch
            t_batch_start = self.hdf5_file_processor.get_hdf5_file_time_from_filename(
                batch_files[0]
            )
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
            else:  # write to file (mem threshold exceeded)
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
                cache_file_count_0 = len(
                    glob(os.path.join(particle_dir_0, f"{particle_names[0]}_df_*.df.feather"))
                )

                if tasks:
                    # 并行，不知为何容易出问题，合并卡住不动
                    # with ctx.Pool(
                    #     processes=self.config.processes_count,
                    #     maxtasksperchild=self.config.tasks_per_child,
                    # ) as pool:
                    #     iterator = pool.imap(self._accumulate_particle_df_wrapper_mp, tasks)
                    #     for _ in tqdm(
                    #         iterator,
                    #         total=len(tasks),
                    #         desc=f"Writing particle caches in {simu_name}" if cache_file_count_0 < n_cache_tol else
                    #         f"Accumulating particle caches in {simu_name}",
                    #     ):
                    #         pass
                    # 改为串行
                    iterator = map(self._accumulate_particle_df_wrapper_mp, tasks)
                    desc = (
                        f"Writing particle caches in {simu_name}"
                        if cache_file_count_0 < n_cache_tol
                        else f"Accumulating particle caches in {simu_name}"
                    )
                    with Progress() as progress:
                        task_id = progress.add_task(desc, total=len(tasks))
                        for _ in iterator:
                            progress.advance(task_id)

                in_mem_particle_dfs = {}
                in_mem_time_start = None

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
                    1,  # n_cache_tol set to 1 to force merge at the end
                )
                for pname, dfs in in_mem_particle_dfs.items()
                if dfs
            ]

            if tasks:
                # with ctx.Pool(
                #     processes=self.config.processes_count,
                #     maxtasksperchild=self.config.tasks_per_child,
                # ) as pool:
                #     iterator = pool.imap(self._accumulate_particle_df_wrapper_mp, tasks)
                #     for _ in tqdm(
                #         iterator,
                #         total=len(tasks),
                #         desc=f"Finally accumulating particle caches in {simu_name}",
                #     ):
                #         pass
                # 同样改为串行
                iterator = map(self._accumulate_particle_df_wrapper_mp, tasks)
                with Progress() as progress:
                    task_id = progress.add_task(
                        f"Finally accumulating particle caches in {simu_name}", total=len(tasks)
                    )
                    for _ in iterator:
                        progress.advance(task_id)

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

            try:
                with Progress() as progress:
                    task_id = progress.add_task(
                        f"Tracking {particle_name} in {simu_name}", total=len(tasks)
                    )
                    for particle_df in iterator:
                        progress.advance(task_id)
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

    def _build_progress_dict(self, simu_name: str, particle_names: List[int]) -> Dict[int, float]:
        """
        Build a progress dictionary by scanning each particle's history_until files.

        For each particle, looks for {particle_name}_history_until_*.df.feather files
        and extracts the timestamp from the filename.

        Args:
            simu_name: Simulation name
            particle_names: List of particle names to check

        Returns:
            Dictionary mapping particle_name -> last processed time.
            Returns -1.0 for particles with no history file or missing directory.
        """
        cache_base = self.config.particle_df_cache_dir_of[simu_name]
        progress_dict: Dict[int, float] = {}

        # Regex pattern for extracting timestamp from filename
        import re

        timestamp_pattern = re.compile(r"_history_until_([0-9.]+)\.df\.feather$")

        for pname in particle_names:
            particle_dir = os.path.join(cache_base, str(pname))

            if not os.path.exists(particle_dir):
                progress_dict[pname] = -1.0
                continue

            # Find all history_until files for this particle using os.listdir
            prefix = f"{pname}_history_until_"
            suffix = ".df.feather"
            try:
                files_in_dir = os.listdir(particle_dir)
            except OSError as e:
                logger.warning(f"Failed to list directory {particle_dir}: {e}")
                progress_dict[pname] = -1.0
                continue

            until_files = [f for f in files_in_dir if f.startswith(prefix) and f.endswith(suffix)]

            if not until_files:
                progress_dict[pname] = -1.0
                continue

            # Parse timestamps from filenames using regex
            timestamps = []
            for filename in until_files:
                match = timestamp_pattern.search(filename)
                if match:
                    try:
                        timestamps.append(float(match.group(1)))
                    except ValueError as e:
                        logger.warning(f"Failed to parse timestamp from {filename}: {e}")
                else:
                    logger.warning(f"Failed to match timestamp pattern in {filename}")

            if not timestamps:
                progress_dict[pname] = -1.0
                continue

            max_timestamp = max(timestamps)

            # Warn if multiple until files exist
            if len(timestamps) > 1:
                logger.warning(
                    f"Particle {pname} has {len(timestamps)} history_until files. "
                    f"Using max timestamp: {max_timestamp:.2f}"
                )

            progress_dict[pname] = max_timestamp

        return progress_dict

    def _process_one_dfdict_for_particle_wrapper_mp(
        self, args: Tuple[str, int, str]
    ) -> pd.DataFrame:
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
        use_miltithread: bool = True,
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
            use_miltithread: Whether to use multithreading when reading feather files
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

        # 保险：进程被杀可能导致一些文件已经合并，一些没有
        # 若已有 history_until 文件的 until 时间戳 >= 当前片段 t_start，则视为已合并，清理片段文件后返回
        if merged_cache_files:
            latest_merged = merged_cache_files[0]
            base = os.path.basename(latest_merged)
            # {particle_name}_history_until_{max_ttot:.2f}.df.feather
            until_str = base.split("_history_until_")[1].split(".df.feather")[0]
            latest_until = float(until_str)

            if latest_until is not None and latest_until >= t_start:
                for cache_file in individual_cache_files:
                    try:
                        os.remove(cache_file)
                    except Exception as e:
                        logger.warning(f"Failed to delete cache file {cache_file}: {e}")
                return

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
                    df = pd.read_feather(cache_file, use_threads=use_miltithread)
                    if not df.empty:
                        dfs_to_merge.append(df)
                except Exception as e:
                    logger.warning(f"Failed to read cache file {cache_file}: {e}")

            # Read existing merged file if present
            if merged_cache_files:
                try:
                    old_merged_df = pd.read_feather(
                        merged_cache_files[0], use_threads=use_miltithread
                    )
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
            simu_name,
            particle_name,
            new_particle_df,
            t_start,
            t_end,
            n_cache_tol,
            use_miltithread=False,
        )  # mp时手动指定关闭read_feather的多线程，避免多进程+多线程导致爆线程数

    def _process_one_hdf5_file_for_particles_wrapper_mp(
        self, task: HDF5ParticleTask
    ) -> Tuple[str, Dict[int, pd.DataFrame]]:
        """
        Worker function for parallel processing of particles from an HDF5 file.

        Filters out particles that have already been processed beyond this HDF5 file's time.

        Args:
            task: HDF5ParticleTask containing file path, simulation name,
                  particle names, and progress dict

        Returns:
            Tuple of (hdf5_file_path, dict mapping particle_name -> DataFrame)
        """
        hdf5_file_path = task.hdf5_file_path
        simu_name = task.simu_name
        particle_names = task.particle_names
        progress_dict = task.progress_dict

        try:
            # Get the time of this HDF5 file
            hdf5_time = self.hdf5_file_processor.get_hdf5_file_time_from_filename(hdf5_file_path)

            # Filter particles: skip those already processed beyond this HDF5 file's time
            particles_to_process = [
                pname for pname in particle_names if hdf5_time > progress_dict.get(pname, -1.0)
            ]

            if not particles_to_process:
                return hdf5_file_path, {}

            df_dict = self.hdf5_file_processor.read_file(hdf5_file_path, simu_name)
            result_dict = self.get_particle_df_from_hdf5_file(
                df_dict,
                particle_name=particles_to_process,
                hdf5_file_path=hdf5_file_path,
                simu_name=simu_name,
            )
            return hdf5_file_path, result_dict
        except Exception as e:
            logger.error(f"Failed to process HDF5 file {hdf5_file_path}: {type(e).__name__}: {e}")
            return hdf5_file_path, {}
