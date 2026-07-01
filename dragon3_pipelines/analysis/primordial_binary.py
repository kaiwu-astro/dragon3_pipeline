"""Identify and cache primordial binaries from the first HDF5 output."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from dragon3_pipelines.analysis.cache_paths import PRIMORDIAL_BINARY_FEATURE, analysis_cache_dir
from dragon3_pipelines.io import HDF5FileProcessor

logger = logging.getLogger(__name__)


class PrimordialBinaryIdentifier:
    """Load primordial binaries from the strict ``TTOT == 0.0`` binary snapshot."""

    SCHEMA_VERSION = 1
    REQUIRED_BINARY_COLUMNS = {"Bin Name1", "Bin Name2", "TTOT"}
    TTOT_RULE = "binaries['TTOT'].astype(float) == 0.0"

    def __init__(self, config_manager: Any) -> None:
        self.config = config_manager
        self.hdf5_file_processor = HDF5FileProcessor(config_manager)

    def load_primordial_binaries(
        self,
        simu_name: str,
        *,
        update: bool = True,
        wait_age_hour: int | float | None = None,
        use_hdf5_cache: bool | None = None,
        exclude_bad_dirname: bool = True,
    ) -> pd.DataFrame:
        """Return the full cached table of primordial binaries for one simulation."""
        if not update:
            return self._read_cache(simu_name)
        hdf5_config = getattr(self.config, "hdf5", {}) or {}
        file_selection = hdf5_config.get("file_selection", {})
        table_cache = hdf5_config.get("table_cache", {})
        if wait_age_hour is None:
            wait_age_hour = file_selection.get("wait_age_hour", 24)
        if use_hdf5_cache is None:
            use_hdf5_cache = table_cache.get("use_hdf5_cache", True)

        first_hdf5_path = self._first_hdf5_path(
            simu_name,
            wait_age_hour=wait_age_hour,
            exclude_bad_dirname=exclude_bad_dirname,
        )
        source_mtime = os.path.getmtime(first_hdf5_path)
        meta = self._read_meta(simu_name)
        if self._cache_is_fresh(simu_name, meta, first_hdf5_path, source_mtime):
            return self._read_cache(simu_name)

        df_dict = self.hdf5_file_processor.read_tables(
            first_hdf5_path,
            simu_name,
            tables=["scalars", "binaries"],
            columns_by_table={"scalars": ["TTOT"], "binaries": None},
            use_cache=use_hdf5_cache,
        )
        binaries = df_dict.get("binaries", pd.DataFrame())
        scalars = df_dict.get("scalars", pd.DataFrame())
        discovered_ttot_values = self._ttot_values(scalars)
        primordial = self._identify_primordial_binaries(binaries)
        self._write_cache_and_meta(
            simu_name,
            primordial,
            {
                "schema_version": self.SCHEMA_VERSION,
                "source_hdf5_path": first_hdf5_path,
                "source_mtime": source_mtime,
                "discovered_ttot_values": discovered_ttot_values,
                "row_count": int(len(primordial)),
                "ttot_rule": self.TTOT_RULE,
            },
        )
        return primordial

    def _first_hdf5_path(
        self,
        simu_name: str,
        *,
        wait_age_hour: int | float | None,
        exclude_bad_dirname: bool,
    ) -> str:
        kwargs: Dict[str, Any] = {
            "sample_every_nb_time": None,
            "exclude_bad_dirname": exclude_bad_dirname,
        }
        if wait_age_hour is not None:
            kwargs["wait_age_hour"] = wait_age_hour
        hdf5_paths = self.hdf5_file_processor.get_all_hdf5_paths(simu_name, **kwargs)
        if not hdf5_paths:
            raise ValueError(f"No HDF5 files found for simulation {simu_name!r}.")
        return str(hdf5_paths[0])

    def _identify_primordial_binaries(self, binaries: pd.DataFrame) -> pd.DataFrame:
        missing = self.REQUIRED_BINARY_COLUMNS.difference(binaries.columns)
        if missing:
            raise ValueError(
                "Binary table missing required columns for primordial binary identification: "
                + ", ".join(sorted(missing))
            )

        zero_snapshot = binaries.loc[binaries["TTOT"].astype(float) == 0.0].copy()
        if zero_snapshot.empty:
            raise ValueError("First HDF5 file has no binary snapshot with strict TTOT == 0.0.")

        try:
            name1 = zero_snapshot["Bin Name1"].astype(int)
            name2 = zero_snapshot["Bin Name2"].astype(int)
        except (TypeError, ValueError) as exc:
            raise ValueError("Binary name columns must be convertible to integer IDs.") from exc

        mask = (name1 - name2).abs() == 1
        primordial = zero_snapshot.loc[mask].copy()
        if primordial.empty:
            primordial["primordial_name_min"] = pd.Series(dtype="int64")
            primordial["primordial_name_max"] = pd.Series(dtype="int64")
            primordial["primordial_pair_key"] = pd.Series(dtype="object")
            primordial["is_primordial_binary"] = pd.Series(dtype="bool")
            return primordial.reset_index(drop=True)

        name_min = pd.concat([name1.loc[mask], name2.loc[mask]], axis=1).min(axis=1).astype(int)
        name_max = pd.concat([name1.loc[mask], name2.loc[mask]], axis=1).max(axis=1).astype(int)
        primordial["primordial_name_min"] = name_min.to_numpy()
        primordial["primordial_name_max"] = name_max.to_numpy()
        primordial["primordial_pair_key"] = [
            f"{min_id}-{max_id}" for min_id, max_id in zip(name_min, name_max)
        ]
        primordial["is_primordial_binary"] = True
        return primordial.reset_index(drop=True)

    def _cache_dir(self, simu_name: str) -> Path:
        return analysis_cache_dir(self.config, simu_name, PRIMORDIAL_BINARY_FEATURE)

    def _cache_path(self, simu_name: str) -> Path:
        return self._cache_dir(simu_name) / "primordial_binaries.feather"

    def _meta_path(self, simu_name: str) -> Path:
        return self._cache_dir(simu_name) / "primordial_binaries.meta.json"

    def _read_cache(self, simu_name: str) -> pd.DataFrame:
        cache_path = self._cache_path(simu_name)
        if not cache_path.exists():
            return pd.DataFrame()
        return pd.read_feather(cache_path)

    def _read_meta(self, simu_name: str) -> Dict[str, Any]:
        meta_path = self._meta_path(simu_name)
        if not meta_path.exists():
            return {}
        try:
            return json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read primordial binary metadata %s: %r", meta_path, exc)
            return {}

    def _cache_is_fresh(
        self,
        simu_name: str,
        meta: Dict[str, Any],
        first_hdf5_path: str,
        source_mtime: float,
    ) -> bool:
        return (
            self._cache_path(simu_name).exists()
            and meta.get("schema_version") == self.SCHEMA_VERSION
            and meta.get("source_hdf5_path") == first_hdf5_path
            and meta.get("source_mtime") == source_mtime
        )

    def _write_cache_and_meta(self, simu_name: str, df: pd.DataFrame, meta: Dict[str, Any]) -> None:
        cache_dir = self._cache_dir(simu_name)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._cache_path(simu_name)
        meta_path = self._meta_path(simu_name)
        tmp_cache_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        tmp_meta_path = meta_path.with_suffix(meta_path.suffix + ".tmp")

        df.to_feather(tmp_cache_path)
        os.replace(tmp_cache_path, cache_path)
        tmp_meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
        os.replace(tmp_meta_path, meta_path)

    def _ttot_values(self, scalars: pd.DataFrame) -> list[float]:
        if "TTOT" not in scalars.columns:
            return []
        return sorted(float(ttot) for ttot in pd.unique(scalars["TTOT"].astype(float)))
