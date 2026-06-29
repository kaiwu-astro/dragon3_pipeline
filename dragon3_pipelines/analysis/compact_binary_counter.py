"""Cross-snapshot compact binary counting."""

from __future__ import annotations

import logging
from numbers import Number
from typing import Any, Dict, Hashable, Iterable, Tuple

import numpy as np
import pandas as pd

from dragon3_pipelines.io import HDF5FileProcessor

logger = logging.getLogger(__name__)


DETAIL_COLUMNS = [
    "binary_key",
    "bin_name1",
    "bin_name2",
    "first_ttot",
    "last_ttot",
    "first_time_myr",
    "last_time_myr",
    "first_kw1",
    "first_kw2",
    "last_kw1",
    "last_kw2",
    "first_stellar_type",
    "last_stellar_type",
    "n_snapshots_seen_in_category",
]


class CompactBinaryCounter:
    """Count unique compact binary systems seen across HDF5 snapshots."""

    CATEGORIES = ("gw_source", "pulsar", "xray_binary")
    NS_KW = 13
    BH_KW = 14
    WD_KW = frozenset({10, 11, 12})

    def __init__(self, config_manager: Any) -> None:
        self.config = config_manager
        self.hdf5_file_processor = HDF5FileProcessor(config_manager)
        self.compact_object_kw = frozenset(
            int(kw) for kw in getattr(config_manager, "compact_object_KW", [10, 11, 12, 13, 14])
        )

    def summarize_simulation(
        self,
        simu_name: str,
        *,
        sample_every_nb_time: float = 1.0,
        wait_age_hour: int = 24,
        exclude_bad_dirname: bool = True,
        use_hdf5_cache: bool = True,
    ) -> Dict[str, Any]:
        """Summarize unique compact binary systems seen from available snapshots."""
        hdf5_paths = self.hdf5_file_processor.get_all_hdf5_paths(
            simu_name,
            sample_every_nb_time=sample_every_nb_time,
            exclude_bad_dirname=exclude_bad_dirname,
            wait_age_hour=wait_age_hour,
        )
        records_by_category: Dict[str, Dict[Tuple[Hashable, Hashable], Dict[str, Any]]] = {
            category: {} for category in self.CATEGORIES
        }
        scanned_snapshots = 0
        max_ttot = np.nan
        max_time_myr = np.nan

        for hdf5_path in hdf5_paths:
            df_dict = self._read_counting_tables(hdf5_path, simu_name, use_hdf5_cache)
            used_counting_cache = set(df_dict.keys()) == {"scalars", "binaries"}
            ttot_values = self._snapshot_times(df_dict)

            if used_counting_cache:
                scanned_snapshots += len(ttot_values)
                if ttot_values:
                    max_ttot = self._nanmax(max_ttot, max(ttot_values))
                file_max_time = self._file_max_time_myr(df_dict)
                max_time_myr = self._nanmax(max_time_myr, file_max_time)
                binaries = df_dict.get("binaries", pd.DataFrame())
                if not binaries.empty and "TTOT" in binaries.columns:
                    for _, binary_df_at_t in binaries.groupby("TTOT", sort=True):
                        self._accumulate_snapshot(records_by_category, binary_df_at_t)
                continue

            for ttot in ttot_values:
                _, binary_df_at_t, is_valid = self.hdf5_file_processor.get_snapshot_at_t(
                    df_dict, ttot
                )
                if not is_valid or binary_df_at_t is None:
                    logger.warning(
                        "Skipping invalid compact binary snapshot %s TTOT=%s", hdf5_path, ttot
                    )
                    continue

                scanned_snapshots += 1
                max_ttot = self._nanmax(max_ttot, float(ttot))
                snapshot_max_time = self._snapshot_time_myr(df_dict, binary_df_at_t, ttot)
                max_time_myr = self._nanmax(max_time_myr, snapshot_max_time)

                if binary_df_at_t.empty:
                    continue
                self._accumulate_snapshot(records_by_category, binary_df_at_t)

        details = {
            category: self._records_to_dataframe(records_by_category[category])
            for category in self.CATEGORIES
        }
        summary = {category: len(details[category]) for category in self.CATEGORIES}
        summary.update(
            {
                "scanned_files": len(hdf5_paths),
                "scanned_snapshots": scanned_snapshots,
                "max_ttot": None if np.isnan(max_ttot) else max_ttot,
                "max_time_myr": None if np.isnan(max_time_myr) else max_time_myr,
            }
        )
        return {"summary": summary, "details": details}

    def _read_counting_tables(
        self, hdf5_path: str, simu_name: str, use_hdf5_cache: bool
    ) -> Dict[str, pd.DataFrame]:
        if use_hdf5_cache:
            cached = self._read_counting_tables_from_cache(hdf5_path)
            if cached is not None:
                return cached
        return self.hdf5_file_processor.read_file(
            hdf5_path,
            simu_name,
            use_cache=use_hdf5_cache,
        )

    def _read_counting_tables_from_cache(self, hdf5_path: str) -> Dict[str, pd.DataFrame] | None:
        feather_path_of = self.hdf5_file_processor._get_feather_path_of(hdf5_path)
        scalars_path = feather_path_of["scalars"]
        binaries_path = feather_path_of["binaries"]
        if not pd.io.common.file_exists(scalars_path) or not pd.io.common.file_exists(
            binaries_path
        ):
            return None

        scalars = pd.read_feather(scalars_path, columns=["TTOT", "Time[Myr]"])
        binary_columns = [
            "Bin Name1",
            "Bin Name2",
            "Bin KW1",
            "Bin KW2",
            "TTOT",
            "Time[Myr]",
            "Stellar Type",
        ]
        try:
            binaries = pd.read_feather(binaries_path, columns=binary_columns)
        except (KeyError, ValueError, TypeError):
            binaries = pd.read_feather(binaries_path, columns=binary_columns[:-1])
        if "TTOT" not in scalars.columns:
            logger.warning("[cache] scalars feather missing column 'TTOT' for %s", hdf5_path)
            return None
        scalars = scalars.set_index("TTOT", drop=False)
        logger.info("[cache] Loaded compact-binary counting cache for %s", hdf5_path)
        return {"scalars": scalars, "binaries": binaries}

    def _binary_snapshot_from_tables(
        self, df_dict: Dict[str, pd.DataFrame], ttot: float
    ) -> pd.DataFrame:
        binaries = df_dict.get("binaries", pd.DataFrame())
        if "TTOT" not in binaries.columns:
            return pd.DataFrame()
        return binaries[binaries["TTOT"] == ttot].copy()

    def _snapshot_times(self, df_dict: Dict[str, pd.DataFrame]) -> Iterable[float]:
        scalars = df_dict.get("scalars", pd.DataFrame())
        if "TTOT" in scalars.columns:
            return sorted(float(t) for t in scalars["TTOT"].dropna().unique())
        return sorted(float(t) for t in scalars.index.dropna().unique())

    def _snapshot_time_myr(
        self, df_dict: Dict[str, pd.DataFrame], binary_df_at_t: pd.DataFrame, ttot: float
    ) -> float:
        if "Time[Myr]" in binary_df_at_t.columns and not binary_df_at_t["Time[Myr]"].empty:
            return float(binary_df_at_t["Time[Myr]"].max())

        scalars = df_dict.get("scalars", pd.DataFrame())
        if "Time[Myr]" not in scalars.columns:
            return np.nan
        try:
            scalar_row = scalars.loc[ttot]
        except KeyError:
            if "TTOT" not in scalars.columns:
                return np.nan
            rows = scalars[scalars["TTOT"] == ttot]
            if rows.empty:
                return np.nan
            scalar_row = rows.iloc[0]
        if isinstance(scalar_row, pd.DataFrame):
            scalar_row = scalar_row.iloc[0]
        return float(scalar_row["Time[Myr]"])

    def _file_max_time_myr(self, df_dict: Dict[str, pd.DataFrame]) -> float:
        scalars = df_dict.get("scalars", pd.DataFrame())
        if "Time[Myr]" not in scalars.columns or scalars.empty:
            return np.nan
        return float(scalars["Time[Myr]"].max())

    def _accumulate_snapshot(
        self,
        records_by_category: Dict[str, Dict[Tuple[Hashable, Hashable], Dict[str, Any]]],
        binary_df_at_t: pd.DataFrame,
    ) -> None:
        required = {"Bin Name1", "Bin Name2", "Bin KW1", "Bin KW2", "TTOT"}
        missing = required.difference(binary_df_at_t.columns)
        if missing:
            raise ValueError(
                "Binary snapshot missing required columns for compact binary counting: "
                + ", ".join(sorted(missing))
            )

        candidate_df = binary_df_at_t.loc[self._candidate_mask(binary_df_at_t)]
        for _, row in candidate_df.iterrows():
            kw1 = int(row["Bin KW1"])
            kw2 = int(row["Bin KW2"])
            categories = self._categories_for(kw1, kw2)
            if not categories:
                continue

            key, bin_name1, bin_name2 = self._binary_key(row["Bin Name1"], row["Bin Name2"])
            for category in categories:
                self._update_category_record(
                    records_by_category[category],
                    key,
                    bin_name1,
                    bin_name2,
                    row,
                    kw1,
                    kw2,
                )

    def _categories_for(self, kw1: int, kw2: int) -> Tuple[str, ...]:
        categories = []
        has_bh = kw1 == self.BH_KW or kw2 == self.BH_KW
        has_ns = kw1 == self.NS_KW or kw2 == self.NS_KW

        if (kw1 == self.BH_KW and kw2 in {self.BH_KW, self.NS_KW}) or (
            kw2 == self.BH_KW and kw1 == self.NS_KW
        ):
            categories.append("gw_source")
        if has_ns and not has_bh:
            categories.append("pulsar")

        compact1 = kw1 in self.compact_object_kw
        compact2 = kw2 in self.compact_object_kw
        if compact1 != compact2:
            categories.append("xray_binary")
        return tuple(categories)

    def _candidate_mask(self, binary_df_at_t: pd.DataFrame) -> pd.Series:
        kw1 = binary_df_at_t["Bin KW1"]
        kw2 = binary_df_at_t["Bin KW2"]
        has_bh = (kw1 == self.BH_KW) | (kw2 == self.BH_KW)
        has_ns = (kw1 == self.NS_KW) | (kw2 == self.NS_KW)
        gw_source = ((kw1 == self.BH_KW) & (kw2.isin([self.BH_KW, self.NS_KW]))) | (
            (kw2 == self.BH_KW) & (kw1 == self.NS_KW)
        )
        pulsar = has_ns & ~has_bh
        compact1 = kw1.isin(self.compact_object_kw)
        compact2 = kw2.isin(self.compact_object_kw)
        xray_binary = compact1 ^ compact2
        return gw_source | pulsar | xray_binary

    def _binary_key(
        self, name1: Hashable, name2: Hashable
    ) -> Tuple[Tuple[Hashable, Hashable], Hashable, Hashable]:
        ordered = sorted((name1, name2), key=self._name_sort_key)
        return (ordered[0], ordered[1]), ordered[0], ordered[1]

    def _name_sort_key(self, value: Hashable) -> Tuple[int, Any]:
        if isinstance(value, Number):
            return (0, float(value))
        return (1, str(value))

    def _update_category_record(
        self,
        records: Dict[Tuple[Hashable, Hashable], Dict[str, Any]],
        key: Tuple[Hashable, Hashable],
        bin_name1: Hashable,
        bin_name2: Hashable,
        row: pd.Series,
        kw1: int,
        kw2: int,
    ) -> None:
        ttot = float(row["TTOT"])
        time_myr = (
            float(row["Time[Myr]"]) if "Time[Myr]" in row and pd.notna(row["Time[Myr]"]) else np.nan
        )
        stellar_type = self._stellar_type(row, kw1, kw2)

        if key not in records:
            records[key] = {
                "binary_key": key,
                "bin_name1": bin_name1,
                "bin_name2": bin_name2,
                "first_ttot": ttot,
                "last_ttot": ttot,
                "first_time_myr": time_myr,
                "last_time_myr": time_myr,
                "first_kw1": kw1,
                "first_kw2": kw2,
                "last_kw1": kw1,
                "last_kw2": kw2,
                "first_stellar_type": stellar_type,
                "last_stellar_type": stellar_type,
                "n_snapshots_seen_in_category": 1,
            }
            return

        record = records[key]
        if ttot < record["first_ttot"]:
            record["first_ttot"] = ttot
            record["first_time_myr"] = time_myr
            record["first_kw1"] = kw1
            record["first_kw2"] = kw2
            record["first_stellar_type"] = stellar_type
        if ttot >= record["last_ttot"]:
            record["last_ttot"] = ttot
            record["last_time_myr"] = time_myr
            record["last_kw1"] = kw1
            record["last_kw2"] = kw2
            record["last_stellar_type"] = stellar_type
        record["n_snapshots_seen_in_category"] += 1

    def _stellar_type(self, row: pd.Series, kw1: int, kw2: int) -> str:
        if "Stellar Type" in row and pd.notna(row["Stellar Type"]):
            return row["Stellar Type"]
        kw_to_type = getattr(self.config, "kw_to_stellar_type", {})
        return f"{kw_to_type.get(kw1, kw1)}-{kw_to_type.get(kw2, kw2)}"

    def _records_to_dataframe(
        self, records: Dict[Tuple[Hashable, Hashable], Dict[str, Any]]
    ) -> pd.DataFrame:
        if not records:
            return pd.DataFrame(columns=DETAIL_COLUMNS)
        df = pd.DataFrame(records.values())
        return df.sort_values(["first_ttot", "bin_name1", "bin_name2"]).reset_index(drop=True)[
            DETAIL_COLUMNS
        ]

    def _nanmax(self, previous: float, value: float) -> float:
        if np.isnan(previous):
            return value
        if np.isnan(value):
            return previous
        return max(previous, value)
