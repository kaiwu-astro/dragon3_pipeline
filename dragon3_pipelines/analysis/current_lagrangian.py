"""Current-mass Lagrangian table construction from HDF5 snapshots."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from dragon3_pipelines.io import HDF5FileProcessor
from dragon3_pipelines.io.text_parsers import make_l7header, transform_l7df_to_sns_friendly

logger = logging.getLogger(__name__)


class CurrentMassLagrangianProcessor:
    """Build and cache current-mass Lagrangian profiles from HDF5 snapshots."""

    SCHEMA_VERSION = 1
    METRICS = [
        "rlagr",
        "avmass",
        "nshell",
        "vx",
        "vy",
        "vz",
        "v",
        "vr",
        "vt",
        "sigma2",
        "sigma_r2",
        "sigma_t2",
        "vrot",
    ]

    def __init__(self, config_manager: Any) -> None:
        self.config = config_manager
        self.hdf5_file_processor = HDF5FileProcessor(config_manager)

    @property
    def percentages(self) -> List[str]:
        """Return lagr.7 total-population percentage suffixes."""
        return [
            col.removeprefix("rlagr")
            for col in make_l7header()
            if col.startswith("rlagr") and not col.startswith(("rlagr_s", "rlagr_b"))
        ]

    def _current_lagrangian_config(self) -> Dict[str, Any]:
        defaults = {
            "enabled": True,
            "sample_every_nb_time": 1.0,
            "wait_age_hour": 24,
            "use_hdf5_cache": True,
            "cache_filename": "current_mass_lagr.feather",
        }
        user_config = getattr(self.config, "current_lagrangian", {}) or {}
        return {**defaults, **user_config}

    def _cache_dir(self, simu_name: str) -> Path:
        return Path(self.config.particle_df_cache_dir_of[simu_name]) / "current_lagrangian"

    def _cache_path(self, simu_name: str) -> Path:
        return self._cache_dir(simu_name) / self._current_lagrangian_config()["cache_filename"]

    def _meta_path(self, simu_name: str) -> Path:
        cache_path = self._cache_path(simu_name)
        return cache_path.with_name(cache_path.stem + ".meta.json")

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
            logger.warning("Failed to read current Lagrangian metadata %s: %r", meta_path, exc)
            return {}

    def _write_cache_and_meta(
        self, simu_name: str, df: pd.DataFrame, processed_files: Dict[str, Dict[str, Any]]
    ) -> None:
        cache_dir = self._cache_dir(simu_name)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._cache_path(simu_name)
        meta_path = self._meta_path(simu_name)

        tmp_cache_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        tmp_meta_path = meta_path.with_suffix(meta_path.suffix + ".tmp")

        df.to_feather(tmp_cache_path)
        os.replace(tmp_cache_path, cache_path)

        current_config = self._current_lagrangian_config()
        meta = {
            "schema_version": self.SCHEMA_VERSION,
            "sample_every_nb_time": current_config["sample_every_nb_time"],
            "wait_age_hour": current_config["wait_age_hour"],
            "statistics": "current-mass weighted, singles table only",
            "percentages": self.percentages,
            "processed_files": processed_files,
        }
        tmp_meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
        os.replace(tmp_meta_path, meta_path)

    def _is_file_fresh_in_meta(
        self, hdf5_path: str, meta: Dict[str, Any], cached_times: set[float]
    ) -> bool:
        file_meta = meta.get("processed_files", {}).get(hdf5_path)
        if not file_meta:
            return False
        try:
            current_mtime = os.path.getmtime(hdf5_path)
        except OSError:
            return False
        if not np.isclose(float(file_meta.get("mtime", np.nan)), current_mtime):
            return False
        return set(file_meta.get("ttot", [])).issubset(cached_times)

    def update(self, simu_name: str) -> pd.DataFrame:
        """Update the cached current-mass Lagrangian table for one simulation."""
        current_config = self._current_lagrangian_config()
        cache_df = self._read_cache(simu_name)
        meta = self._read_meta(simu_name)
        processed_files = dict(meta.get("processed_files", {}))
        cached_times = (
            set(cache_df["Time[NB]"].astype(float).tolist())
            if "Time[NB]" in cache_df.columns
            else set()
        )

        hdf5_paths = self.hdf5_file_processor.get_all_hdf5_paths(
            simu_name,
            wait_age_hour=current_config["wait_age_hour"],
            sample_every_nb_time=current_config["sample_every_nb_time"],
        )
        new_rows = []

        for hdf5_path in hdf5_paths:
            if self._is_file_fresh_in_meta(hdf5_path, meta, cached_times):
                continue

            df_dict = self.hdf5_file_processor.read_file(
                hdf5_path,
                simu_name,
                use_cache=current_config["use_hdf5_cache"],
            )
            file_times = [float(t) for t in sorted(df_dict["scalars"]["TTOT"].unique())]
            old_file_meta = meta.get("processed_files", {}).get(hdf5_path)
            file_mtime = os.path.getmtime(hdf5_path)
            if old_file_meta and not np.isclose(
                float(old_file_meta.get("mtime", np.nan)), file_mtime
            ):
                times_to_compute = file_times
            else:
                times_to_compute = [t for t in file_times if t not in cached_times]

            for ttot in times_to_compute:
                single_df_at_t, _, is_valid = self.hdf5_file_processor.get_snapshot_at_t(
                    df_dict, ttot
                )
                if not is_valid or single_df_at_t is None:
                    logger.warning(
                        "Skipping invalid current Lagrangian snapshot %s TTOT=%s",
                        hdf5_path,
                        ttot,
                    )
                    continue
                scalar_row = df_dict["scalars"].loc[ttot]
                new_rows.append(self.compute_snapshot(single_df_at_t, scalar_row))

            processed_files[hdf5_path] = {
                "mtime": file_mtime,
                "ttot": file_times,
            }

        if new_rows:
            new_df = pd.DataFrame(new_rows)
            if not cache_df.empty and "Time[NB]" in cache_df.columns:
                cache_df = cache_df[~cache_df["Time[NB]"].isin(new_df["Time[NB]"])]
            cache_df = pd.concat([cache_df, new_df], ignore_index=True, sort=False)

        if not cache_df.empty:
            cache_df = cache_df.sort_values("Time[NB]").drop_duplicates(
                subset=["Time[NB]"], keep="last"
            )
            cache_df = cache_df.reset_index(drop=True)

        self._write_cache_and_meta(simu_name, cache_df, processed_files)
        return cache_df

    def load_sns_friendly_data(self, simu_name: str, update: bool = True) -> pd.DataFrame:
        """Return a seaborn-friendly long table compatible with ``LagrVisualizer``."""
        df = self.update(simu_name) if update else self._read_cache(simu_name)
        if df.empty:
            return pd.DataFrame(columns=["Time[Myr]", "Percentage", "Metric", "Value", "%"])

        plot_df = df.drop(columns=["Time[NB]"], errors="ignore")
        l7df_sns = transform_l7df_to_sns_friendly(plot_df)
        return self._append_sigma_rows(l7df_sns)

    def compute_snapshot(
        self, single_df_at_t: pd.DataFrame, scalar_row: pd.Series
    ) -> Dict[str, float]:
        """Compute one wide-table row for a single snapshot."""
        row: Dict[str, float] = {
            "Time[NB]": float(scalar_row["TTOT"]),
            "Time[Myr]": float(scalar_row["Time[Myr]"]),
        }

        snapshot_df = single_df_at_t.copy()
        if snapshot_df.empty:
            for metric in self.METRICS:
                for suffix in self.percentages:
                    row[f"{metric}{suffix}"] = np.nan
            return row

        mass = snapshot_df["M"].to_numpy(dtype=float)
        total_mass = mass.sum()
        if total_mass <= 0:
            raise ValueError("Current-mass Lagrangian snapshot has non-positive total mass.")

        velocity = snapshot_df[["V1", "V2", "V3"]].to_numpy(dtype=float)
        center_of_mass_velocity = np.average(velocity, axis=0, weights=mass)
        velocity = velocity - center_of_mass_velocity

        radius = snapshot_df["Distance_to_cluster_center[pc]"].to_numpy(dtype=float)
        position = self._position_array(snapshot_df, radius)
        order = np.argsort(radius, kind="mergesort")
        sorted_mass = mass[order]
        cumulative_mass = np.cumsum(sorted_mass)

        for suffix in self.percentages:
            if suffix == "<RC":
                rc_pc = float(scalar_row["RC"]) * float(scalar_row["RBAR"])
                mask = radius <= rc_pc
            else:
                target_mass = total_mass * float(suffix)
                idx = int(np.searchsorted(cumulative_mass, target_mass, side="left"))
                idx = min(idx, len(order) - 1)
                lagr_radius = radius[order[idx]]
                mask = radius <= lagr_radius

            stats = self._compute_region_stats(mass, radius, position, velocity, mask)
            for metric, value in stats.items():
                row[f"{metric}{suffix}"] = value

        return row

    def _position_array(self, snapshot_df: pd.DataFrame, radius: np.ndarray) -> np.ndarray:
        if {"X [pc]", "Y [pc]", "Z [pc]"}.issubset(snapshot_df.columns):
            return snapshot_df[["X [pc]", "Y [pc]", "Z [pc]"]].to_numpy(dtype=float)
        position_cols = ["X1", "X2", "X3"]
        if set(position_cols).issubset(snapshot_df.columns):
            return snapshot_df[position_cols].to_numpy(dtype=float)
        position = np.zeros((len(radius), 3), dtype=float)
        position[:, 0] = radius
        return position

    def _compute_region_stats(
        self,
        mass: np.ndarray,
        radius: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
        mask: Iterable[bool],
    ) -> Dict[str, float]:
        mask = np.asarray(mask, dtype=bool)
        n_stars = int(mask.sum())
        if n_stars == 0:
            return {
                "rlagr": np.nan,
                "avmass": np.nan,
                "nshell": 0,
                "vx": np.nan,
                "vy": np.nan,
                "vz": np.nan,
                "v": np.nan,
                "vr": np.nan,
                "vt": np.nan,
                "sigma2": np.nan,
                "sigma_r2": np.nan,
                "sigma_t2": np.nan,
                "vrot": np.nan,
            }

        region_mass = mass[mask]
        region_position = position[mask]
        region_velocity = velocity[mask]
        region_radius = radius[mask]
        region_mass_sum = region_mass.sum()

        mean_velocity = np.average(region_velocity, axis=0, weights=region_mass)
        speed = np.linalg.norm(region_velocity, axis=1)

        radial_unit = np.divide(
            region_position,
            region_radius[:, None],
            out=np.zeros_like(region_position, dtype=float),
            where=region_radius[:, None] > 0,
        )
        radial_velocity = np.sum(region_velocity * radial_unit, axis=1)
        tangential_velocity2 = np.maximum(speed**2 - radial_velocity**2, 0.0)
        tangential_velocity = np.sqrt(tangential_velocity2)

        mean_speed = float(np.average(speed, weights=region_mass))
        mean_radial_velocity = float(np.average(radial_velocity, weights=region_mass))
        mean_tangential_velocity = float(np.average(tangential_velocity, weights=region_mass))

        velocity_delta = region_velocity - mean_velocity
        sigma2 = float(np.average(np.sum(velocity_delta**2, axis=1), weights=region_mass))
        sigma_r2 = float(
            np.average((radial_velocity - mean_radial_velocity) ** 2, weights=region_mass)
        )
        sigma_t2 = max(sigma2 - sigma_r2, 0.0)

        angular_momentum_speed = np.linalg.norm(np.cross(region_position, region_velocity), axis=1)
        vrot = np.divide(
            angular_momentum_speed,
            region_radius,
            out=np.zeros_like(angular_momentum_speed, dtype=float),
            where=region_radius > 0,
        )

        return {
            "rlagr": float(np.max(region_radius)),
            "avmass": float(region_mass_sum / n_stars),
            "nshell": n_stars,
            "vx": float(mean_velocity[0]),
            "vy": float(mean_velocity[1]),
            "vz": float(mean_velocity[2]),
            "v": mean_speed,
            "vr": mean_radial_velocity,
            "vt": mean_tangential_velocity,
            "sigma2": sigma2,
            "sigma_r2": sigma_r2,
            "sigma_t2": sigma_t2,
            "vrot": float(np.average(vrot, weights=region_mass)),
        }

    def _append_sigma_rows(self, l7df_sns: pd.DataFrame) -> pd.DataFrame:
        new_rows = []
        for metric_old in ["sigma2", "sigma_r2", "sigma_t2"]:
            df_subset = l7df_sns[l7df_sns["Metric"] == metric_old].copy()
            if not df_subset.empty:
                df_subset["Value"] = np.sqrt(df_subset["Value"])
                df_subset["Metric"] = metric_old[:-1]
                new_rows.append(df_subset)
        if not new_rows:
            return l7df_sns
        return pd.concat([l7df_sns] + new_rows, ignore_index=True)
