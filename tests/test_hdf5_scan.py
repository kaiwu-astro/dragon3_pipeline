"""Tests for shared HDF5 scan data-reduction helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from dragon3_pipelines.analysis.b_type_binary import BTypeBinaryExtractor
from dragon3_pipelines.analysis.binary_stellar_type import BinaryStellarTypeExtractor
from dragon3_pipelines.analysis.primordial_binary import PrimordialBinaryIdentifier
from dragon3_pipelines.analysis.hdf5_scan import HDF5ScanOptions, HDF5ScanRunner
from dragon3_pipelines.io import HDF5FileProcessor


def make_config(tmp_path: Path) -> Mock:
    config = Mock()
    config.pathof = {"sim": str(tmp_path)}
    config.analysis_cache_dir_of = {"sim": str(tmp_path / "cache" / "sim")}
    config.particle_df_cache_dir_of = {"sim": str(tmp_path / "cache" / "sim" / "particle_df")}
    config.input_file_path_of = {"sim": str(tmp_path / "input.inp")}
    config.processes_count = 2
    config.kw_to_stellar_type = {1: "MS", 13: "NS", 14: "BH"}
    config.stellar_type_to_kw = {"MS": 1, "NS": 13, "BH": 14}
    config.binary_stellar_type_extraction = {
        "sample_every_nb_time": 1.0,
        "wait_age_hour": 0,
        "use_hdf5_cache": True,
        "parallel": False,
        "processes": None,
        "cache_filename_template": "binaries_with_{target}_until_{last_ttot:.6f}.feather",
    }
    return config


def test_binary_stellar_type_resolves_type_and_kw(tmp_path: Path) -> None:
    extractor = BinaryStellarTypeExtractor(make_config(tmp_path))

    assert extractor.resolve_target(stellar_type="bh") == (14, "BH")
    assert extractor.resolve_target(stellar_type="MS") == (1, "MS")
    assert extractor.resolve_target(kw=14) == (14, "BH")
    assert extractor.resolve_target(kw="13") == (13, "NS")

    with pytest.raises(ValueError, match="exactly one"):
        extractor.resolve_target()
    with pytest.raises(ValueError, match="exactly one"):
        extractor.resolve_target(stellar_type="BH", kw=14)
    with pytest.raises(ValueError, match="Unknown stellar_type"):
        extractor.resolve_target(stellar_type="NOPE")
    with pytest.raises(ValueError, match="Unknown KW"):
        extractor.resolve_target(kw=99)


def test_analysis_cache_helper_accepts_legacy_particle_cache_mock(tmp_path: Path) -> None:
    config = Mock()
    config.particle_df_cache_dir_of = {"sim": str(tmp_path / "legacy_cache")}
    config.kw_to_stellar_type = {14: "BH"}
    config.stellar_type_to_kw = {"BH": 14}
    config.binary_stellar_type_extraction = {}

    task = BinaryStellarTypeExtractor(config)
    assert task.load_binaries_with_stellar_type("sim", stellar_type="BH", update=False).empty
    assert (tmp_path / "legacy_cache" / "binary_stellar_type").exists() is False


class FakeProcessor:
    def __init__(self, hdf5_paths: list[str], tables_by_path: dict[str, dict[str, pd.DataFrame]]):
        self.hdf5_paths = hdf5_paths
        self.tables_by_path = tables_by_path
        self.read_count = 0

    def get_all_hdf5_paths(self, *args, **kwargs):
        return self.hdf5_paths

    def read_tables(self, hdf5_path, simu_name, tables, columns_by_table=None, use_cache=True):
        self.read_count += 1
        return {table: self.tables_by_path[hdf5_path][table] for table in tables}


def test_binary_extractor_returns_full_matching_binary_rows_and_writes_meta(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    hdf5_path = tmp_path / "snap.40_1.0.h5part"
    hdf5_path.write_text("fake")
    tables = {
        str(hdf5_path): {
            "scalars": pd.DataFrame({"TTOT": [1.0, 2.0]}).set_index("TTOT", drop=False),
            "binaries": pd.DataFrame(
                {
                    "Bin KW1": [14, 1, 13],
                    "Bin KW2": [1, 14, 1],
                    "TTOT": [1.0, 1.0, 2.0],
                    "Time[Myr]": [10.0, 10.0, 20.0],
                    "Bin Name1": [101, 102, 103],
                    "Bin Name2": [201, 202, 203],
                    "extra_processed_column": ["keep-a", "keep-b", "keep-c"],
                }
            ),
        }
    }
    extractor = BinaryStellarTypeExtractor(config)
    fake_processor = FakeProcessor([str(hdf5_path)], tables)
    extractor.hdf5_file_processor = fake_processor

    result = extractor.load_binaries_with_stellar_type("sim", stellar_type="bh")

    assert len(result) == 2
    assert set(result["extra_processed_column"]) == {"keep-a", "keep-b"}
    assert list(result["TTOT"]) == [1.0, 1.0]
    cache_files = list((tmp_path / "cache" / "sim" / "binary_stellar_type").glob("*.feather"))
    assert cache_files[0].name == "binaries_with_14_BH_until_2.000000.feather"
    meta = json.loads(cache_files[0].with_name(cache_files[0].stem + ".meta.json").read_text())
    assert meta["target_kw"] == 14
    assert meta["target_stellar_type"] == "BH"
    assert meta["processed_files"][str(hdf5_path)]["ttot"] == [1.0, 2.0]

    result_again = extractor.load_binaries_with_stellar_type("sim", stellar_type="BH")
    assert fake_processor.read_count == 1
    pd.testing.assert_frame_equal(result, result_again)


def test_binary_extractor_writes_metadata_for_empty_match(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    hdf5_path = tmp_path / "snap.40_1.0.h5part"
    hdf5_path.write_text("fake")
    tables = {
        str(hdf5_path): {
            "scalars": pd.DataFrame({"TTOT": [1.0]}).set_index("TTOT", drop=False),
            "binaries": pd.DataFrame({"Bin KW1": [1], "Bin KW2": [1], "TTOT": [1.0]}),
        }
    }
    extractor = BinaryStellarTypeExtractor(config)
    extractor.hdf5_file_processor = FakeProcessor([str(hdf5_path)], tables)

    result = extractor.load_binaries_with_stellar_type("sim", kw=14)

    assert result.empty
    cache_files = list((tmp_path / "cache" / "sim" / "binary_stellar_type").glob("*.feather"))
    assert cache_files
    meta = json.loads(cache_files[0].with_name(cache_files[0].stem + ".meta.json").read_text())
    assert meta["processed_files"][str(hdf5_path)]["ttot"] == [1.0]


def test_hdf5_file_processor_reads_selected_feather_tables(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    processor = HDF5FileProcessor(config)
    hdf5_path = str(tmp_path / "snap.40_1.0.h5part")
    pd.DataFrame({"TTOT": [1.0], "Time[Myr]": [10.0], "unused": [0]}).to_feather(
        hdf5_path + ".scalars.df.feather"
    )
    pd.DataFrame({"TTOT": [1.0], "Bin KW1": [14], "unused": [0]}).to_feather(
        hdf5_path + ".binaries.df.feather"
    )
    processor.read_file = Mock(side_effect=AssertionError("should not parse full HDF5"))

    result = processor.read_tables(
        hdf5_path,
        "sim",
        tables=["scalars", "binaries"],
        columns_by_table={"scalars": ["TTOT"], "binaries": ["TTOT", "Bin KW1"]},
        use_cache=True,
    )

    assert list(result["scalars"].columns) == ["TTOT"]
    assert list(result["binaries"].columns) == ["TTOT", "Bin KW1"]
    processor.read_file.assert_not_called()


def test_hdf5_file_processor_falls_back_when_feather_columns_missing(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    processor = HDF5FileProcessor(config)
    hdf5_path = str(tmp_path / "snap.40_1.0.h5part")
    pd.DataFrame({"TTOT": [1.0]}).to_feather(hdf5_path + ".scalars.df.feather")
    pd.DataFrame({"TTOT": [1.0]}).to_feather(hdf5_path + ".binaries.df.feather")
    fallback = {
        "scalars": pd.DataFrame({"TTOT": [1.0], "Time[Myr]": [10.0]}).set_index("TTOT", drop=False),
        "binaries": pd.DataFrame({"TTOT": [1.0], "Bin KW1": [14]}),
    }
    processor.read_file = Mock(return_value=fallback)

    result = processor.read_tables(
        hdf5_path,
        "sim",
        tables=["scalars", "binaries"],
        columns_by_table={"scalars": ["TTOT", "Time[Myr]"], "binaries": ["Bin KW1"]},
        use_cache=True,
    )

    assert result["binaries"]["Bin KW1"].iloc[0] == 14
    processor.read_file.assert_called_once()


class FakeTask:
    required_tables = ("scalars", "binaries")
    columns_by_table = {"scalars": ["TTOT"], "binaries": ["TTOT"]}

    def __init__(self, name: str):
        self.name = name
        self.writes = 0

    def read_cache(self):
        return pd.DataFrame()

    def read_meta(self):
        return {}

    def is_file_fresh(self, hdf5_path, meta, cache_df):
        return False

    def process_file(self, hdf5_path, df_dict, meta, cache_df):
        return {
            "rows": pd.DataFrame({"task": [self.name], "path": [hdf5_path]}),
            "file_meta": {"mtime": 1.0, "ttot": [1.0]},
        }

    def merge_file_result(self, cache_df, hdf5_path, result):
        return pd.concat([cache_df, result["rows"]], ignore_index=True)

    def write_cache_and_meta(self, cache_df, processed_files, options):
        self.writes += 1

    def finalize_cache(self, cache_df):
        return cache_df


def test_scan_runner_reads_each_hdf5_file_once_for_multiple_tasks(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    hdf5_path = tmp_path / "snap.40_1.0.h5part"
    hdf5_path.write_text("fake")
    tables = {
        str(hdf5_path): {
            "scalars": pd.DataFrame({"TTOT": [1.0]}),
            "binaries": pd.DataFrame({"TTOT": [1.0]}),
        }
    }
    processor = FakeProcessor([str(hdf5_path)], tables)
    runner = HDF5ScanRunner(config, processor)
    task_a = FakeTask("a")
    task_b = FakeTask("b")

    result = runner.run("sim", [task_a, task_b], HDF5ScanOptions(wait_age_hour=0))

    assert processor.read_count == 1
    assert set(result) == {"a", "b"}
    assert result["a"]["task"].tolist() == ["a"]
    assert result["b"]["task"].tolist() == ["b"]
    assert task_a.writes == 1
    assert task_b.writes == 1


def make_primordial_tables(
    hdf5_path: Path, binaries: pd.DataFrame
) -> dict[str, dict[str, pd.DataFrame]]:
    return {
        str(hdf5_path): {
            "scalars": pd.DataFrame({"TTOT": sorted(binaries["TTOT"].unique())}).set_index(
                "TTOT", drop=False
            ),
            "binaries": binaries,
        }
    }


def test_primordial_identifier_filters_adjacent_integer_name_pairs(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    hdf5_path = tmp_path / "snap.40_0.0.h5part"
    hdf5_path.write_text("fake")
    binaries = pd.DataFrame(
        {
            "Bin Name1": [1, 10, 4],
            "Bin Name2": [2, 9, 6],
            "TTOT": [0.0, 0.0, 0.0],
            "extra_processed_column": ["keep-a", "keep-b", "drop-c"],
        }
    )
    identifier = PrimordialBinaryIdentifier(config)
    fake_processor = FakeProcessor([str(hdf5_path)], make_primordial_tables(hdf5_path, binaries))
    identifier.hdf5_file_processor = fake_processor

    result = identifier.load_primordial_binaries("sim", wait_age_hour=0)

    assert result["extra_processed_column"].tolist() == ["keep-a", "keep-b"]
    assert result["primordial_name_min"].tolist() == [1, 9]
    assert result["primordial_name_max"].tolist() == [2, 10]
    assert result["primordial_pair_key"].tolist() == ["1-2", "9-10"]
    assert result["is_primordial_binary"].tolist() == [True, True]
    assert fake_processor.read_count == 1

    cache_path = tmp_path / "cache" / "sim" / "primordial_binary" / "primordial_binaries.feather"
    meta_path = tmp_path / "cache" / "sim" / "primordial_binary" / "primordial_binaries.meta.json"
    assert cache_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["schema_version"] == 1
    assert meta["source_hdf5_path"] == str(hdf5_path)
    assert meta["source_mtime"] == hdf5_path.stat().st_mtime
    assert meta["discovered_ttot_values"] == [0.0]
    assert meta["row_count"] == 2
    assert meta["ttot_rule"] == "binaries['TTOT'].astype(float) == 0.0"


def test_primordial_identifier_uses_strict_zero_ttot_snapshot(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    hdf5_path = tmp_path / "snap.40_0.0.h5part"
    hdf5_path.write_text("fake")
    binaries = pd.DataFrame(
        {
            "Bin Name1": [1, 20, 30],
            "Bin Name2": [2, 21, 31],
            "TTOT": [0.0, 0.1, 1.0],
            "extra_processed_column": ["keep-zero", "drop-nonzero", "drop-one"],
        }
    )
    identifier = PrimordialBinaryIdentifier(config)
    identifier.hdf5_file_processor = FakeProcessor(
        [str(hdf5_path)], make_primordial_tables(hdf5_path, binaries)
    )

    result = identifier.load_primordial_binaries("sim", wait_age_hour=0)

    assert result["extra_processed_column"].tolist() == ["keep-zero"]
    assert result["TTOT"].tolist() == [0.0]


def test_primordial_identifier_reuses_fresh_cache_without_rereading_hdf5(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    hdf5_path = tmp_path / "snap.40_0.0.h5part"
    hdf5_path.write_text("fake")
    binaries = pd.DataFrame({"Bin Name1": [1], "Bin Name2": [2], "TTOT": [0.0]})
    identifier = PrimordialBinaryIdentifier(config)
    fake_processor = FakeProcessor([str(hdf5_path)], make_primordial_tables(hdf5_path, binaries))
    identifier.hdf5_file_processor = fake_processor

    first = identifier.load_primordial_binaries("sim", wait_age_hour=0)
    second = identifier.load_primordial_binaries("sim", wait_age_hour=0)

    assert fake_processor.read_count == 1
    pd.testing.assert_frame_equal(first, second)


def test_primordial_identifier_update_false_reads_cache(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    hdf5_path = tmp_path / "snap.40_0.0.h5part"
    hdf5_path.write_text("fake")
    binaries = pd.DataFrame({"Bin Name1": [1], "Bin Name2": [2], "TTOT": [0.0]})
    identifier = PrimordialBinaryIdentifier(config)
    fake_processor = FakeProcessor([str(hdf5_path)], make_primordial_tables(hdf5_path, binaries))
    identifier.hdf5_file_processor = fake_processor

    first = identifier.load_primordial_binaries("sim", wait_age_hour=0)
    fake_processor.hdf5_paths = []
    second = identifier.load_primordial_binaries("sim", update=False)

    assert fake_processor.read_count == 1
    pd.testing.assert_frame_equal(first, second)


def test_primordial_identifier_fails_when_no_hdf5_files(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    identifier = PrimordialBinaryIdentifier(config)
    identifier.hdf5_file_processor = FakeProcessor([], {})

    with pytest.raises(ValueError, match="No HDF5 files"):
        identifier.load_primordial_binaries("sim", wait_age_hour=0)


def test_primordial_identifier_fails_when_required_name_columns_missing(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    hdf5_path = tmp_path / "snap.40_0.0.h5part"
    hdf5_path.write_text("fake")
    binaries = pd.DataFrame({"Bin Name1": [1], "TTOT": [0.0]})
    identifier = PrimordialBinaryIdentifier(config)
    identifier.hdf5_file_processor = FakeProcessor(
        [str(hdf5_path)], make_primordial_tables(hdf5_path, binaries)
    )

    with pytest.raises(ValueError, match="Bin Name2"):
        identifier.load_primordial_binaries("sim", wait_age_hour=0)


def test_primordial_identifier_fails_when_zero_ttot_snapshot_missing(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    hdf5_path = tmp_path / "snap.40_0.0.h5part"
    hdf5_path.write_text("fake")
    binaries = pd.DataFrame({"Bin Name1": [1], "Bin Name2": [2], "TTOT": [0.1]})
    identifier = PrimordialBinaryIdentifier(config)
    identifier.hdf5_file_processor = FakeProcessor(
        [str(hdf5_path)], make_primordial_tables(hdf5_path, binaries)
    )

    with pytest.raises(ValueError, match="TTOT == 0.0"):
        identifier.load_primordial_binaries("sim", wait_age_hour=0)


def test_b_type_extractor_filters_members_marks_primordial_and_writes_meta(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    hdf5_path = tmp_path / "snap.40_1.0.h5part"
    hdf5_path.write_text("fake")
    binaries = pd.DataFrame(
        {
            "Bin KW1": [2, 2, 1, 2, 1, 2, 1, 1, 2],
            "Bin KW2": [2, 2, 2, 1, 1, 1, 2, 2, 2],
            "Bin Teff1*": [9000, 9000, 10500, 9000, 20000, 20000, 10499, 20000, 20000],
            "Bin Teff2*": [9000, 9000, 9000, 31500, 20000, 20000, 20000, 20000, 20000],
            "Bin M1*": [1.0, 1.0, 2.75, 1.0, 10.0, 10.0, 10.0, 200.0, 10.0],
            "Bin M2*": [1.0, 1.0, 1.0, 17.7, 10.0, 2.74, 10.0, 10.0, 10.0],
            "TTOT": [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "Time[Myr]": [0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            "Bin Name1": [1, 9, 2, 30, 10, 50, 60, 70, 80],
            "Bin Name2": [2, 10, 1, 31, 9, 51, 61, 71, 81],
            "extra_processed_column": [
                "primordial-a",
                "primordial-b",
                "member1-lower-boundary",
                "member2-upper-boundary",
                "both-members",
                "drop-mass-low",
                "drop-teff-low",
                "drop-mass-high",
                "drop-kw",
            ],
        }
    )
    extractor = BTypeBinaryExtractor(config)
    fake_processor = FakeProcessor([str(hdf5_path)], make_primordial_tables(hdf5_path, binaries))
    extractor.hdf5_file_processor = fake_processor

    result = extractor.load_b_type_binaries("sim", wait_age_hour=0)

    assert result["extra_processed_column"].tolist() == [
        "member1-lower-boundary",
        "both-members",
        "member2-upper-boundary",
    ]
    assert result["b_type_member1"].tolist() == [True, True, False]
    assert result["b_type_member2"].tolist() == [False, True, True]
    assert result["b_type_member_count"].tolist() == [1, 2, 1]
    assert result["b_type_pair_key"].tolist() == ["1-2", "9-10", "30-31"]
    assert result["is_primordial_binary"].tolist() == [True, True, False]
    assert "extra_processed_column" in result.columns

    cache_path = (
        tmp_path / "cache" / "sim" / "b_type_binary" / "b_type_binaries_until_1.000000.feather"
    )
    meta_path = cache_path.with_name(cache_path.stem + ".meta.json")
    assert cache_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["schema_version"] == 1
    assert meta["b_type_criteria"] == {
        "kw": 1,
        "teff_min": 10500,
        "teff_max": 31500,
        "mass_min": 2.75,
        "mass_max": 17.7,
    }
    assert meta["primordial_signature"]["meta"]["row_count"] == 2
    assert meta["processed_files"][str(hdf5_path)]["ttot"] == [0.0, 1.0]

    result_again = extractor.load_b_type_binaries("sim", wait_age_hour=0)
    assert fake_processor.read_count == 2
    pd.testing.assert_frame_equal(result, result_again)


def test_b_type_extractor_refreshes_when_primordial_cache_changes(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    hdf5_path = tmp_path / "snap.40_1.0.h5part"
    hdf5_path.write_text("fake")
    binaries = pd.DataFrame(
        {
            "Bin KW1": [2, 1],
            "Bin KW2": [2, 2],
            "Bin Teff1*": [9000, 20000],
            "Bin Teff2*": [9000, 9000],
            "Bin M1*": [1.0, 5.0],
            "Bin M2*": [1.0, 1.0],
            "TTOT": [0.0, 1.0],
            "Bin Name1": [1, 2],
            "Bin Name2": [2, 1],
        }
    )
    extractor = BTypeBinaryExtractor(config)
    fake_processor = FakeProcessor([str(hdf5_path)], make_primordial_tables(hdf5_path, binaries))
    extractor.hdf5_file_processor = fake_processor

    first = extractor.load_b_type_binaries("sim", wait_age_hour=0)
    assert first["is_primordial_binary"].tolist() == [True]
    assert fake_processor.read_count == 2

    primordial_meta = (
        tmp_path / "cache" / "sim" / "primordial_binary" / "primordial_binaries.meta.json"
    )
    meta = json.loads(primordial_meta.read_text())
    meta["row_count"] = 0
    primordial_meta.write_text(json.dumps(meta, indent=2, sort_keys=True))

    second = extractor.load_b_type_binaries("sim", wait_age_hour=0)

    assert fake_processor.read_count == 3
    assert second["is_primordial_binary"].tolist() == [True]


def test_b_type_extractor_handles_duplicate_binary_indices(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    hdf5_path = tmp_path / "snap.40_1.0.h5part"
    hdf5_path.write_text("fake")
    binaries = pd.DataFrame(
        {
            "Bin KW1": [2, 1, 1],
            "Bin KW2": [2, 2, 1],
            "Bin Teff1*": [9000, 20000, 20000],
            "Bin Teff2*": [9000, 9000, 20000],
            "Bin M1*": [1.0, 5.0, 5.0],
            "Bin M2*": [1.0, 1.0, 5.0],
            "TTOT": [0.0, 1.0, 2.0],
            "Bin Name1": [1, 10, 20],
            "Bin Name2": [2, 11, 21],
        },
        index=[0, 0, 0],
    )
    extractor = BTypeBinaryExtractor(config)
    extractor.hdf5_file_processor = FakeProcessor(
        [str(hdf5_path)], make_primordial_tables(hdf5_path, binaries)
    )

    result = extractor.load_b_type_binaries("sim", wait_age_hour=0)

    assert result["TTOT"].tolist() == [1.0, 2.0]
    assert result["b_type_member1"].tolist() == [True, True]
    assert result["b_type_member2"].tolist() == [False, True]
