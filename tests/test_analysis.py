"""Tests for analysis module"""

import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from dragon3_pipelines.analysis import ParticleTracker, tau_gw


class TestParticleTracker:
    """Tests for ParticleTracker class"""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration manager"""
        config = Mock()
        config.particle_df_cache_dir_of = {"test_simu": tempfile.mkdtemp()}
        config.pathof = {"test_simu": "/path/to/test"}
        config.processes_count = 2
        config.tasks_per_child = 10
        return config

    @pytest.fixture
    def particle_tracker(self, mock_config):
        """Create a ParticleTracker instance"""
        return ParticleTracker(mock_config)

    @pytest.fixture
    def sample_df_dict(self):
        """Create sample data for testing"""
        singles_df = pd.DataFrame(
            {
                "Name": [1000, 1000, 2000],
                "TTOT": [0.0, 1.0, 0.0],
                "Time[Myr]": [0.0, 10.0, 0.0],
                "M": [10.0, 9.5, 15.0],
                "KW": [1, 1, 2],
                "X1": [0.1, 0.2, 0.3],
                "X2": [0.1, 0.2, 0.3],
                "X3": [0.1, 0.2, 0.3],
            }
        )

        binaries_df = pd.DataFrame(
            {
                "Bin Name1": [1000, 3000],
                "Bin Name2": [2000, 4000],
                "TTOT": [2.0, 0.0],
                "Time[Myr]": [20.0, 0.0],
                "Bin M1*": [9.0, 20.0],
                "Bin M2*": [14.5, 25.0],
            }
        )

        scalars_df = pd.DataFrame(
            {
                "TTOT": [0.0, 1.0, 2.0],
                "RBAR": [1.0, 1.1, 1.2],
            }
        )

        return {
            "singles": singles_df,
            "binaries": binaries_df,
            "scalars": scalars_df,
        }

    def test_init(self, mock_config):
        """Test ParticleTracker initialization"""
        tracker = ParticleTracker(mock_config)
        assert tracker.config == mock_config
        assert tracker.hdf5_file_processor is not None

    def test_get_particle_df_from_hdf5_file_single_only(self, particle_tracker, sample_df_dict):
        """Test tracking a particle that remains single"""
        result = particle_tracker.get_particle_df_from_hdf5_file(sample_df_dict, 2000)

        # Particle 2000 appears in singles at TTOT=0.0 and as Bin Name2 at TTOT=2.0
        # So we get 2 rows after merge
        assert len(result) == 2
        # The single record should have state 'single' where Name is not NaN
        single_row = result[result["Name"].notna()]
        assert len(single_row) == 1
        assert single_row["Name"].iloc[0] == 2000

    def test_get_particle_df_from_hdf5_file_single_and_binary(
        self, particle_tracker, sample_df_dict
    ):
        """Test tracking a particle that goes from single to binary"""
        result = particle_tracker.get_particle_df_from_hdf5_file(sample_df_dict, 1000)

        # Should have 3 rows: TTOT 0.0, 1.0, 2.0
        assert len(result) == 3
        assert sorted(result["TTOT"].tolist()) == [0.0, 1.0, 2.0]

        # Check binary state (TTOT=2.0)
        binary_rows = result[result["state"] == "binary"]
        assert len(binary_rows) == 1
        assert binary_rows["companion_name"].iloc[0] == 2000

    def test_get_particle_df_from_hdf5_file_not_found(self, particle_tracker, sample_df_dict):
        """Test tracking a non-existent particle"""
        result = particle_tracker.get_particle_df_from_hdf5_file(sample_df_dict, 99999)

        assert result.empty

    def test_get_particle_df_from_hdf5_file_both_binary_members(self, particle_tracker):
        """Test graceful handling and deduplication when particle appears as both binary members"""
        # Create data where particle is both Name1 and Name2
        test_df_dict = {
            "singles": pd.DataFrame(
                {
                    "Name": [1000],
                    "TTOT": [0.0],
                }
            ),
            "binaries": pd.DataFrame(
                {
                    "Bin Name1": [1000, 1000],
                    "Bin Name2": [2000, 1000],  # 1000 appears as Name2 here too!
                    "TTOT": [0.0, 1.0],
                }
            ),
            "scalars": pd.DataFrame({"TTOT": [0.0, 1.0]}),
        }

        # Should not raise error, but should log warning and deduplicate
        result = particle_tracker.get_particle_df_from_hdf5_file(test_df_dict, 1000)

        # Should have deduplicated the TTOT=1.0 entry
        assert not result.empty
        assert result["TTOT"].is_unique

    def test_get_particle_summary_empty(self, particle_tracker):
        """Test summary with empty DataFrame"""
        result = particle_tracker.get_particle_summary(pd.DataFrame())
        assert result == {}

    def test_get_particle_summary_complete(self, particle_tracker):
        """Test summary with complete particle history"""
        particle_df = pd.DataFrame(
            {
                "Name": [1000, 1000, 1000],
                "TTOT": [0.0, 1.0, 2.0],
                "Time[Myr]": [0.0, 10.0, 20.0],
                "M": [10.0, 9.5, 9.0],
                "KW": [1, 1, 2],
                "state": ["single", "single", "binary"],
            }
        )

        result = particle_tracker.get_particle_summary(particle_df)

        assert result["particle_name"] == 1000
        assert result["total_snapshots"] == 3
        assert result["time_range_myr"] == (0.0, 20.0)
        assert result["single_count"] == 2
        assert result["binary_count"] == 1
        assert result["initial_mass"] == 10.0
        assert result["final_mass"] == 9.0
        assert result["stellar_types"] == [1, 2]

    def test_process_single_hdf5_for_particle(self, particle_tracker, sample_df_dict):
        """Test worker function for parallel processing"""
        with patch.object(
            particle_tracker.hdf5_file_processor, "read_file", return_value=sample_df_dict
        ):
            result = particle_tracker._process_single_hdf5_for_particle(
                ("/path/to/file.h5", 1000, "test_simu")
            )

            assert not result.empty
            assert 1000 in result["Name"].values

    def test_process_single_hdf5_for_particle_error(self, particle_tracker):
        """Test error handling in worker function"""
        with patch.object(
            particle_tracker.hdf5_file_processor, "read_file", side_effect=Exception("Test error")
        ):
            result = particle_tracker._process_single_hdf5_for_particle(
                ("/path/to/file.h5", 1000, "test_simu")
            )

            assert result.empty

    @patch("dragon3_pipelines.analysis.particle_tracker.glob")
    @patch("os.path.exists")
    def test_get_particle_df_all_no_cache(
        self, mock_exists, mock_glob, particle_tracker, mock_config
    ):
        """Test getting particle data with no existing cache"""
        mock_exists.return_value = False
        mock_glob.return_value = []

        result = particle_tracker.get_particle_df_all("test_simu", 1000, update=False)

        assert result.empty

    @patch("dragon3_pipelines.analysis.particle_tracker.glob")
    @patch("pandas.read_feather")
    @patch("os.path.exists")
    def test_get_particle_df_all_with_cache(
        self, mock_exists, mock_read_feather, mock_glob, particle_tracker
    ):
        """Test getting particle data with existing cache"""
        cached_df = pd.DataFrame(
            {
                "Name": [1000],
                "TTOT": [5.0],
                "M": [10.0],
            }
        )

        mock_exists.return_value = True
        mock_read_feather.return_value = cached_df
        # Mock glob to return a merged cache file path
        mock_glob.return_value = ["/fake/cache/1000/1000_history_until_5.00.df.feather"]

        result = particle_tracker.get_particle_df_all("test_simu", 1000, update=False)

        assert len(result) == 1
        assert result["Name"].iloc[0] == 1000

    def test_get_particle_df_from_hdf5_file_all_particles(self, particle_tracker, sample_df_dict):
        """Test processing all particles from HDF5 file"""
        # Mock the HDF5 file processor to return sample data
        with patch.object(
            particle_tracker.hdf5_file_processor,
            "read_file",
            return_value=sample_df_dict,
        ):
            result = particle_tracker.get_particle_df_from_hdf5_file(sample_df_dict, "all")

        # Should return a dict
        assert isinstance(result, dict)
        # Should have data for particles 1000 and 2000
        assert 1000 in result
        assert 2000 in result
        # Each should be a DataFrame
        assert isinstance(result[1000], pd.DataFrame)
        assert isinstance(result[2000], pd.DataFrame)

    def test_get_particle_df_from_hdf5_file_all_with_cache(
        self, particle_tracker, sample_df_dict, mock_config, tmp_path
    ):
        """Test processing all particles with save_cache=True"""
        # Update config to use temp directory
        mock_config.particle_df_cache_dir_of["test_simu"] = str(tmp_path)

        # Mock the HDF5 file processor to return sample data and a fake time
        with patch.object(
            particle_tracker.hdf5_file_processor,
            "read_file",
            return_value=sample_df_dict,
        ), patch.object(
            particle_tracker.hdf5_file_processor,
            "get_hdf5_file_time_from_filename",
            return_value=1.5,
        ):
            result = particle_tracker.get_particle_df_from_hdf5_file(
                sample_df_dict,
                "all",
                hdf5_file_path="/fake/path/snap.40_1.5.h5part",
                simu_name="test_simu",
                save_cache=True,
            )

        # Should have processed particles
        assert isinstance(result, dict)
        assert len(result) > 0

        # Check that cache files were created for each particle
        for particle_name in result.keys():
            if not result[particle_name].empty:
                particle_dir = tmp_path / str(particle_name)
                assert particle_dir.exists()
                cache_files = list(particle_dir.glob("*.feather"))
                assert len(cache_files) > 0

    def test_get_particle_df_from_hdf5_file_all_requires_params(
        self, particle_tracker, sample_df_dict
    ):
        """Test that 'all' mode with save_cache requires hdf5_file_path and simu_name"""
        with pytest.raises(ValueError, match="hdf5_file_path and simu_name are required"):
            particle_tracker.get_particle_df_from_hdf5_file(sample_df_dict, "all", save_cache=True)

    def test_read_write_progress_file(self, particle_tracker, mock_config, tmp_path):
        """Test progress file reading and writing"""
        mock_config.particle_df_cache_dir_of["test_simu"] = str(tmp_path)

        # Initially should return -1.0
        progress = particle_tracker._read_progress_file("test_simu")
        assert progress == -1.0

        # Write progress
        particle_tracker._write_progress_file("test_simu", 123.45)

        # Read it back
        progress = particle_tracker._read_progress_file("test_simu")
        assert progress == 123.45

        # Update progress
        particle_tracker._write_progress_file("test_simu", 456.78)
        progress = particle_tracker._read_progress_file("test_simu")
        assert progress == 456.78

    def test_merge_and_cleanup_particle_cache(self, particle_tracker, mock_config, tmp_path):
        """Test merging and cleanup of particle cache files"""
        mock_config.particle_df_cache_dir_of["test_simu"] = str(tmp_path)

        # Create some temporary cache files for particle 1000
        particle_dir = tmp_path / "1000"
        particle_dir.mkdir()

        # Create temporary cache files
        df1 = pd.DataFrame({"Name": [1000], "TTOT": [1.0], "M": [10.0]})
        df2 = pd.DataFrame({"Name": [1000], "TTOT": [2.0], "M": [9.5]})
        df3 = pd.DataFrame({"Name": [1000], "TTOT": [3.0], "M": [9.0]})

        df1.to_feather(particle_dir / "1.000000.1000.df.feather")
        df2.to_feather(particle_dir / "2.000000.1000.df.feather")
        df3.to_feather(particle_dir / "3.000000.1000.df.feather")

        # Run merge
        particle_tracker._merge_and_cleanup_particle_cache("test_simu")

        # Check that merged file exists
        merged_files = list(particle_dir.glob("1000_history_until_*.df.feather"))
        assert len(merged_files) == 1

        # Check merged file content
        merged_df = pd.read_feather(merged_files[0])
        assert len(merged_df) == 3
        assert list(merged_df["TTOT"]) == [1.0, 2.0, 3.0]

        # Check that temporary files were deleted
        temp_files = list(particle_dir.glob("*.000000.1000.df.feather"))
        assert len(temp_files) == 0

    def test_merge_with_existing_merged_file(self, particle_tracker, mock_config, tmp_path):
        """Test merging when an old merged file already exists"""
        mock_config.particle_df_cache_dir_of["test_simu"] = str(tmp_path)

        particle_dir = tmp_path / "2000"
        particle_dir.mkdir()

        # Create an existing merged file
        old_df = pd.DataFrame({"Name": [2000], "TTOT": [0.5], "M": [15.0]})
        old_df.to_feather(particle_dir / "2000_history_until_0.50.df.feather")

        # Create new temporary files
        new_df = pd.DataFrame({"Name": [2000], "TTOT": [1.5], "M": [14.5]})
        new_df.to_feather(particle_dir / "1.500000.2000.df.feather")

        # Run merge
        particle_tracker._merge_and_cleanup_particle_cache("test_simu")

        # Should have one merged file
        merged_files = list(particle_dir.glob("2000_history_until_*.df.feather"))
        assert len(merged_files) == 1

        # Should contain both old and new data
        merged_df = pd.read_feather(merged_files[0])
        assert len(merged_df) == 2
        assert set(merged_df["TTOT"]) == {0.5, 1.5}

        # Old merged file should be deleted
        assert not (particle_dir / "2000_history_until_0.50.df.feather").exists()

    def test_update_one_particle_reads_merged_cache(self, particle_tracker, mock_config, tmp_path):
        """Test that update_one_particle_history_df prioritizes merged cache format"""
        mock_config.particle_df_cache_dir_of["test_simu"] = str(tmp_path)

        # Create merged cache file
        particle_dir = tmp_path / "3000"
        particle_dir.mkdir()

        cached_df = pd.DataFrame(
            {
                "Name": [3000, 3000],
                "TTOT": [1.0, 2.0],
                "M": [20.0, 19.5],
            }
        )
        cached_df.to_feather(particle_dir / "3000_history_until_2.00.df.feather")

        # Read with update=False
        result = particle_tracker.update_one_particle_history_df("test_simu", 3000, update=False)

        assert len(result) == 2
        assert list(result["TTOT"]) == [1.0, 2.0]
        assert list(result["Name"]) == [3000, 3000]


class TestPhysics:
    """Tests for physics functions"""

    def test_tau_gw_basic(self):
        """Test tau_gw with basic float inputs"""
        # Using simple values
        a = 1e10  # 10 Gm semi-major axis
        e = 0.0  # circular orbit
        mu = 1e30  # ~1 solar mass
        M = 2e30  # ~2 solar masses

        result = tau_gw(a, e, mu, M)

        # Should return a positive time
        assert result > 0
        assert isinstance(result, float)

    def test_tau_gw_eccentric(self):
        """Test tau_gw with eccentric orbit"""
        a = 1e10
        e = 0.5  # eccentric
        mu = 1e30
        M = 2e30

        result_eccentric = tau_gw(a, e, mu, M)
        result_circular = tau_gw(a, 0.0, mu, M)

        # Eccentric orbit should merge faster (smaller tau)
        assert result_eccentric < result_circular

    def test_tau_gw_with_astropy_units(self):
        """Test tau_gw with astropy Quantity inputs"""
        from astropy import units as u

        a = 1e10 * u.m
        e = 0.0
        mu = 1e30 * u.kg
        M = 2e30 * u.kg

        result = tau_gw(a, e, mu, M)

        # Should return a Quantity
        assert hasattr(result, "unit")
        assert result.value > 0

    def test_tau_gw_high_eccentricity(self):
        """Test tau_gw with high eccentricity"""
        a = 1e10
        e = 0.9  # very eccentric
        mu = 1e30
        M = 2e30

        result = tau_gw(a, e, mu, M)

        # Should still return valid positive time
        assert result > 0
        assert np.isfinite(result)
