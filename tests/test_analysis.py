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
        config.mem_cap_bytes = 40 * 1024**3 // 2  # 20 GB
        config.inode_limit = 2000000
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

    @patch("dragon3_pipelines.analysis.particle_tracker.glob")
    @patch("os.path.exists")
    def test_update_one_particle_history_df_no_cache(
        self, mock_exists, mock_glob, particle_tracker, mock_config
    ):
        """Test getting particle data with no existing cache"""
        mock_exists.return_value = False
        mock_glob.return_value = []

        result = particle_tracker.update_one_particle_history_df("test_simu", 1000, update=False)

        assert result.empty

    @patch("dragon3_pipelines.analysis.particle_tracker.glob")
    @patch("pandas.read_feather")
    @patch("os.path.exists")
    def test_update_one_particle_history_df_with_cache(
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

        result = particle_tracker.update_one_particle_history_df("test_simu", 1000, update=False)

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

    def test_build_progress_dict_no_files(self, particle_tracker, mock_config, tmp_path):
        """Test _build_progress_dict with no existing files"""
        mock_config.particle_df_cache_dir_of["test_simu"] = str(tmp_path)

        result = particle_tracker._build_progress_dict("test_simu", [1000, 2000, 3000])

        assert result == {1000: -1.0, 2000: -1.0, 3000: -1.0}

    def test_build_progress_dict_with_history_files(self, particle_tracker, mock_config, tmp_path):
        """Test _build_progress_dict with existing history files"""
        mock_config.particle_df_cache_dir_of["test_simu"] = str(tmp_path)

        # Create history files for particles
        particle_1_dir = tmp_path / "1000"
        particle_1_dir.mkdir()
        (particle_1_dir / "1000_history_until_5.00.df.feather").touch()

        particle_2_dir = tmp_path / "2000"
        particle_2_dir.mkdir()
        (particle_2_dir / "2000_history_until_10.50.df.feather").touch()

        result = particle_tracker._build_progress_dict("test_simu", [1000, 2000, 3000])

        assert result[1000] == 5.0
        assert result[2000] == 10.5
        assert result[3000] == -1.0  # No file for this particle

    def test_build_progress_dict_multiple_files_uses_max(
        self, particle_tracker, mock_config, tmp_path
    ):
        """Test _build_progress_dict with multiple history files takes max timestamp"""
        mock_config.particle_df_cache_dir_of["test_simu"] = str(tmp_path)

        # Create multiple history files for one particle
        particle_dir = tmp_path / "1000"
        particle_dir.mkdir()
        (particle_dir / "1000_history_until_3.00.df.feather").touch()
        (particle_dir / "1000_history_until_7.50.df.feather").touch()
        (particle_dir / "1000_history_until_5.00.df.feather").touch()

        result = particle_tracker._build_progress_dict("test_simu", [1000])

        # Should use max timestamp (7.50)
        assert result[1000] == 7.5

    def test_read_history_with_direct_path(self, particle_tracker, tmp_path):
        """Test read_history with direct feather path"""
        # Create a test feather file
        test_df = pd.DataFrame(
            {
                "Name": [1000, 1000],
                "TTOT": [1.0, 2.0],
                "M": [10.0, 9.5],
            }
        )
        feather_path = tmp_path / "test_history.feather"
        test_df.to_feather(feather_path)

        result = particle_tracker.read_history(feather_path=str(feather_path))

        assert len(result) == 2
        assert list(result["Name"]) == [1000, 1000]
        assert list(result["TTOT"]) == [1.0, 2.0]

    def test_read_history_with_simu_and_particle_name(
        self, particle_tracker, mock_config, tmp_path
    ):
        """Test read_history with simulation name and particle name"""
        mock_config.particle_df_cache_dir_of["test_simu"] = str(tmp_path)

        # Create particle directory and history file
        particle_dir = tmp_path / "5000"
        particle_dir.mkdir()

        test_df = pd.DataFrame(
            {
                "Name": [5000, 5000, 5000],
                "TTOT": [0.0, 1.0, 2.0],
                "M": [15.0, 14.5, 14.0],
            }
        )
        test_df.to_feather(particle_dir / "5000_history_until_2.00.df.feather")

        result = particle_tracker.read_history(simu_name="test_simu", particle_name=5000)

        assert len(result) == 3
        assert result["Name"].iloc[0] == 5000
        assert list(result["TTOT"]) == [0.0, 1.0, 2.0]

    def test_read_history_missing_file(self, particle_tracker):
        """Test read_history returns empty DataFrame for missing file"""
        result = particle_tracker.read_history(feather_path="/nonexistent/path.feather")
        assert result.empty

    def test_read_history_missing_params_raises_error(self, particle_tracker):
        """Test read_history raises ValueError when required params missing"""
        with pytest.raises(ValueError, match="Either feather_path or both simu_name"):
            particle_tracker.read_history()

    def test_read_history_partial_params_raises_error(self, particle_tracker):
        """Test read_history raises ValueError when only simu_name is provided"""
        with pytest.raises(ValueError, match="Either feather_path or both simu_name"):
            particle_tracker.read_history(simu_name="test_simu")

    def test_read_history_picks_latest_file(self, particle_tracker, mock_config, tmp_path):
        """Test read_history picks the latest history file"""
        mock_config.particle_df_cache_dir_of["test_simu"] = str(tmp_path)

        particle_dir = tmp_path / "6000"
        particle_dir.mkdir()

        # Create multiple history files with different timestamps
        df1 = pd.DataFrame({"Name": [6000], "TTOT": [5.0]})
        df2 = pd.DataFrame({"Name": [6000, 6000], "TTOT": [5.0, 10.0]})

        df1.to_feather(particle_dir / "6000_history_until_5.00.df.feather")
        df2.to_feather(particle_dir / "6000_history_until_10.00.df.feather")

        result = particle_tracker.read_history(simu_name="test_simu", particle_name=6000)

        # Should read the file with the higher timestamp (10.00)
        assert len(result) == 2
        assert result["TTOT"].max() == 10.0


class TestHDF5ParticleTask:
    """Tests for HDF5ParticleTask dataclass"""

    def test_dataclass_creation(self):
        """Test HDF5ParticleTask dataclass can be created"""
        from dragon3_pipelines.analysis.particle_tracker import HDF5ParticleTask

        task = HDF5ParticleTask(
            hdf5_file_path="/path/to/file.h5part",
            simu_name="test_simu",
            particle_names=[1000, 2000, 3000],
            progress_dict={1000: 5.0, 2000: 10.0, 3000: -1.0},
        )

        assert task.hdf5_file_path == "/path/to/file.h5part"
        assert task.simu_name == "test_simu"
        assert task.particle_names == [1000, 2000, 3000]
        assert task.progress_dict == {1000: 5.0, 2000: 10.0, 3000: -1.0}


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


class TestBinaryOrbitFunctions:
    """Tests for binary orbit calculation functions"""

    def test_compute_binary_orbit_relative_positions_equal_masses(self):
        """Test compute_binary_orbit_relative_positions with equal masses"""
        from dragon3_pipelines.analysis import compute_binary_orbit_relative_positions

        m1, m2 = 10.0, 10.0
        rel_x, rel_y, rel_z = 2.0, 0.0, 0.0

        (x1, y1, z1), (x2, y2, z2) = compute_binary_orbit_relative_positions(
            m1, m2, rel_x, rel_y, rel_z
        )

        # Equal masses: positions should be equal and opposite
        assert x1 == pytest.approx(-1.0)
        assert x2 == pytest.approx(1.0)
        assert y1 == 0.0
        assert y2 == 0.0
        assert z1 == 0.0
        assert z2 == 0.0

    def test_compute_binary_orbit_relative_positions_unequal_masses(self):
        """Test compute_binary_orbit_relative_positions with unequal masses"""
        from dragon3_pipelines.analysis import compute_binary_orbit_relative_positions

        m1, m2 = 30.0, 10.0  # 3:1 mass ratio
        rel_x, rel_y, rel_z = 4.0, 0.0, 0.0

        (x1, y1, z1), (x2, y2, z2) = compute_binary_orbit_relative_positions(
            m1, m2, rel_x, rel_y, rel_z
        )

        # r1 = -m2/(m1+m2) * rel = -10/40 * 4 = -1
        # r2 = m1/(m1+m2) * rel = 30/40 * 4 = 3
        assert x1 == pytest.approx(-1.0)
        assert x2 == pytest.approx(3.0)

    def test_compute_binary_orbit_relative_positions_zero_mass(self):
        """Test compute_binary_orbit_relative_positions with zero total mass"""
        from dragon3_pipelines.analysis import compute_binary_orbit_relative_positions

        m1, m2 = 0.0, 0.0
        rel_x, rel_y, rel_z = 2.0, 1.0, 0.5

        (x1, y1, z1), (x2, y2, z2) = compute_binary_orbit_relative_positions(
            m1, m2, rel_x, rel_y, rel_z
        )

        # Zero mass should return zero positions
        assert (x1, y1, z1) == (0.0, 0.0, 0.0)
        assert (x2, y2, z2) == (0.0, 0.0, 0.0)

    def test_compute_individual_orbit_params_equal_masses(self):
        """Test compute_individual_orbit_params with equal masses"""
        from dragon3_pipelines.analysis import compute_individual_orbit_params

        a_bin, ecc_bin = 10.0, 0.5
        m1, m2 = 10.0, 10.0

        (a1, e1), (a2, e2) = compute_individual_orbit_params(a_bin, ecc_bin, m1, m2)

        # Equal masses: each star has half the semi-major axis
        assert a1 == pytest.approx(5.0)
        assert a2 == pytest.approx(5.0)
        assert e1 == pytest.approx(0.5)
        assert e2 == pytest.approx(0.5)

    def test_compute_individual_orbit_params_unequal_masses(self):
        """Test compute_individual_orbit_params with unequal masses"""
        from dragon3_pipelines.analysis import compute_individual_orbit_params

        a_bin, ecc_bin = 100.0, 0.3
        m1, m2 = 30.0, 10.0  # 3:1 mass ratio

        (a1, e1), (a2, e2) = compute_individual_orbit_params(a_bin, ecc_bin, m1, m2)

        # a1 = a * m2/(m1+m2) = 100 * 10/40 = 25
        # a2 = a * m1/(m1+m2) = 100 * 30/40 = 75
        assert a1 == pytest.approx(25.0)
        assert a2 == pytest.approx(75.0)
        assert e1 == pytest.approx(0.3)
        assert e2 == pytest.approx(0.3)

    def test_compute_individual_orbit_params_zero_mass(self):
        """Test compute_individual_orbit_params with zero total mass"""
        from dragon3_pipelines.analysis import compute_individual_orbit_params

        a_bin, ecc_bin = 10.0, 0.5
        m1, m2 = 0.0, 0.0

        (a1, e1), (a2, e2) = compute_individual_orbit_params(a_bin, ecc_bin, m1, m2)

        # Zero mass should return zero semi-major axes
        assert a1 == 0.0
        assert a2 == 0.0
        assert e1 == pytest.approx(0.5)
        assert e2 == pytest.approx(0.5)
