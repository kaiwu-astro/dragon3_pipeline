"""
Tests for dragon3_pipelines.config module
"""
import pytest
import tempfile
import yaml
from pathlib import Path

from dragon3_pipelines.config import ConfigManager, load_config


class TestConfigManager:
    """Tests for ConfigManager class"""
    
    def test_load_default_config(self):
        """Test loading default configuration"""
        config = ConfigManager()
        
        # Check basic attributes are loaded
        assert hasattr(config, 'pathof')
        assert hasattr(config, 'plot_dir')
        assert hasattr(config, 'figname_prefix')
        assert hasattr(config, 'processes_count')
        assert hasattr(config, 'kw_to_stellar_type')
        
        # Check specific values
        assert isinstance(config.pathof, dict)
        assert isinstance(config.processes_count, int)
        assert config.processes_count == 40
        
        # Check stellar types are loaded
        assert 14 in config.kw_to_stellar_type
        assert config.kw_to_stellar_type[14] == "BH"
        assert 13 in config.kw_to_stellar_type
        assert config.kw_to_stellar_type[13] == "NS"
    
    def test_derived_attributes(self):
        """Test that derived attributes are created correctly"""
        config = ConfigManager()
        
        # Check reverse mappings
        assert hasattr(config, 'stellar_type_to_kw')
        assert "BH" in config.stellar_type_to_kw
        assert config.stellar_type_to_kw["BH"] == 14
        
        # Check verbose mappings
        assert hasattr(config, 'kw_to_stellar_type_verbose')
        assert 14 in config.kw_to_stellar_type_verbose
        assert "14" in config.kw_to_stellar_type_verbose[14]
        assert "BH" in config.kw_to_stellar_type_verbose[14]
        
        # Check plotting attributes
        assert hasattr(config, 'palette_st')
        assert hasattr(config, 'marker_fill_list')
        assert len(config.marker_fill_list) > 0
    
    def test_load_config_function(self):
        """Test load_config convenience function"""
        config = load_config()
        assert isinstance(config, ConfigManager)
        assert hasattr(config, 'pathof')
    
    def test_user_config_merge(self, temp_dir):
        """Test merging user configuration with defaults"""
        # Create a user config file
        user_config = {
            'paths': {
                'simulations': {
                    'test_sim': '/path/to/test'
                },
                'plot_dir': '/custom/plot/dir'
            },
            'processing': {
                'processes_count': 20
            }
        }
        
        user_config_path = temp_dir / "user_config.yaml"
        with open(user_config_path, 'w') as f:
            yaml.dump(user_config, f)
        
        config = ConfigManager(config_path=str(user_config_path))
        
        # Check that user config is merged
        assert config.plot_dir == '/custom/plot/dir'
        assert config.processes_count == 20
        assert 'test_sim' in config.pathof
        
        # Check that defaults are still present
        assert '0sb' in config.pathof  # from default config
    
    def test_parse_argv_skip_until(self):
        """Test command line argument parsing"""
        config = ConfigManager()
        
        # Test with numeric value
        config._parse_argv([('--skip-until', '100')])
        for key in config.skip_until_of:
            assert config.skip_until_of[key] == 100.0
        
        # Test with string value
        config._parse_argv([('--skip-until', 'last')])
        for key in config.skip_until_of:
            assert config.skip_until_of[key] == 'last'
    
    def test_physics_constants(self):
        """Test physics constants are loaded correctly"""
        config = ConfigManager()
        
        assert config.ECLOSE_INPUT == 1.0
        assert config.universe_age_myr == 13800.0
        assert isinstance(config.IMBH_mass_range_msun, tuple)
        assert len(config.IMBH_mass_range_msun) == 2
        assert isinstance(config.PISNe_mass_gap, tuple)
        assert len(config.PISNe_mass_gap) == 2
    
    def test_limits_and_labels(self):
        """Test that limits and labels are loaded"""
        config = ConfigManager()
        
        assert hasattr(config, 'limits')
        assert isinstance(config.limits, dict)
        assert len(config.limits) > 0
        
        assert hasattr(config, 'colname_to_label')
        assert isinstance(config.colname_to_label, dict)
        assert len(config.colname_to_label) > 0
        
        # Check specific limits
        assert 'Bin A [au]' in config.limits or 'Bin A au' in config.limits
