"""
Test backward compatibility of plot_nbody.py and nbody_tools.py wrappers
"""
import warnings
import pytest


class TestPlotNbodyBackwardCompatibility:
    """Test that plot_nbody.py wrapper maintains backward compatibility"""
    
    def test_import_plot_nbody(self):
        """Test that plot_nbody can be imported"""
        import plot_nbody
        assert plot_nbody is not None
    
    def test_plot_nbody_has_main(self):
        """Test that plot_nbody exports main function"""
        import plot_nbody
        assert hasattr(plot_nbody, 'main')
        assert callable(plot_nbody.main)
    
    def test_plot_nbody_has_simulation_plotter(self):
        """Test that plot_nbody exports SimulationPlotter class"""
        import plot_nbody
        assert hasattr(plot_nbody, 'SimulationPlotter')
        assert isinstance(plot_nbody.SimulationPlotter, type)
    
    def test_plot_nbody_imports_same_as_new(self):
        """Test that plot_nbody imports point to new package"""
        import plot_nbody
        from dragon3_pipelines import main, SimulationPlotter
        
        assert plot_nbody.main is main
        assert plot_nbody.SimulationPlotter is SimulationPlotter


class TestNbodyToolsBackwardCompatibility:
    """Test that nbody_tools.py wrapper maintains backward compatibility"""
    
    def test_import_nbody_tools_shows_deprecation(self):
        """Test that nbody_tools shows deprecation warning"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import nbody_tools
            
            # Check that a deprecation warning was issued
            assert any(issubclass(warn.category, DeprecationWarning) for warn in w)
            assert any("deprecated" in str(warn.message).lower() for warn in w)
    
    def test_nbody_tools_utility_imports(self):
        """Test that utility functions can be imported from nbody_tools"""
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            from nbody_tools import save, read, get_output, can_convert_to_float
            
            assert callable(save)
            assert callable(read)
            assert callable(get_output)
            assert callable(can_convert_to_float)
    
    def test_nbody_tools_io_imports(self):
        """Test that I/O classes can be imported from nbody_tools"""
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            from nbody_tools import HDF5FileProcessor, LagrFileProcessor
            
            assert isinstance(HDF5FileProcessor, type)
            assert isinstance(LagrFileProcessor, type)
    
    def test_nbody_tools_config_imports(self):
        """Test that config classes can be imported from nbody_tools"""
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            from nbody_tools import ConfigManager, load_config
            
            assert isinstance(ConfigManager, type)
            assert callable(load_config)
    
    def test_nbody_tools_visualization_imports(self):
        """Test that visualization classes can be imported from nbody_tools"""
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            from nbody_tools import (
                BaseVisualizer, HDF5Visualizer, 
                SingleStarVisualizer, BinaryStarVisualizer
            )
            
            assert isinstance(BaseVisualizer, type)
            assert isinstance(HDF5Visualizer, type)
            assert isinstance(SingleStarVisualizer, type)
            assert isinstance(BinaryStarVisualizer, type)
    
    def test_nbody_tools_analysis_imports(self):
        """Test that analysis classes can be imported from nbody_tools"""
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            from nbody_tools import ParticleTracker
            
            assert isinstance(ParticleTracker, type)
    
    def test_nbody_tools_imports_same_as_new(self):
        """Test that nbody_tools imports point to new package"""
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            from nbody_tools import HDF5FileProcessor as OldHDF5
            from dragon3_pipelines.io import HDF5FileProcessor as NewHDF5
            
            assert OldHDF5 is NewHDF5
    
    def test_nbody_tools_constants(self):
        """Test that constants are available from nbody_tools"""
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            from nbody_tools import pc_to_AU
            
            assert isinstance(pc_to_AU, float)
            assert pc_to_AU > 0


class TestNewPackageImports:
    """Test that new package imports work correctly"""
    
    def test_import_main_from_package(self):
        """Test importing main from dragon3_pipelines"""
        from dragon3_pipelines import main
        assert callable(main)
    
    def test_import_simulation_plotter_from_package(self):
        """Test importing SimulationPlotter from dragon3_pipelines"""
        from dragon3_pipelines import SimulationPlotter
        assert isinstance(SimulationPlotter, type)
    
    def test_import_from_main_module(self):
        """Test importing from __main__ module"""
        from dragon3_pipelines.__main__ import main, SimulationPlotter
        assert callable(main)
        assert isinstance(SimulationPlotter, type)
