"""
Configuration management for dragon3_pipelines
"""
import os
import yaml
import numpy as np
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from glob import glob
import logging

from dragon3_pipelines.utils import can_convert_to_float

logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None) -> 'ConfigManager':
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to user config file. If None, uses default config only.
        
    Returns:
        ConfigManager instance with loaded configuration
    """
    return ConfigManager(config_path=config_path)


class ConfigManager:
    """
    Configuration manager for dragon3_pipelines.
    
    Loads default configuration and optionally merges with user configuration.
    """
    
    def __init__(
        self, 
        config_path: Optional[str] = None,
        opts: List[Tuple[str, str]] = None
    ):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Optional path to user configuration file
            opts: Optional list of command-line options as (option, argument) tuples
        """
        self._load_default_config()
        if config_path and os.path.exists(config_path):
            self._merge_user_config(config_path)
        
        # Initialize derived attributes
        self._setup_derived_attributes()
        
        # Parse command line arguments if provided
        if opts:
            self._parse_argv(opts)
        
        # Handle 'last' in skip_until
        self._resolve_skip_until_last()
    
    def _load_default_config(self) -> None:
        """Load default configuration from package"""
        config_dir = Path(__file__).parent
        default_config_path = config_dir / "default_config.yaml"
        
        with open(default_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Path configurations
        paths = config['paths']
        self.pathof: Dict[str, str] = paths['simulations']
        self.plot_dir: str = paths['plot_dir']
        
        # Build derived paths
        self.particle_df_cache_dir_of = {
            k: v + paths['cache_dir_suffix'] 
            for k, v in self.pathof.items()
        }
        self.input_file_path_of = {
            k: self.pathof[k] + '/' + paths['input_files'][k]
            for k in self.pathof.keys()
        }
        
        # Figure name prefixes
        self.figname_prefix: Dict[str, str] = config['figure_prefixes']
        
        # Processing options
        proc = config['processing']
        self.skip_until_of: Dict[str, Union[float, str]] = proc['skip_until']
        self.skip_existing_plot: bool = proc['skip_existing_plot']
        self.plot_only_int_nbody_time: bool = proc['plot_only_int_nbody_time']
        self.close_figure_in_ipython: bool = proc['close_figure_in_ipython']
        self.processes_count: int = proc['processes_count']
        self.tasks_per_child: int = proc['tasks_per_child']
        
        # Lagrangian radii
        self.selected_lagr_percent: List[str] = config['selected_lagr_percent']
        
        # Physics constants
        phys = config['physics']
        self.ECLOSE_INPUT: float = phys['ECLOSE_INPUT']
        self.universe_age_myr: float = phys['universe_age_myr']
        self.IMBH_mass_range_msun: Tuple[float, float] = tuple(phys['IMBH_mass_range_msun'])
        self.extreme_mass_ratio_upper: float = phys['extreme_mass_ratio_upper']
        self.high_BH_ecc_lower: float = phys['high_BH_ecc_lower']
        self.PISNe_mass_gap: Tuple[int, int] = tuple(phys['PISNe_mass_gap'])
        
        # Stellar types
        self.kw_to_stellar_type: Dict[int, str] = {
            int(k): v for k, v in config['stellar_types'].items()
        }
        self.compact_object_KW: np.ndarray = np.array(config['compact_object_KW'])
        
        # Binary types
        self.wow_binary_st_list: List[str] = config['wow_binary_st_list']
        
        # Limits
        self.limits: Dict[str, Tuple[float, float]] = {
            k.replace('_', ' ').replace('[', ' [').replace(']', ']'): tuple(v)
            for k, v in config['limits'].items()
        }
        
        # Column labels
        self.colname_to_label: Dict[str, str] = {
            k.replace('_', ' ').replace('[', ' [').replace(']', ']'): v
            for k, v in config['column_labels'].items()
        }
        
        # Fixed width font context
        self.fixed_width_font_context: Dict[str, Any] = {
            'rc': {'font.family': 'monospace'}
        }
    
    def _merge_user_config(self, config_path: str) -> None:
        """
        Merge user configuration with defaults.
        
        Args:
            config_path: Path to user YAML configuration file
        """
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
        
        # Simple deep merge - can be extended for more complex merging
        if 'paths' in user_config:
            if 'simulations' in user_config['paths']:
                self.pathof.update(user_config['paths']['simulations'])
            if 'plot_dir' in user_config['paths']:
                self.plot_dir = user_config['paths']['plot_dir']
        
        if 'processing' in user_config:
            proc = user_config['processing']
            if 'processes_count' in proc:
                self.processes_count = proc['processes_count']
            if 'skip_until' in proc:
                self.skip_until_of.update(proc['skip_until'])
    
    def _setup_derived_attributes(self) -> None:
        """Set up derived attributes that depend on configuration"""
        # Reverse mappings
        self.stellar_type_to_kw: Dict[str, int] = {
            v: k for k, v in self.kw_to_stellar_type.items()
        }
        
        self.kw_to_stellar_type_verbose: Dict[int, str] = {
            k: f"{k:2d}:{v}" for k, v in self.kw_to_stellar_type.items()
        }
        
        self.stellar_type_verbose_to_kw: Dict[str, int] = {
            v: k for k, v in self.kw_to_stellar_type_verbose.items()
        }
        
        # Plotting styles
        _marker_fill_list = ['d', 'v', '^', '<', '>', 'h', '8', 's', 'p', 'H', 'D', 'o']
        self.marker_fill_list = (
            ['o',] * (17 - len(_marker_fill_list)) + _marker_fill_list
        )
        self.marker_nofill_list = (
            (['1', '+', '3', 'x'] * 5)[:len(self.kw_to_stellar_type)]
        )
        
        self.star_type_verbose_to_marker: Dict[str, str] = dict(zip(
            list(self.kw_to_stellar_type_verbose.values()), 
            self.marker_nofill_list
        ))
        
        # Color palettes
        self.palette_st = (
            sns.color_palette(n_colors=10) + 
            sns.color_palette("husl", 7)[::2] + 
            sns.color_palette("husl", 7)[1::2][:-3] + 
            [(1, 0, 0), (0, 0, 0), (0.8, 0.8, 0.8)]
        )
        
        self.st_verbose_to_color: Dict[str, Tuple[float, float, float]] = {
            k: self.palette_st[v+1] 
            for k, v in self.stellar_type_verbose_to_kw.items()
        }
    
    def _parse_argv(self, opts: List[Tuple[str, str]]) -> None:
        """
        Parse command line arguments and update configuration.
        
        Args:
            opts: List of (option, argument) tuples from command line parsing
            
        Example:
            --skip-until=0: start from t=0 (first file)
            --skip-until=last: read last timestamp from existing plots
        """
        for opt, arg in opts:
            if opt == '--skip-until':
                if can_convert_to_float(arg):
                    arg = float(arg)
                for key in self.skip_until_of:
                    self.skip_until_of[key] = arg
    
    def _resolve_skip_until_last(self) -> None:
        """
        Resolve 'last' values in skip_until by finding maximum time from existing plots.
        """
        for simu_name in self.pathof.keys():
            if self.skip_until_of.get(simu_name) == 'last':
                # Find existing plots: plot_dir/figname_prefix*ttot_*.pdf
                pattern = (
                    f"{self.plot_dir}/{self.figname_prefix[simu_name]}*ttot_*.pdf"
                )
                all_pdf_plots = glob(pattern)
                
                if all_pdf_plots:
                    # Extract time from filename
                    def get_time(path: str) -> float:
                        return float(path.split('ttot_')[1].split('_')[0])
                    
                    all_times = np.array([get_time(x) for x in all_pdf_plots])
                    self.skip_until_of[simu_name] = float(all_times.max())
                    logger.info(
                        f"[{simu_name}] Set skip-until={self.skip_until_of[simu_name]}"
                    )
                else:
                    self.skip_until_of[simu_name] = 0
