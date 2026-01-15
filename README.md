# Dragon3 Pipelines - N-body Simulation Data Analysis

A modular Python package for analyzing and visualizing N-body simulation data from Dragon3 simulations.

## Installation

### For Users

Clone this repository and install the package:

```bash
git clone https://github.com/kaiwu-astro/dragon3_pipeline.git
cd dragon3_pipeline
pip install -e .
```

### For Developers

Install with development dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Using the Command Line

```bash
# Run with default configuration
python -m dragon3_pipelines

# Or use the legacy script (backward compatible)
python plot_nbody.py --skip-until=100

# Create movies from plots
bash dragon3_jpg_to_movie.sh
```

### Using as a Python Package

```python
from dragon3_pipelines import main, SimulationPlotter
from dragon3_pipelines.config import ConfigManager
from dragon3_pipelines.io import HDF5FileProcessor
from dragon3_pipelines.analysis import ParticleTracker
from dragon3_pipelines.visualization import SingleStarVisualizer

# Create custom configuration
config = ConfigManager()
config.processes_count = 20

# Use visualizers
visualizer = SingleStarVisualizer(hdf5_processor, config)
```

## Configuration

### Using YAML Configuration

Create a custom `config.yaml`:

```yaml
paths:
  simulations:
    my_sim: "/path/to/simulation"
  plot_dir: "/path/to/plots"

processing:
  processes_count: 40
  skip_existing_plot: true
```

Load it in your code:

```python
from dragon3_pipelines.config import load_config
config = load_config("config.yaml")
```

### Legacy Configuration

For backward compatibility, you can still edit paths directly in `plot_nbody.py` as before.

## Package Structure

```
dragon3_pipelines/
├── config/          # Configuration management
├── io/              # Data I/O (HDF5, text files)
├── analysis/        # Particle tracking, physics calculations
├── visualization/   # Plotting and visualization
├── utils/           # Utility functions
└── scripts/         # Shell scripts for movie generation
```

## Features

- **Modular Design**: Clean separation of I/O, analysis, and visualization
- **Type Annotations**: Full type hints for better IDE support
- **Backward Compatible**: All old scripts and imports still work
- **Configurable**: YAML-based configuration with sensible defaults
- **Parallel Processing**: Multi-process support for analyzing large datasets
- **Comprehensive Testing**: 87+ unit tests covering all modules

## Usage Examples

### Analyze HDF5 Files

```python
from dragon3_pipelines.io import HDF5FileProcessor
from dragon3_pipelines.config import ConfigManager

config = ConfigManager()
processor = HDF5FileProcessor(config, "my_sim")
df_singles, df_binaries = processor.get_hdf5_dataframes()
```

### Track Particles

```python
from dragon3_pipelines.analysis import ParticleTracker

tracker = ParticleTracker(config, "my_sim")
particle_history = tracker.get_particle_new_df_all(particle_id=12345)
```

### Create Visualizations

```python
from dragon3_pipelines.visualization import BinaryStarVisualizer

viz = BinaryStarVisualizer(hdf5_processor, config)
viz.create_mass_ratio_vs_a_scatter(df_binaries, time=100.0)
```

## Command Line Options

- `--skip-until=N`: Start processing from time N
- `--skip-until=last`: Resume from last processed time
- `--debug`: Enable debug logging

## Contributing

Feel free to open issues or submit pull requests!

## License

See LICENSE file for details. 