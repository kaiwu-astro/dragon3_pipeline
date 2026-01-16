# Dragon3 Pipelines API Reference

## Configuration

### `dragon3_pipelines.config.ConfigManager`

Main configuration class for managing simulation settings.

```python
from dragon3_pipelines.config import ConfigManager, load_config

# Load default configuration
config = ConfigManager()

# Load with custom YAML
config = load_config("my_config.yaml")
```

## I/O Operations

### `dragon3_pipelines.io.HDF5FileProcessor`

Read and process HDF5 simulation files.

### `dragon3_pipelines.io.LagrFileProcessor`

Process Lagrangian radii files.

### `dragon3_pipelines.io.text_parsers`

Functions for parsing text-based simulation output files.

## Analysis

### `dragon3_pipelines.analysis.ParticleTracker`

Track individual particles through simulation snapshots.

### `dragon3_pipelines.analysis.tau_gw`

Calculate gravitational wave merger timescales.

## Visualization

### Base Classes

- `BaseVisualizer`: Base class for all visualizers
- `BaseHDF5Visualizer`: Base for HDF5-based visualizations

### Visualizer Classes

- `SingleStarVisualizer`: Visualize single star properties
- `BinaryStarVisualizer`: Visualize binary star systems
- `LagrVisualizer`: Visualize Lagrangian radii evolution
- `CollCoalVisualizer`: Visualize collision and coalescence events

## Utilities

### `dragon3_pipelines.utils`

- `save()`, `read()`: Pickle serialization
- `get_output()`: Execute shell commands
- `log_time`: Decorator for timing functions
- `BlackbodyColorConverter`: Temperature to RGB color conversion
