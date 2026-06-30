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

### `dragon3_pipelines.analysis.BinaryStellarTypeExtractor`

Extract complete processed binary rows where either component matches a StellarType abbreviation or KW code.

```python
from dragon3_pipelines.analysis import BinaryStellarTypeExtractor
from dragon3_pipelines.config import ConfigManager

config = ConfigManager()
extractor = BinaryStellarTypeExtractor(config)
bh_binaries = extractor.load_binaries_with_stellar_type("20sb", stellar_type="BH")
ns_binaries = extractor.load_binaries_with_stellar_type("20sb", kw="13")
```

Specify exactly one of `stellar_type` or `kw`. StellarType abbreviations are matched case-insensitively using `default_config.yaml` `stellar_types`.

### `dragon3_pipelines.analysis.BTypeBinaryExtractor`

Extract complete processed binary rows where either component satisfies the project B-type main-sequence criteria: `Bin KW* == 1`, `10500 <= Bin Teff* <= 31500`, and `2.75 <= Bin M* <= 17.7`.

```python
from dragon3_pipelines.analysis import BTypeBinaryExtractor
from dragon3_pipelines.config import ConfigManager

config = ConfigManager()
df = BTypeBinaryExtractor(config).load_b_type_binaries("20sb")
```

The returned table preserves the processed binary rows and adds `b_type_member1`, `b_type_member2`, `b_type_member_count`, `b_type_pair_key`, and `is_primordial_binary`. Results are cached separately under the simulation particle cache directory.

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
