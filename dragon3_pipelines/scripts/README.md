# Dragon3 Pipeline Scripts

This directory contains shell scripts for the dragon3_pipelines package.

## Scripts

### dragon3_jpg_to_movie.sh
Main script for creating movies from JPG snapshots.

```bash
# Create movies from all preset plot patterns
bash dragon3_jpg_to_movie.sh

# Create movies for one or more selected/custom plot patterns
bash dragon3_jpg_to_movie.sh create _CMD.jpg _custom_suffix.jpg

# Show top-level help or preset plot patterns
bash dragon3_jpg_to_movie.sh --help
bash dragon3_jpg_to_movie.sh create help
```

## Backward Compatibility

For backward compatibility, these scripts are also available in the repository root directory.
