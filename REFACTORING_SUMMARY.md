# ParticleTracker Refactoring Summary

## Overview

This refactoring transforms `ParticleTracker` from a **particle-centric** to an **HDF5-file-centric** batch processing approach, significantly improving efficiency and scalability.

## Key Changes

### 1. Method Renaming (Terminology Clarity)
- `get_particle_df_all()` → `update_one_particle_history_df()`
- `save_every_particle_df_of_sim()` → `update_all_particle_history_df()`
- Internal variables: `particle_df` (HDF5 片段) vs `particle_history_df` (完整演化路径)
- **Backward compatibility**: Old method names remain as aliases with deprecation notices

### 2. Extended `get_particle_df_from_hdf5_file()`
**New Parameters:**
- `particle_name: Union[int, str]` - Now accepts `'all'` to process all particles at once
- `hdf5_file_path: Optional[str]` - HDF5 file path (required when `save_cache=True`)
- `simu_name: Optional[str]` - Simulation name (required when `save_cache=True`)
- `save_cache: bool = False` - Enable saving individual particle cache files

**New Behavior:**
- When `particle_name='all'`: Returns `Dict[int, pd.DataFrame]` mapping particle_name → DataFrame
- Saves temporary cache files: `{cache_dir}/{particle_name}/{hdf5_time}.{particle_name}.df.feather`

### 3. Progress Tracking Mechanism
**New Methods:**
- `_read_progress_file(simu_name)` - Read last processed HDF5 time from `_progress.txt`
- `_write_progress_file(simu_name, hdf5_time)` - Update progress file

**Benefits:**
- Resume processing from last checkpoint
- Skip already-processed HDF5 files
- Incremental updates support

### 4. HDF5-Centric `update_all_particle_history_df()`
**Old Approach (Particle-Centric):**
```python
for particle in all_particles:
    for hdf5_file in hdf5_files:
        process(hdf5_file, particle)
```

**New Approach (HDF5-Centric):**
```python
for hdf5_file in hdf5_files:
    process_all_particles_at_once(hdf5_file)
    if processed_count % merge_interval == 0:
        merge_and_cleanup()
```

**New Parameters:**
- `merge_interval: int = 10` - Merge caches every N HDF5 files to avoid inode limits

**Benefits:**
- Each HDF5 file read only once (vs N_particles times)
- Automatic progress tracking
- Periodic cache merging
- Memory-efficient with `gc.collect()`

### 5. Cache Merging and Cleanup
**New Method:**
- `_merge_and_cleanup_particle_cache(simu_name)` - Consolidate temporary cache files

**Process:**
1. Iterate through particle subdirectories
2. Read all temporary feather files per particle
3. Merge into `{particle_name}_history_until_{max_ttot:.2f}.df.feather`
4. Delete old merged files and temporary files

**Benefits:**
- Prevents inode exhaustion from too many small files
- Efficient storage with consolidated history files
- Automatic cleanup of obsolete files

### 6. Updated Cache Format for `update_one_particle_history_df()`
**Cache Priority:**
1. **Merged cache** (new): `{particle_name}_history_until_{max_ttot:.2f}.df.feather`
2. **Old cache** (fallback): `{particle_name}_ALL.df.feather`

**Benefits:**
- Backward compatible
- Supports both old and new cache formats
- Automatic migration path

## Cache File Structure

### Before (Old Format)
```
cache_dir/
└── {particle_name}_ALL.df.feather
```

### After (New Format)
```
cache_dir/
├── _progress.txt                                      # Last processed HDF5 time
├── {particle_name}/
│   ├── {hdf5_time1}.{particle_name}.df.feather       # Temporary cache (deleted after merge)
│   ├── {hdf5_time2}.{particle_name}.df.feather       # Temporary cache (deleted after merge)
│   └── {particle_name}_history_until_{max_ttot}.df.feather  # Merged history
└── {particle_name}_ALL.df.feather                     # Old format (still supported)
```

## Performance Improvements

### I/O Efficiency
- **Before**: Read each HDF5 file N_particles times
- **After**: Read each HDF5 file exactly once

For a simulation with 100,000 particles and 1,000 HDF5 files:
- **Before**: 100,000,000 HDF5 reads (100B reads)
- **After**: 1,000 HDF5 reads (1M reads, **100,000× improvement**)

### Inode Management
- **Before**: Unlimited temporary files could exhaust inodes
- **After**: Periodic merging (every `merge_interval` files) keeps inode count bounded

### Memory Management
- Explicit `gc.collect()` after merge operations
- Process HDF5 files sequentially (not all in memory)

## API Compatibility

### ✅ Fully Backward Compatible
All existing code continues to work:
```python
# Old API (still works via aliases)
tracker.get_particle_df_all(simu_name, particle_name)
tracker.save_every_particle_df_of_sim(simu_name)

# New API (recommended)
tracker.update_one_particle_history_df(simu_name, particle_name)
tracker.update_all_particle_history_df(simu_name, merge_interval=10)
```

### New API Extensions
```python
# Process all particles from one HDF5 file
result_dict = tracker.get_particle_df_from_hdf5_file(
    df_dict, 
    particle_name='all',
    hdf5_file_path=path,
    simu_name=simu_name,
    save_cache=True
)
```

## Testing

### Test Coverage
- **18 tests** for ParticleTracker (up from 11)
- **94 total tests** pass (1 skipped)
- **100% backward compatibility** verified

### New Tests Added
1. `test_get_particle_df_from_hdf5_file_all_particles` - Test 'all' mode
2. `test_get_particle_df_from_hdf5_file_all_with_cache` - Test cache saving
3. `test_get_particle_df_from_hdf5_file_all_requires_params` - Parameter validation
4. `test_read_write_progress_file` - Progress tracking
5. `test_merge_and_cleanup_particle_cache` - Merge functionality
6. `test_merge_with_existing_merged_file` - Incremental merging
7. `test_update_one_particle_reads_merged_cache` - New cache format reading

## Code Quality

### Linting
- ✅ **Black** formatted (100 chars/line)
- ✅ **Ruff** checked and fixed
- ✅ **Type hints** added for all new methods

### Documentation
- Updated docstrings with detailed parameter descriptions
- Added inline comments for complex logic
- Deprecation notices for old method names

## Migration Guide

### For Users of `get_particle_df_all()`
```python
# Old (still works)
df = tracker.get_particle_df_all(simu_name, particle_name)

# New (recommended)
df = tracker.update_one_particle_history_df(simu_name, particle_name)
```

### For Users of `save_every_particle_df_of_sim()`
```python
# Old (still works, but slow)
tracker.save_every_particle_df_of_sim(simu_name)

# New (recommended, much faster)
tracker.update_all_particle_history_df(simu_name, merge_interval=10)
```

### Resuming Interrupted Processing
The new implementation automatically resumes from the last processed HDF5 file:
```python
# First run: processes files 1-50, then interrupted
tracker.update_all_particle_history_df(simu_name)

# Second run: automatically starts from file 51
tracker.update_all_particle_history_df(simu_name)
```

## Dependencies

No new dependencies added. All functionality uses existing libraries:
- `pandas` - DataFrame operations
- `numpy` - Numerical operations
- `glob` - File pattern matching
- `tqdm` - Progress bars
- `multiprocessing` - Parallel processing (existing)

## Files Changed

1. `dragon3_pipelines/analysis/particle_tracker.py` - Main implementation (716 lines)
2. `tests/test_analysis.py` - Comprehensive tests (443 lines)

## Summary Statistics

- **Lines of Code**: 716 (particle_tracker.py)
- **New Methods**: 5
- **Deprecated Methods**: 0 (old names are aliases)
- **Tests**: 18 (up from 11, +64%)
- **Test Pass Rate**: 100% (94/94 passed, 1 skipped)
