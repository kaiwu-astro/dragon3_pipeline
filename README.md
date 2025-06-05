# Pipelines of dragon 3 data analysis

## Installation

clone this repo, and install necessary python packages using

```sh
pip install -r requirements.txt
```

## Prepare
1. Open `plot_nbody.py` . Edit paths in `ConfigManager`. Most importantly:
   - `pathof`: simulation_name: path_to_simulation_directory. You can freely name the sim.
   - `input_file_path_of`: put the path of the initial condition files. Some plots uses the initial values for data processing.
   - `figname_prefix`: prefix of every figure filename. Put something to make yourself & collaborators clear
   - `plot_dir`: path to save figures. [TODO: now save all figures in a single dir; sometimes cause slow speed to load with `ls`; will seperate into subdirs later]
   - `processes_count`: how many HDF5 files do you wish to analyze at the same time? More = faster, but at the risk of blowing up your computer's memory. In my exp, 40 uses 100 GB. Also, the number should not go beyond #cores of CPU.
2. Open `dragon3_jpg_to_movie.sh`,
   - change `PLOT_DIR` in the beginning
   - change `MAX_PARALLEL_JOBS` to restrict how many movie to make at the same time. More = uses more GPU/CPU memory. In my exp 15 = 8GB.
   - change the `module load` command if you are using LMOD (e.g., on a supercomputer), to load `FFmpeg`. Video-making would be much faster if `FFmpeg` is compiled with GPU acclerations (e.g., for NVIDIA GPUs, check with `ffmpeg -encoders | grep nvenc` to see if `hevc_nvenc` is there). In case of no GPU support, uncomment line #88 and comment #89, to use the pure CPU command.

## Usage
1. `python3 plot_nbody.py` . Optional param: `python3 plot_nbody.py --skip-until=[number or 'last']`. See comments in `ConfigManager._parse_argv`.
2. after all plots are done, `bash dragon3_jpg_to_movie.sh` to make movies. 

# Structure:
1. `plot_nbody.py` depend on `nbody_tools.py` and imports everything from it.
2. `_dragon3_jpg_to_movie_serial.sh` is a serial version of `dragon3_jpg_to_movie.sh`. It is slow (can take 3 hours to finish all movies as of 5 June 2025), and is mainly for debug.

Feel free to ask any question by opening an issue thread. 