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
2. Open `dragon3_jpg_to_movie.sh`,
   - change `PLOT_DIR` in the beginning
   - change the `module load` command if you are using LMOD (e.g., on a supercomputer), to load `FFmpeg`. Video-making would be much faster if `FFmpeg` is compiled with GPU acclerations (e.g., for NVIDIA GPUs, check with `ffmpeg -encoders | grep nvenc` to see if `hevc_nvenc` is there). In case of no GPU support, uncomment line #88 and comment #89, to use the pure CPU command.

## Usage
1. `python3 plot_nbody.py` . Optional param: `python3 plot_nbody.py --skip-until=[number or 'last']`. See comments in `ConfigManager._parse_argv`.
2. after all plots are done, `bash dragon3_jpg_to_movie.sh` to make movies. 

Feel free to ask any question by opening an issue thread. 