# in your shell, first run:
#   module load Stages/2025 GCC IPython && pip install -r /p/project1/madnuc/wu13/dragon3_pipelines/requirements.txt
# type ipython to start on command line. Notebook interface on Juwels is a bit complicated; ask Kai.
# below are python commands which you can explore with 

import sys; sys.path.append('/p/project1/madnuc/wu13/dragon3_pipelines'); from plot_nbody import *
config = ConfigManager(); hdf5_file_processor = HDF5FileProcessor(config); hdf5_visualizer = HDF5Visualizer(config) # init
df_dict = hdf5_file_processor.read_file('/p/scratch/madnuc/cho/DRAGON3/1M_Disk_Halo_trial6/snap.40_0.0000.h5part', N0=1e6) # read hdf5 file; N0 is the initial particle number
df_dict['singles'] # gives pandas DataFrame of "single" (all) stars
df_dict['binaries'] # gives pandas DataFrame of binaries
