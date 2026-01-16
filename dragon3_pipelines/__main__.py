#!/usr/bin/env python3
"""
Main entry point for dragon3_pipelines CLI
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # 限制线程数避免forkserver问题

import sys
import gc
import getopt
import logging
import functools
import multiprocessing
import time
from glob import glob
from typing import Optional

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from dragon3_pipelines.config import ConfigManager
from dragon3_pipelines.io import HDF5FileProcessor, LagrFileProcessor
from dragon3_pipelines.visualization import HDF5Visualizer, LagrVisualizer
from dragon3_pipelines.analysis import ParticleTracker
from dragon3_pipelines.utils import init_worker_logging

# Setup logger
try:
    logger
except NameError:
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.setLevel(logging.INFO)


class SimulationPlotter:
    """模拟处理类，管理整个模拟处理流程"""
    
    def __init__(self, config_manager: ConfigManager) -> None:
        self.config = config_manager
        self.hdf5_file_processor = HDF5FileProcessor(config_manager)
        self.lagr_file_processor = LagrFileProcessor(config_manager)
        self.hdf5_visualizer = HDF5Visualizer(config_manager)
        self.lagr_visualizer = LagrVisualizer(config_manager)
        self.particle_tracker = ParticleTracker(config_manager)

    def plot_hdf5_file(self, hdf5_file_path: str, simu_name: str) -> None:
        """处理单个HDF5文件（包含多个snapshot）
        
        Args:
            hdf5_file_path: Path to HDF5 file (contains multiple snapshots)
            simu_name: Name of the simulation
        """
        # 获取HDF5文件的代表时间
        t_nbody_in_filename = self.hdf5_file_processor.get_hdf5_file_time_from_filename(hdf5_file_path)
        if t_nbody_in_filename < self.config.skip_until_of[simu_name]:
            logger.debug("skipped")
            return
        
        # 加载数据
        df_dict = self.hdf5_file_processor.read_file(hdf5_file_path, simu_name)
        
        # 处理每个时间点
        for ttot in df_dict['scalars']['TTOT'].unique():
            if ttot < self.config.skip_until_of[simu_name]:
                continue
            if self.config.plot_only_int_nbody_time and not ttot.is_integer():
                continue
            logger.debug(f"{ttot=}", end=' | ')
            
            # 获取该时间点的数据
            single_df_at_t, binary_df_at_t, is_valid = self.hdf5_file_processor.get_snapshot_at_t(df_dict, ttot)
            if not is_valid:
                logger.info(f"Warning: {simu_name} {hdf5_file_path} {ttot=} data validation failed, skipping")
                continue
            
            # 位置散点图
            self.hdf5_visualizer.single.create_position_plot_jpg(single_df_at_t, simu_name)
            self.hdf5_visualizer.single.create_position_plot_hightlight_compact_objects_jpg(single_df_at_t, simu_name)
            self.hdf5_visualizer.single.create_position_plot_hightlight_compact_objects_wide_pc_jpg(single_df_at_t, simu_name)

            # 质量-距离关系图
            self.hdf5_visualizer.single.create_mass_distance_plot_density(single_df_at_t, simu_name)
            # CMD图
            self.hdf5_visualizer.single.create_CMD_plot_density(single_df_at_t, simu_name)
            # 彩色CMD图
            self.hdf5_visualizer.single.create_color_CMD_jpg(single_df_at_t, simu_name)
            # 速度-位置 # 不知为何非常非常慢，先不弄
            # self.hdf5_visualizer.single.create_vx_x_plot_density(single_df_at_t, simu_name)

            
            # 双星
            if binary_df_at_t is not None and not binary_df_at_t.empty:
                # 质量比-主星质量图
                self.hdf5_visualizer.binary.create_mass_ratio_m1_plot_density(binary_df_at_t, simu_name)
                self.hdf5_visualizer.binary.create_mass_ratio_m1_plot_jpg_compact_object_only(binary_df_at_t, simu_name)
                # 半长轴-主星质量图
                self.hdf5_visualizer.binary.create_semi_m1_plot_density(binary_df_at_t, simu_name)
                self.hdf5_visualizer.binary.create_semi_m1_plot_jpg_compact_object_only(binary_df_at_t, simu_name)
                # 偏心率-半长轴图
                self.hdf5_visualizer.binary.create_ecc_semi_plot_density(binary_df_at_t, simu_name)
                self.hdf5_visualizer.binary.create_ecc_semi_plot_jpg_compact_object_only(binary_df_at_t, simu_name)
                self.hdf5_visualizer.binary.create_ecc_semi_plot_jpg_compact_object_only_loglog(binary_df_at_t, simu_name)
                # 绑定能-半长轴图
                self.hdf5_visualizer.binary.create_ebind_semi_plot_density(binary_df_at_t, simu_name)
                self.hdf5_visualizer.binary.create_ebind_semi_plot_jpg_compact_object_only(binary_df_at_t, simu_name)
                # GW时间-半长轴图
                self.hdf5_visualizer.binary.create_taugw_semi_plot_jpg_compact_object_only(binary_df_at_t, simu_name)
                # 总质量-距离关系图
                self.hdf5_visualizer.binary.create_mtot_distance_plot_density(binary_df_at_t, simu_name)
                self.hdf5_visualizer.binary.create_mtot_distance_plot_jpg_compact_object_only(binary_df_at_t, simu_name)
                # 速度-位置
                self.hdf5_visualizer.binary.create_bin_vx_x_plot_density(binary_df_at_t, simu_name)
                self.hdf5_visualizer.binary.create_bin_vx_x_plot_jpg_compact_object_only(binary_df_at_t, simu_name)
                # 半长轴-距离
                self.hdf5_visualizer.binary.create_semi_distance_plot_density(binary_df_at_t, simu_name)
                self.hdf5_visualizer.binary.create_semi_distance_plot_jpg_compact_object_only(binary_df_at_t, simu_name)
            
            # 清理内存
            plt.close('all')
            gc.collect()

    def plot_lagr(self, simu_name: str) -> None:
        """处理Lagrangian半径数据"""
        l7df_sns = self.lagr_file_processor.load_sns_friendly_data(simu_name)
        self.lagr_visualizer.create_lagr_radii_plot(l7df_sns, simu_name)
        self.lagr_visualizer.create_lagr_avmass_plot(l7df_sns, simu_name)
        self.lagr_visualizer.create_lagr_velocity_dispersion_plot(l7df_sns, simu_name)
        plt.close('all')
        gc.collect()

    def plot_all_simulations(self) -> None:
        """处理所有模拟"""
        for simu_name in self.config.pathof.keys():
            # 先画lagr
            self.plot_lagr(simu_name)

            # 获取所有HDF5文件
            hdf5_files = sorted(
                glob(self.config.pathof[simu_name] + '/**/*.h5part'), 
                key=lambda fn: self.hdf5_file_processor.get_hdf5_file_time_from_filename(fn)
            )
            WAIT_HDF5_FILE_AGE_HOUR = 24
            cutoff = time.time() - WAIT_HDF5_FILE_AGE_HOUR * 3600
            hdf5_files = [
                fn for fn in hdf5_files
                if os.path.getmtime(fn) <= cutoff
            ]
            
            # 创建带固定参数的部分函数
            process_file_partial = functools.partial(
                self.plot_hdf5_file,
                simu_name=simu_name
            )
            
            # 使用进程池并行处理
            ctx = multiprocessing.get_context('forkserver')
            with ctx.Pool(
                processes=self.config.processes_count, 
                maxtasksperchild=self.config.tasks_per_child,
                initializer=init_worker_logging
            ) as pool:
                list(
                    tqdm(
                        pool.imap(process_file_partial, hdf5_files), 
                        total=len(hdf5_files), 
                        desc=f'{simu_name} HDF5 Files'
                    )
                )


def main() -> int:
    """Main CLI entry point"""
    try:
        long_options = ["skip-until=", 'debug'] 
        opts, args = getopt.getopt(sys.argv[1:], "k:", long_options)
        if '--debug' in dict(opts):
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
    except getopt.GetoptError as err:
        print(err) 
        sys.exit(2)

    config = ConfigManager(opts=opts)
    
    # 初始化处理器
    plotter = SimulationPlotter(config)
    
    # 处理所有模拟
    plotter.plot_all_simulations()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
