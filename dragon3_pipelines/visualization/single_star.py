"""Single star visualization tools"""

import logging
import os
from typing import Any, Callable, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dragon3_pipelines.utils import log_time
from dragon3_pipelines.visualization.base import BaseHDF5Visualizer, add_grid


class SingleStarVisualizer(BaseHDF5Visualizer):
    """Visualizer for single star data"""
    
    @log_time(__name__)
    def create_mass_distance_plot_density(
        self, 
        single_df_at_t: pd.DataFrame, 
        simu_name: str
    ) -> None:
        """Create mass-distance relationship plot"""
        self._create_jointplot_density(
            df_at_t=single_df_at_t,
            simu_name=simu_name,
            x_col='Distance_to_cluster_center[pc]',
            y_col='M',
            log_scale=(True, True),
            filename_var_part='mass_vs_distance_loglog',
        )
    
    @log_time(__name__)
    def create_vx_x_plot_density(
        self, 
        single_df_at_t: pd.DataFrame, 
        simu_name: str
    ) -> None:
        """Create velocity-position plot"""
        self._create_jointplot_density(
            df_at_t=single_df_at_t,
            simu_name=simu_name,
            x_col='X [pc]',
            y_col='V1',  
            log_scale=(False, False),  
            filename_var_part='allstar_vx_vs_x',
            xlim_key='position_pc_lim', 
            ylim_key='velocity_kmps_lim', 
        )
    
    @log_time(__name__)
    def create_CMD_plot_density(
        self, 
        single_df_at_t: pd.DataFrame, 
        simu_name: str
    ) -> None:
        """Create color-magnitude diagram"""
        def _custom_decorator(ax: plt.Axes, df: pd.DataFrame) -> None:
            ax.invert_xaxis()
        self._create_jointplot_density(
            df_at_t=single_df_at_t,
            simu_name=simu_name,
            x_col='Teff*',
            y_col='L*',
            log_scale=(True, True),
            filename_var_part='L_vs_Teff_loglog',
            custom_ax_joint_decorator=_custom_decorator
        )
    
    @log_time(__name__)
    def create_position_plot_jpg(
        self, 
        single_df_at_t: pd.DataFrame, 
        simu_name: str,
        filename_suffix: Optional[str] = None,
        extra_data_handler: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        extra_ax_handler: Optional[Callable[[plt.Axes], None]] = None,
        custom_ax_decorator: Optional[Callable[[plt.Axes], None]] = None,
        uniform_color_and_size: bool = False
    ) -> None:
        """Create position scatter plot"""
        logger = logging.getLogger(__name__)
        ttot = single_df_at_t['TTOT'].iloc[0]
        tmyr = single_df_at_t['Time[Myr]'].iloc[0]
        t_over_tcr0 = single_df_at_t['TTOT/TCR0'].iloc[0]
        t_over_trh0 = single_df_at_t['TTOT/TRH0'].iloc[0]
        if extra_data_handler is not None:
            single_df_at_t = extra_data_handler(single_df_at_t)
        if filename_suffix is None:
            save_jpg_path = f"{self.config.plot_dir}/jpg/{self.config.figname_prefix[simu_name]}output_ttot_{ttot}_x1_vs_x2.jpg"
        else:
            save_jpg_path = f"{self.config.plot_dir}/jpg/{self.config.figname_prefix[simu_name]}output_ttot_{ttot}_x1_vs_x2_{filename_suffix}.jpg"
        
        if self.config.skip_existing_plot and os.path.exists(save_jpg_path):
            logger.debug(f"Skip existing plot: {save_jpg_path}")
            return
        color_rgb_darker = self.teff_to_rgb_converter.get_rgb(single_df_at_t['Teff*'].values) * \
            self.luminosity_to_plot_alpha(single_df_at_t['L*'].values)[:, np.newaxis]
        if not uniform_color_and_size:
            size = np.sqrt(single_df_at_t['R*'])
            color = color_rgb_darker
        else:
            size = 10
            color = 'white' 
        with plt.style.context('dark_background'):
            ax = sns.scatterplot(
                data=single_df_at_t, 
                x='X [pc]', y='Y [pc]', marker='.', lw=0,
                s=size,
                color=color
            )
            if extra_ax_handler is not None:
                extra_ax_handler(ax)
            self.decorate_jointfig(
                ax, single_df_at_t, 'X [pc]', 'Y [pc]', 
                self.config.limits['position_pc_lim'], self.config.limits['position_pc_lim'], 
                simu_name, 
                ttot, tmyr, t_over_tcr0, t_over_trh0
            )
            if custom_ax_decorator is not None:
                custom_ax_decorator(ax)
            ax.figure.savefig(save_jpg_path, transparent=False)
            try:
                __IPYTHON__
                if self.config.close_figure_in_ipython:
                    plt.close(ax.figure)
            except NameError:
                plt.close(ax.figure)

    @log_time(__name__)
    def create_position_plot_wide_pc_jpg(
        self, 
        single_df_at_t: pd.DataFrame, 
        simu_name: str
    ) -> None:
        """Create wide position plot"""
        def _set_wide_pos_lim_pc(ax: plt.Axes) -> None:
            ax.set_xlim(*self.config.limits['position_pc_lim_MAX'])
            ax.set_ylim(*self.config.limits['position_pc_lim_MAX'])

        self.create_position_plot_jpg(
            single_df_at_t=single_df_at_t,
            simu_name=simu_name,
            filename_suffix='wide_pc',
            custom_ax_decorator=_set_wide_pos_lim_pc,
            uniform_color_and_size=True
        )

    @log_time(__name__)
    def create_position_plot_hightlight_compact_objects_jpg(
        self, 
        single_df_at_t: pd.DataFrame, 
        simu_name: str, 
        filename_suffix: Optional[str] = None,
        extra_data_handler: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        extra_ax_handler: Optional[Callable[[plt.Axes], None]] = None,
        custom_ax_decorator: Optional[Callable[[plt.Axes], None]] = None
    ) -> None:
        """Create position plot highlighting compact objects"""
        logger = logging.getLogger(__name__)
        ttot = single_df_at_t['TTOT'].iloc[0]
        tmyr = single_df_at_t['Time[Myr]'].iloc[0]
        t_over_tcr0 = single_df_at_t['TTOT/TCR0'].iloc[0]
        t_over_trh0 = single_df_at_t['TTOT/TRH0'].iloc[0]

        processed_df = single_df_at_t
        if extra_data_handler is not None:
            processed_df = extra_data_handler(single_df_at_t)

        if filename_suffix is None:
            save_jpg_path = f"{self.config.plot_dir}/jpg/{self.config.figname_prefix[simu_name]}output_ttot_{ttot}_x1_vs_x2_highlight_compact_objects.jpg"
        else:
            save_jpg_path = f"{self.config.plot_dir}/jpg/{self.config.figname_prefix[simu_name]}output_ttot_{ttot}_x1_vs_x2_highlight_compact_objects_{filename_suffix}.jpg"
        
        if self.config.skip_existing_plot and os.path.exists(save_jpg_path):
            logger.debug(f"Skip existing plot: {save_jpg_path}")
            return

        fig, ax = plt.subplots()

        df_hewd = processed_df[processed_df['KW'] == 10]
        df_cowd = processed_df[processed_df['KW'] == 11]
        df_onewd = processed_df[processed_df['KW'] == 12]
        df_ns = processed_df[processed_df['KW'] == 13]
        df_bh = processed_df[processed_df['KW'] == 14]
        df_others = processed_df[~processed_df['KW'].isin([10, 11, 12, 13, 14])]

        if not df_others.empty:
            sns.scatterplot(
                data=df_others,
                x='X [pc]', y='Y [pc]', marker='o', lw=0,
                s=np.sqrt(df_others['R*']),
                color='gray',
                ax=ax
            )

        compact_object_plot_configs = [
            (df_hewd, self.config.marker_nofill_list[10], 'blue',  'HeWD',  None   , 1.5),
            (df_cowd, self.config.marker_nofill_list[11], 'blue',  'COWD',  None   , 1.5),
            (df_onewd,self.config.marker_nofill_list[12], 'blue',  'ONeWD', None   , 1.5),
            (df_ns,   self.config.marker_fill_list[13],   'red',   'NS',    'green', 0),
            (df_bh,   self.config.marker_fill_list[14],   'black', 'BH',    'white', 0),
        ]

        for df_compact, marker, color, label, edgecolors, lw in compact_object_plot_configs:
            if not df_compact.empty:
                plot_kwargs = {
                    'data': df_compact,
                    'x': 'X [pc]', 
                    'y': 'Y [pc]',
                    'marker': marker,
                    's': 30,
                    'color': color,
                    'label': label,
                    'alpha': 0.7,
                    'edgecolors': edgecolors,
                    'lw': lw,
                    'ax': ax
                }
                sns.scatterplot(**plot_kwargs)
        
        if extra_ax_handler is not None:
            extra_ax_handler(ax)

        self.decorate_jointfig(
            ax, processed_df, 'X [pc]', 'Y [pc]', 
            self.config.limits['position_pc_lim'], self.config.limits['position_pc_lim'], 
            simu_name, 
            ttot, tmyr, t_over_tcr0, t_over_trh0
        )
        
        if ax.legend_ is not None:
            ax.legend_.remove()

        if custom_ax_decorator is not None:
            custom_ax_decorator(ax)

        add_grid(ax)
        fig.savefig(save_jpg_path, transparent=False) 
        try:
            __IPYTHON__
            if self.config.close_figure_in_ipython:
                plt.close(fig)
        except NameError:
            plt.close(fig)

    @log_time(__name__)
    def create_position_plot_hightlight_compact_objects_wide_pc_jpg(
        self, 
        single_df_at_t: pd.DataFrame, 
        simu_name: str
    ) -> None:
        """Create wide position plot highlighting compact objects"""
        def _set_wide_pos_lim_pc(ax: plt.Axes) -> None:
            ax.set_xlim(*self.config.limits['position_pc_lim_MAX'])
            ax.set_ylim(*self.config.limits['position_pc_lim_MAX'])
        self.create_position_plot_hightlight_compact_objects_jpg(
            single_df_at_t=single_df_at_t,
            simu_name=simu_name,
            filename_suffix='wide_pc',
            custom_ax_decorator=_set_wide_pos_lim_pc
        )

    @log_time(__name__)
    def create_color_CMD_jpg(
        self, 
        single_df_at_t: pd.DataFrame, 
        simu_name: str,
        extra_data_handler: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        extra_ax_handler: Optional[Callable[[plt.Axes], None]] = None
    ) -> None:
        """Create color CMD plot"""
        logger = logging.getLogger(__name__)
        ttot = single_df_at_t['TTOT'].iloc[0]
        tmyr = single_df_at_t['Time[Myr]'].iloc[0]
        t_over_tcr0 = single_df_at_t['TTOT/TCR0'].iloc[0]
        t_over_trh0 = single_df_at_t['TTOT/TRH0'].iloc[0]
        if extra_data_handler is not None:
            single_df_at_t = extra_data_handler(single_df_at_t)
        save_jpg_path = f"{self.config.plot_dir}/jpg/{self.config.figname_prefix[simu_name]}output_ttot_{ttot}_CMD.jpg"
        if self.config.skip_existing_plot and os.path.exists(save_jpg_path):
            logger.debug(f"Skip existing plot: {save_jpg_path}")
            return
        all_stellar_types_sorted = sorted(
            self.config.kw_to_stellar_type_verbose.values(),
            key=lambda x: int(x.split(':')[0])
        )
        
        ax = sns.scatterplot(
            data=single_df_at_t, 
            x='Teff*', 
            y='L*', 
            hue='Stellar Type',
            hue_order=all_stellar_types_sorted,
            style='Stellar Type',
            palette=self.config.st_verbose_to_color,
            s=20,
            linewidths=0.8,
            markers=self.config.star_type_verbose_to_marker,
            legend='full',
        )
        ax.set(xscale='log', yscale='log')
        if extra_ax_handler is not None:
            extra_ax_handler(ax)
        self.decorate_jointfig(
            ax, 
            single_df_at_t, 
            'Teff*', 
            'L*', 
            self.config.limits['Teff*'], 
            self.config.limits['L*'], 
            simu_name, 
            ttot, tmyr, t_over_tcr0, t_over_trh0
        )

        legend_handles = []
        placeholder_marker_color = (0, 0, 0, 0.0) 
        legend_marker_size = np.sqrt(20) 
        legend_marker_edge_width = 0.8

        current_stellar_types = set(single_df_at_t['Stellar Type'].unique())
        for st_type in all_stellar_types_sorted:
            marker_shape = self.config.star_type_verbose_to_marker.get(st_type, '+') 

            if st_type in current_stellar_types:
                label = st_type
                color = self.config.st_verbose_to_color.get(st_type, 'black')
                mec = color
            else:
                label = " "*8
                mec = placeholder_marker_color
            
            handle = mpl.lines.Line2D(
                [0], [0], 
                marker=marker_shape, 
                linestyle='None',
                label=label,
                markeredgecolor=mec,
                markersize=legend_marker_size,
                markeredgewidth=legend_marker_edge_width 
            )
            legend_handles.append(handle)

        with mpl.rc_context(**self.config.fixed_width_font_context):
            plt.legend(
                handles=legend_handles, 
                title='Stellar Type', 
                fontsize=12, 
                title_fontsize=12, 
                bbox_to_anchor=(1, 1), 
                loc='upper left'
            )

        ax.invert_xaxis()
        ax.figure.savefig(save_jpg_path)
        try:
            __IPYTHON__
            if self.config.close_figure_in_ipython:
                plt.close(ax.figure)
        except NameError:
            plt.close(ax.figure)
