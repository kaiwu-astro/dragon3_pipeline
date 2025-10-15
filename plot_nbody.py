#!/usr/bin/env python3
from nbody_tools import *
import gc
import sys
import seaborn as sns
import getopt
import logging
import functools
import multiprocessing
from typing import Callable, Optional
try:
    logger
except NameError:
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.DEBUG)
try:
    import petar
except ImportError:
    logger.warning('Petar module not found.')

def set_mpl_fonts():
    # check default： plt.rcParams
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    # BIGGER_SIZE = 12
    BIGGER_SIZE = 15
    plt.rc("mathtext", fontset='dejavuserif')
    plt.rc('font', family='serif')
    # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    # plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    # plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rc('figure', figsize=(6, 6))
    plt.rc('savefig', transparent=True, dpi=330, bbox='tight')
    plt.rc('errorbar', capsize=3)
    plt.rc('legend', framealpha=0.1)

def add_grid(axs, which='both', axis='both'):
    '''
    Add grid to axes. 
        which: {'both', 'major', 'minor'}
        axis: {'both', 'x', 'y'}
    '''
    try:
        axs[0]
        axes = axs
    except:
        axes = [axs,]

    for ax in axes:
        if which == 'both' or which == 'minor':
            if ax.get_yscale() == 'linear':
                ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
            if ax.get_xscale() == 'linear':
                ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.grid(visible=True, which=which, axis=axis, color='black', alpha=0.1)

class ConfigManager:
    """配置管理类，集中管理所有配置参数"""
    def _parse_argv(self, opts=[]):
        """
        Parse command line arguments and update configuration
        For example: --skip-until=0: start processing from t=0, i.e., the first file, used to add a new image to all data
                     --skip-until=last: read the last timestamp from the image list, do not read any previous data, used for daily operation
        """
        for opt, arg in opts:
            if opt == '--skip-until':
                if can_convert_to_float(arg):
                    arg = float(arg)
                for key in self.skip_until_of:
                    self.skip_until_of[key] = arg
    def __init__(self, opts=[]):
        # 路径配置
        self.pathof = {
            '0sb' : '/p/scratch/madnuc/wu13/mar17-dragon3-start/00softBinary-5hardBinary',
            '20sb': '/p/scratch/madnuc/wu13/mar17-dragon3-start/20softBinary-5hardBinary',
            '60sb': '/p/scratch/madnuc/wu13/mar17-dragon3-start/60softBinary-5hardBinary',
        }
        self.input_file_path_of = {
            '0sb' : self.pathof['0sb'] + '/N1m-Dragon3.inp.init',
            '20sb': self.pathof['20sb'] + '/N1m-Dragon3.inp.init',
            '60sb': self.pathof['60sb'] + '/N1m-Dragon3.inp.init',
        }
        # plot_file_prefix
        self.figname_prefix = {
            '0sb' : 'dragon3_1m_5hb_0sb_Apr2025',
            '20sb': 'dragon3_1m_5hb_20sb_Apr2025',
            '60sb': 'dragon3_1m_5hb_60sb_Apr2025',
        }
        self.plot_dir: str = '/p/home/jusers/wu13/juwels/scratch/plot'
        
        # 通用配置
        self.skip_until_of: dict = { # can change by command line
            '0sb' : 700,
            '20sb': 60,
            '60sb': 0,
        }
        self.skip_existing_plot: bool = True
        self.plot_only_int_nbody_time: bool = True
        self.close_figure_in_ipython: bool = False
        self.processes_count: int = 40
        self.tasks_per_child: int = 3
        self.selected_lagr_percent: list = ['<RC', '10%', '30%', '50%', '70%', '90%', '95%']
        self.ECLOSE_INPUT = 1.0 # Binding energy per unit mass for hard binary (positive)
        self.universe_age_myr = 13.8e3
        self.kw_to_stellar_type = {
            -1: "PMS",
            0:  "LMS",
            1:  "MS",
            2:  "HG",
            3:  "GB",
            4:  "CHeB",
            5:  "EAGB",
            6:  "TPAGB",
            7:  "HeMS",
            8:  "HeHG",
            9:  "HeGB",
            10: "HeWD",
            11: "COWD",
            12: "ONeWD",
            13: "NS",
            14: "BH",
            15: "MLR",
        }
        self.compact_object_KW = np.array([10, 11, 12, 13, 14])
        self.stellar_type_to_kw = {v: k for k, v in self.kw_to_stellar_type.items()}
        self.kw_to_stellar_type_verbose = {k: f"{k:2d}:{v}" for k, v in self.kw_to_stellar_type.items()}
        self.stellar_type_verbose_to_kw = {v: k for k, v in self.kw_to_stellar_type_verbose.items()}
        self.wow_binary_st_list = ['BH-NS',]
        self.fixed_width_font_context = dict(rc={'font.family': 'monospace'})
        # binary_star_types = [f'{st1}-{st2}' for i1, st1 in enumerate(['BH', 'NS', 'WD', 'OO']) for i2, st2 in enumerate(['BH', 'NS', 'WD', 'OO']) if i1 <= i2]

        # 画图样式
        _marker_fill_list = ['d', 'v', '^', '<', '>', 'h', '8', 's', 'p', 'H', 'D', 'o']
        self.marker_fill_list = ['o',] * (17 - len(_marker_fill_list)) + _marker_fill_list
        self.marker_nofill_list = (['1', '+', '3', 'x'] * 5)[:len(self.kw_to_stellar_type)]
        self.star_type_verbose_to_marker = dict(zip(list(self.kw_to_stellar_type_verbose.values()), self.marker_nofill_list))
        self.palette_st = sns.color_palette(n_colors=10) + \
                sns.color_palette("husl", 7)[::2] + \
                sns.color_palette("husl", 7)[1::2][:-3] + \
                [(1,0,0), (0,0,0), (0.8,0.8,0.8)]
        self.st_verbose_to_color = {k: self.palette_st[v+1] for k, v in self.stellar_type_verbose_to_kw.items()}
        self.limits = {
            'Distance_to_cluster_center[pc]': (1e-3, 1e4),
            'Bin A[au]': (1e-3, 1e5),
            'mass_ratio': (5e-4, 1.01),
            'M': (0.06, 1050),
            'Bin ECC': (1e-4, 1.2),
            'Teff*': (1e3, 1.05e6),
            'L*': (0.9e-5, 1e8),
            'position_pc_lim': (-15, 15),
            'position_pc_lim_MAX': (-500, 500),
            'Ebind/kT': (2e-7, 5e2),
            'tau_gw[Myr]': (9, 14000),
            'velocity_kmps_lim': (-200, 200),
            'sum_of_radius[au]': (1e-7, 1),
            'peri[au]': (1e-4, 1e5),
        }
        self.colname_to_label = {
            'Distance_to_cluster_center[pc]': 'r [pc]',
            'Bin A[au]': 'a [au]',
            'Bin ECC': 'e',
            'mass_ratio': 'mass ratio',
            'primary_mass[solar]': r'primary mass [M$_\odot$]',
            'secondary_mass[solar]': r'secondary mass [M$_\odot$]',
            'total_mass[solar]': r'binary total mass [M$_\odot$]',
            'M': r'mass [M$_\odot$]',
            'Ebind/kT': r'$E_{bind}/kT$',
            'Teff*': r'$T_{eff}$ [K]',
            'L*': r'$L$ [L$_\odot$]',
            'tau_gw[Myr]': r'$\tau_{gw}$ [Myr]',
            'V1': r'$V_{x}$ [km/s]',
            'V2': r'$V_{y}$ [km/s]',
            'V3': r'$V_{z}$ [km/s]',
            'Bin cm V1': r'Bin $V_{x,cm}$ [km/s]',
            'Bin cm V2': r'Bin $V_{y,cm}$ [km/s]',
            'Bin cm V3': r'Bin $V_{z,cm}$ [km/s]',
        }
        self._parse_argv(opts=opts)
        for simu_name in self.pathof.keys():
            if self.skip_until_of[simu_name] == 'last':
                # 查找 plot_dir/figname_prefix+'ttot_'
                all_pdf_plots = glob(
                    self.plot_dir + '/' + self.figname_prefix[simu_name] + '*ttot_*.pdf'
                    )
                if all_pdf_plots:
                    # pdf_plot_times 对 all_pdf_plots 取数字 
                    _get_time = lambda x: float(x.split('ttot_')[1].split('_')[0])
                    all_times = np.array([_get_time(x) for x in all_pdf_plots])
                    self.skip_until_of[simu_name] = all_times.max()
                    logger.info(f"[{simu_name}] Get skip-until={self.skip_until_of[simu_name]}")
                else:
                    self.skip_until_of[simu_name] = 0


class HDF5FileProcessor:
    """读取和画图前预处理HDF5数据"""
    def __init__(self, config_manager):
        self.config = config_manager
    
    @log_time(logger)
    def read_file(self, hdf5_path, simu_name=None, N0=None):
        """加载并预处理HDF5数据. 
            simu_name：用于获取初始条件文件路径，读取N0
            也可simu_name=None，改为直接传入N0"""
        logger.debug(f"\nProcessing {hdf5_path=}...")
        
        # 获取数据框
        df_dict = dataframes_from_hdf5_file(hdf5_path)
        if N0 is None:
            N0 = int(get_valueStr_of_namelist_key(path=self.config.input_file_path_of[simu_name], key='N'))

        # 预处理标量数据
        scalar_df_all = df_dict['scalars']
        scale_dict = get_scale_dict_from_hdf5_df(scalar_df=scalar_df_all)
        scalar_df_all['Time[Myr]'] = scalar_df_all['TTOT'] * scale_dict['t']
        scalar_df_all['CLIGHT'] = 3.0e5 / scalar_df_all['VSTAR']
        rdens_coord_pc = scalar_df_all[['RDENS(1)', 'RDENS(2)', 'RDENS(3)']] * scale_dict['r']

        # 预处理单星数据
        single_df_all = df_dict['singles']
        single_df_all['TTOT/TCR0'] = single_df_all['TTOT'].map(scalar_df_all['TTOT/TCR0'])
        single_df_all['TTOT/TRH0'] = single_df_all['TTOT/TCR0'] / (0.1 * N0 / np.log(0.4*N0))
        single_df_all['Time[Myr]'] = single_df_all['TTOT'] * scale_dict['t']
        offsets_x1 = single_df_all['TTOT'].map(rdens_coord_pc['RDENS(1)'])
        offsets_x2 = single_df_all['TTOT'].map(rdens_coord_pc['RDENS(2)'])
        offsets_x3 = single_df_all['TTOT'].map(rdens_coord_pc['RDENS(3)'])
        single_df_all['X [pc]'] = single_df_all['X1'] - offsets_x1
        single_df_all['Y [pc]'] = single_df_all['X2'] - offsets_x2
        single_df_all['Z [pc]'] = single_df_all['X3'] - offsets_x3
        single_df_all['Distance_to_cluster_center[pc]'] = np.sqrt(
            single_df_all['X [pc]']**2 + single_df_all['Y [pc]']**2 + single_df_all['Z [pc]']**2
        )
        single_df_all['Stellar Type'] = single_df_all['KW'].map(self.config.kw_to_stellar_type_verbose)
        # NS和BH的光度、温度都是artificial。模拟器设置的值离主序太远，修改以方便展示。
        ## 光度统一设为画图的光度下限
        single_df_all.loc[single_df_all['Stellar Type'].isin(['13:NS', '14:BH']), 'L*'] = self.config.limits['L*'][0] * 1.2
        ## 温度小于画图温度下限的，设为下限；超过上限的，设为上限
        single_df_all.loc[single_df_all['Stellar Type'].isin(['13:NS', '14:BH']), 'Teff*'] = np.clip(
            single_df_all.loc[single_df_all['Stellar Type'].isin(['13:NS', '14:BH']), 'Teff*'], 
            self.config.limits['Teff*'][0], 
            self.config.limits['Teff*'][1]
        )
        
        # 预处理双星数据
        binary_df_all = df_dict['binaries']
        binary_df_all['TTOT/TCR0'] = binary_df_all['TTOT'].map(scalar_df_all['TTOT/TCR0'])
        binary_df_all['TTOT/TRH0'] = binary_df_all['TTOT/TCR0'] / (0.1 * N0 / np.log(0.4*N0))
        binary_df_all['Time[Myr]'] = binary_df_all['TTOT'] * scale_dict['t']
        offsets_x1b = binary_df_all['TTOT'].map(rdens_coord_pc['RDENS(1)'])
        offsets_x2b = binary_df_all['TTOT'].map(rdens_coord_pc['RDENS(2)'])
        offsets_x3b = binary_df_all['TTOT'].map(rdens_coord_pc['RDENS(3)'])
        binary_df_all['Bin cm X [pc]'] = binary_df_all['Bin cm X1'] - offsets_x1b
        binary_df_all['Bin cm Y [pc]'] = binary_df_all['Bin cm X2'] - offsets_x2b
        binary_df_all['Bin cm Z [pc]'] = binary_df_all['Bin cm X3'] - offsets_x3b
        binary_df_all['primary_mass[solar]'] = np.max(binary_df_all[['Bin M1*', 'Bin M2*']], axis=1)
        binary_df_all['secondary_mass[solar]'] = np.min(binary_df_all[['Bin M1*', 'Bin M2*']], axis=1)
        binary_df_all['total_mass[solar]'] = binary_df_all['Bin M1*'] + binary_df_all['Bin M2*'] 
        binary_df_all['Distance_to_cluster_center[pc]'] = np.sqrt( 
            binary_df_all['Bin cm X [pc]']**2 + binary_df_all['Bin cm Y [pc]']**2 + binary_df_all['Bin cm Z [pc]']**2
        )
        binary_df_all['mass_ratio'] = binary_df_all['secondary_mass[solar]'] / binary_df_all['primary_mass[solar]']
        binary_df_all['primary_stellar_type'] = np.maximum(binary_df_all['Bin KW1'], binary_df_all['Bin KW2'])
        binary_df_all['secondary_stellar_type'] = np.minimum(binary_df_all['Bin KW1'], binary_df_all['Bin KW2'])
        binary_df_all['Stellar Type'] = binary_df_all['primary_stellar_type'].map(self.config.kw_to_stellar_type) + '-' + \
            binary_df_all['secondary_stellar_type'].map(self.config.kw_to_stellar_type)
        # Ebind_abs = G * M1 * M2 / 2a / (M1 + M2)
        binary_df_all['peri[au]'] = binary_df_all['Bin A[au]'] * (1 - binary_df_all['Bin ECC'])
        binary_df_all['sum_of_radius[solar]'] = binary_df_all['Bin RS1*'] + binary_df_all['Bin RS2*']
        binary_df_all['sum_of_radius[au]'] = binary_df_all['sum_of_radius[solar]'] * u.solRad.to(u.au)
        binary_df_all['Ebind_abs_NBODY'] = \
             binary_df_all['Bin M1*'] / scale_dict['m']         \
                * binary_df_all['Bin M2*'] / scale_dict['m']     \
            / (2 * binary_df_all['Bin A[au]'] / pc_to_AU / scale_dict['r']) \
            / (binary_df_all['Bin M1*'] / scale_dict['m'] + binary_df_all['Bin M2*'] / scale_dict['m'])
        binary_df_all['Ebind/kT'] = \
            binary_df_all['Ebind_abs_NBODY'] / self.config.ECLOSE_INPUT
        binary_df_all['is_hard_binary'] = binary_df_all['Ebind/kT'] >= 1
        # binary_df_all['tau_gw[Myr]'] = tau_gw(
        #     a = binary_df_all['Bin A[au]'] / pc_to_AU / scale_dict['r'],
        #     e = binary_df_all['Bin ECC'],
        #     mu = (binary_df_all['Bin M1*'] * binary_df_all['Bin M2*']) / (binary_df_all['Bin M1*'] + binary_df_all['Bin M2*']) / scale_dict['m'],
        #     M = (binary_df_all['Bin M1*'] + binary_df_all['Bin M2*']) / scale_dict['m'],
        #     G = 1,
        #     c = scalar_df_all['CLIGHT'].iloc[0],
        # ) * scale_dict['t']
        # 结果几乎一样，下面astropy对结果大概小1%。倾向于是多次scale转换出现的舍入误差。相信astropy
        with warnings.catch_warnings():
            # ignore "RuntimeWarning: invalid value encountered in power"
            warnings.simplefilter("ignore", category=RuntimeWarning) 
            binary_df_all['tau_gw[Myr]'] = tau_gw(
                a = binary_df_all['Bin A[au]'].values * u.au,
                e = binary_df_all['Bin ECC'].values,
                mu = ((binary_df_all['Bin M1*'] * binary_df_all['Bin M2*']) / (binary_df_all['Bin M1*'] + binary_df_all['Bin M2*'])).values * u.solMass,
                M = (binary_df_all['Bin M1*'] + binary_df_all['Bin M2*']).values * u.solMass,
                G = constants.G,
                c = constants.c,
            ).to(u.Myr).value
        binary_df_all['tau_gw[Myr]'] = np.minimum(self.config.universe_age_myr, binary_df_all['tau_gw[Myr]'])

        # 处理merger数据
        merger_df_all = df_dict['mergers']
        if merger_df_all is not None and not merger_df_all.empty and 'TTOT' in merger_df_all.columns:
            merger_df_all['TTOT/TCR0'] = merger_df_all['TTOT'].map(scalar_df_all['TTOT/TCR0'])
            merger_df_all['TTOT/TRH0'] = merger_df_all['TTOT/TCR0'] / (0.1 * N0 / np.log(0.4*N0))
            merger_df_all['Time[Myr]'] = merger_df_all['TTOT'] * scale_dict['t']
            # 别的暂时用不上... TODO: 处理双星图闪烁的时候要加回merger...

        return df_dict
    
    @log_time(logger)
    def get_snapshot_time(self, hdf5_path):
        """从文件名中提取快照时间"""
        return int(hdf5_path.split('snap.40_')[1].split('.h5part')[0])
    
    @log_time(logger)
    def get_snapshot_at_t(self, df_dict, ttot):
        """获取特定时间的数据"""
        single_df = df_dict['singles'][df_dict['singles']['TTOT'] == ttot].copy()
        binary_df = df_dict['binaries'][df_dict['binaries']['TTOT'] == ttot].copy()
        
        # 获取单星和双星的数量
        N_SINGLE = df_dict['scalars'].loc[ttot, 'N_SINGLE']
        N_BINARY = df_dict['scalars'].loc[ttot, 'N_BINARY']
        
        # 验证数据完整性
        if not isinstance(N_SINGLE, (float, np.float64, np.float32)) or not isinstance(N_BINARY, (float, np.float64, np.float32)):
            return None, None, False
        
        return single_df, binary_df, True


class ContinousFileProcessor:
    def __init__(self, config_manager, file_basename):
        self.config = config_manager
        self.file_basename = file_basename
        self.file_path = None
        self.firstjobof = {}
        self.scale_dict_of = {}

    def concat_file(self, simu_name):
        gather_file_cmd = f'cd {self.config.pathof[simu_name]};' + \
        f'''tmpf=`mktemp --suffix=.{self.file_basename}`; find . -name '{self.file_basename}*' | xargs ls | xargs cat > $tmpf; echo $tmpf'''
        self.file_path = get_output(gather_file_cmd)[0]
        logger.debug(f'Gathered {self.file_basename} of {simu_name} files into {self.file_path}')
    
    def read_file(self, simu_name):
        self.concat_file(simu_name)
        logger.debug(f'Loading gathered self.file_basename at {self.file_path}')
        raise NotImplementedError("子类必须实现此方法")
    
    def clean_data(self, df, timecol='TIME[NB]'):
        """
        可能因为模拟重跑而造成某个star反复输出
        在类似[1.0, 2.1, 3.2, 4.3, 5.7, 3.5, 4.6, 5.9, 4.7, 4.8, 7.1]的数据里去掉 3.5， 4.6， 4.7，4.8
        """
        is_forwarding = np.array([df[timecol][:i+1].max() == v for i, v in df[timecol].items()])
        # print number of False = data to remove
        if not is_forwarding.all():
            logger.warning(f"[{self.file_basename}] Warning: Found {len(is_forwarding) - is_forwarding.sum()} descending entries in {timecol}, removing")
        return df[is_forwarding].reset_index(drop=True)

    def compact_object_filter(self, df, col1, col2=None):
        """
        过滤出紧凑天体数据
        """
        compact_object_mask = df[col1].isin(self.config.compact_object_KW)
        if col2 is not None:
            compact_object_mask |= df[col2].isin(self.config.compact_object_KW)
        return df[compact_object_mask]

    def firstjobhere(self, simu_name):
        '''同shell命令，返回jobid。自带缓存机制'''
        if simu_name not in self.firstjobof.keys():
            get_firstj_cmd = f'cd {self.config.pathof[simu_name]};' + \
            r'''ls | grep -E '^[0-9]+$' | sort -n | head -n 1'''
            self.firstjobof[simu_name] = get_output(get_firstj_cmd)[-1]
        return self.firstjobof[simu_name]
    firstj = firstjobhere
    
    def get_scale_dict_from_stdout(self, simu_name):
        """
        从stdout中提取缩放字典。自带缓存机制
        """
        if simu_name not in self.scale_dict_of:
            first_output_file_path = glob(self.config.pathof[simu_name] + '/' + self.firstj(simu_name) + '/N*out')[0]
            self.scale_dict_of[simu_name] = get_scale_dict(first_output_file_path)
            print(f'Got {self.scale_dict_of[simu_name]} from {first_output_file_path}')
        return self.scale_dict_of[simu_name]
    

class LagrFileProcessor(ContinousFileProcessor):
    """读取和画图前预处理lagr.7"""
    def __init__(self, config_manager):
        super().__init__(config_manager, file_basename='lagr.7')
    
    def read_file(self, simu_name):
        self.concat_file(simu_name)
        logger.debug(f'Loading gathered {self.file_basename} of {simu_name} at {self.file_path}')
        l7df = read_lagr_7(self.file_path)
        l7df = self.clean_data(l7df)
        l7df = l7df_to_physical_units(l7df, self.get_scale_dict_from_stdout(simu_name))
        return l7df
    
    def clean_data(self, l7df):
        """
        1) 丢弃包含非数值型数据的行（应全为 int/float）
           规则：对整表执行 to_numeric(errors='coerce')，若某单元格原本非 NaN，
           转换后为 NaN，则视为该单元格非数值；含此类单元格的整行剔除。
        2) 处理 'Time[NB]' 的重复：保留最后一次出现（避免中途中断导致的不完整行）。
        """
        # Step 1: drop rows with non-numeric cells
        numeric_df = l7df.apply(pd.to_numeric, errors='coerce')
        non_numeric_mask = (numeric_df.isna() & l7df.notna()).any(axis=1)
        if non_numeric_mask.any():
            if 'Time[NB]' in l7df.columns:
                bad_times = np.unique(l7df.loc[non_numeric_mask, 'Time[NB]'].values)
                logger.warning(f"[lagr.7] Warning: Found non-numeric entries; dropping {non_numeric_mask.sum()} rows at Time[NB]={bad_times}")
            else:
                logger.warning(f"[lagr.7] Warning: Found non-numeric entries; dropping {non_numeric_mask.sum()} rows (no 'Time[NB]' column)")
        l7df = numeric_df.loc[~non_numeric_mask].copy()

        # Step 2: de-duplicate on Time[NB], keep last
        if 'Time[NB]' in l7df.columns:
            duplicated_times = l7df['Time[NB]'].duplicated(keep=False)
            if duplicated_times.any():
                dup_vals = np.unique(l7df.loc[duplicated_times, 'Time[NB]'].values)
                logger.warning(f"[lagr.7] Warning: Duplicate 'Time[NB]' detected at {dup_vals}; using the last occurrence")
                l7df = l7df.loc[l7df['Time[NB]'].duplicated(keep='last') | ~duplicated_times]
        else:
            logger.warning("[lagr.7] Warning: 'Time[NB]' column not found when de-duplicating.")
        return l7df
    
    def load_sns_friendly_data(self, simu_name):
        l7df_sns = transform_l7df_to_sns_friendly(self.read_file(simu_name))
        # 对于Metric = sigma2, sigma_r2, sigma_t2的，将值取平方根，新Metric = sigma, sigma_r, sigma_t
        metrics_to_transform = ['sigma2', 'sigma_r2', 'sigma_t2']
        new_rows = []
        for metric_old in metrics_to_transform:
            # 筛选出需要转换的行
            df_subset = l7df_sns[l7df_sns['Metric'] == metric_old].copy()
            if not df_subset.empty:
                # 计算平方根
                df_subset['Value'] = np.sqrt(df_subset['Value'])
                # 更新 Metric 名称
                metric_new = metric_old[:-1] # 去掉末尾的 '2'
                df_subset['Metric'] = metric_new
                new_rows.append(df_subset)
        if new_rows:
            # 将新数据添加到原 DataFrame
            l7df_sns = pd.concat([l7df_sns, ] + new_rows, ignore_index=True)

        return l7df_sns


class _Coll_Coal_FileProcessor(ContinousFileProcessor):
    """读取和画图前预处理coll.13和coal.24"""
    def read_file(self, simu_name, read_csv_func: Callable):
        self.concat_file(simu_name)
        logger.debug(f'Loading gathered {self.file_basename} of {simu_name} at {self.file_path}')
        df = read_csv_func(self.file_path)
        df = self.clean_data(df)
        df.insert(
            0, 'Time[Myr]', 
            df['TIME[NB]'] * self.get_scale_dict_from_stdout(simu_name)['t']
        )
        df['primary_mass[solar]'] = np.max(df[['M(I1)[M*]','M(I2)[M*]']], axis=1)
        df['secondary_mass[solar]'] = np.min(df[['M(I1)[M*]','M(I2)[M*]']], axis=1)
        df['mass_ratio'] = df['secondary_mass[solar]'] / df['primary_mass[solar]']
        df['primary_stellar_type'] = np.maximum(df['K*(I1)'], df['K*(I2)'])
        df['secondary_stellar_type'] = np.minimum(df['K*(I1)'], df['K*(I2)'])
        df['Stellar Type'] = df['primary_stellar_type'].map(self.config.kw_to_stellar_type) + '-' + \
                            df['secondary_stellar_type'].map(self.config.kw_to_stellar_type)
        return df
    
    def clean_data(self, df):
        return super().clean_data(df, timecol='TIME[NB]')
    
    def merge_coll_coal(self, df1, df2):
        """
        合并coll.13和coal.24类型df的数据. 非共同col的缺失数据自动添加NaN
        """
        # common_columns = list(set(df1.columns) & set(df2.columns))
        return pd.concat([df1, df2], ignore_index=True)


class Coll13FileProcessor(_Coll_Coal_FileProcessor):
    def __init__(self, config_manager):
        super().__init__(config_manager, file_basename='coll.13')

    def read_file(self, simu_name):
        df = super().read_file(simu_name, read_coll_13)
        df['Merger_type'] = 'collision'
        return df


class Coal24FileProcessor(_Coll_Coal_FileProcessor):
    def __init__(self, config_manager):
        super().__init__(config_manager, file_basename='coal.24')

    def read_file(self, simu_name):
        df = super().read_file(simu_name, read_coal_24)
        df['Merger_type'] = 'coalescence'
        return df


class PeTarDataFileProcessor:
    """读取和画图前预处理petar data.x"""
    def __init__(self, config_manager):
        self.config = config_manager

    @log_time(logger)
    def read_file(self, data_path, simu_name):
        """加载petar.dat数据"""
        logger.debug(f'\nProcessing {data_path=}...')
        binary_path = data_path + '.binary'
        if not os.path.exists(binary_path):
            logger.error(f"Binary data file {binary_path} does not exist. Need to run `petar.data.process` first.")
            return None
        header = petar.PeTarDataHeader(data_path, snapshot_format='binary', external_mode='galpy')
        logger.debug('Time',header.time, '\nN',header.n, '\nFile ID',header.file_id)
        logger.debug('Center position:',header.pos_offset, '\nCenter velocity:', header.vel_offset)
        single = petar.Particle(interrupt_mode='bse', external_mode='galpy') # here single means all star
        binary = petar.Binary(member_particle_type=petar.Particle, G=petar.G_MSUN_PC_MYR)
        single.fromfile(data_path, offset=petar.HEADER_OFFSET_WITH_CM) # for BINARY format. +galpy -> +with_cm 
        binary.loadtxt(binary_path, skiprows=1)

        # scalar df need to construct
        

class BaseVisualizer:
    """可视化类，处理所有绘图功能"""
    def __init__(self, config_manager):
        self.config = config_manager
        self.teff_to_rgb_converter = BlackbodyColorConverter()
        self.setup_figure_style()

    def setup_figure_style(self):
        """设置图表样式"""
        set_mpl_fonts()
        plt.rc('savefig', dpi=233)

    def luminosity_to_plot_alpha(self, L_arr):
        result = np.log10(L_arr)
        # 标记所有值为-10的元素
        special_mask = (result == -10)
        # 对数组进行归一化处理
        min_val = np.min(result)
        max_val = np.max(result)
        result = (result - min_val) / (max_val - min_val)
        # 将原始值为-10的元素设为1
        result[special_mask] = 1
        return result


class BaseHDF5Visualizer(BaseVisualizer):
    def decorate_jointfig(self, ax, data_at_t, x, y, xlim, ylim, simu_name, ttot, tmyr, t_over_tcr0, t_over_trh0, highlight_outlier=True):
        """装饰联合图"""

        x_min, x_max = data_at_t[x].min(), data_at_t[x].max()
        y_min, y_max = data_at_t[y].min(), data_at_t[y].max()
        ax.set(xlim=xlim, ylim=ylim)
        xlabel = self.config.colname_to_label.get(x, x)
        ylabel = self.config.colname_to_label.get(y, y)
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        with mpl.rc_context(**self.config.fixed_width_font_context):
            ax.set_xlabel(f"{xlabel}\n{x_min:9.1f} - {x_max:9.1f}", 
                          family=self.config.fixed_width_font_context['rc']['font.family'])
            ax.set_ylabel(f"{ylabel}\n{y_min:9.1f} - {y_max:9.1f}", 
                          family=self.config.fixed_width_font_context['rc']['font.family'])
            # 如果 min, max 超出限制，设置标签为红色
            if highlight_outlier:
                if x_min < xlim[0] or x_max > xlim[1]:
                    ax.xaxis.label.set_color('red')
                    ax.tick_params(axis='x', which='both', colors='red')
                    ax.spines['bottom'].set_color('red')
                if y_min < ylim[0] or y_max > ylim[1]:
                    ax.yaxis.label.set_color('red')
                    ax.tick_params(axis='y', which='both', colors='red')
                    ax.spines['left'].set_color('red')
            ax.figure.suptitle(
                f"{simu_name} | Time = {ttot:9.3f} NB = {tmyr:9.2f} Myr\n{t_over_tcr0:7.0f} TCR0 = {t_over_trh0:3.1f} TRH0",
                )

    def _create_jointplot_density(
        self,
        df_at_t,
        simu_name: str,
        x_col: str,
        y_col: str,
        log_scale: tuple[bool, bool],
        filename_var_part: str,
        xlim_key: str = None,
        ylim_key: str = None,
        extra_data_handler: Callable | None = None,
        extra_ax_handler: Callable | None = None,
        custom_ax_joint_decorator: Callable[[plt.Axes, pd.DataFrame], None] | None = None,
    ):
        """
        辅助函数，用于创建基于sns.jointplot的密度图。

        参数:
            df_at_t: 包含绘图数据的DataFrame。
            simu_name: 模拟名称。
            x_col: DataFrame中用作x轴的列名。
            y_col: DataFrame中用作y轴的列名。
            log_scale: 一个元组 (bool, bool)，指示x轴和y轴是否使用对数刻度。
            filename_var_part: 用于构成保存文件名的变量部分。
            xlim_key: self.config.limits中x轴限制的键名。缺省 = x_col
            ylim_key: self.config.limits中y轴限制的键名。缺省 = y_col
            extra_data_handler: 可选的回调函数，用于在绘图前处理DataFrame。
            extra_ax_handler: 可选的回调函数，用于在主要绘图后自定义联合轴 (ax_joint)。
            custom_ax_joint_decorator: 可选的回调函数，用于在decorate_jointfig之后，保存之前，
                                       对ax_joint进行更具体的绘图操作 (例如添加辅助线、文本等)。
                                       接收 ax_joint 和处理后的 df_at_t 作为参数。
        """
        if xlim_key is None:
            xlim_key = x_col
        if ylim_key is None:
            ylim_key = y_col
        ttot = df_at_t['TTOT'].iloc[0]
        tmyr = df_at_t['Time[Myr]'].iloc[0]
        t_over_tcr0 = df_at_t['TTOT/TCR0'].iloc[0]
        t_over_trh0 = df_at_t['TTOT/TRH0'].iloc[0]

        processed_df = df_at_t
        if extra_data_handler is not None:
            processed_df = extra_data_handler(df_at_t)

        base_filename = f"{self.config.figname_prefix[simu_name]}output_ttot_{ttot}_{filename_var_part}"
        save_pdf_path = f"{self.config.plot_dir}/{base_filename}.pdf"
        save_jpg_path = f"{self.config.plot_dir}/jpg/{base_filename}.jpg"

        if self.config.skip_existing_plot and os.path.exists(save_pdf_path):
            logger.debug(f"Skip existing plot: {save_pdf_path}")
            return

        g = sns.jointplot(
            data=processed_df,
            x=x_col,
            y=y_col,
            kind='hist',
            bins=100,
            log_scale=log_scale
        )

        if extra_ax_handler is not None:
            extra_ax_handler(g.ax_joint)

        self.decorate_jointfig(
            g.ax_joint,
            processed_df, # 使用处理后的DataFrame进行装饰
            x_col,
            y_col,
            self.config.limits[xlim_key],
            self.config.limits[ylim_key],
            simu_name,
            ttot, tmyr, t_over_tcr0, t_over_trh0
        )

        if custom_ax_joint_decorator is not None:
            custom_ax_joint_decorator(g.ax_joint, processed_df) # 传递处理后的DataFrame

        add_grid(g.ax_joint) 
        g.savefig(save_pdf_path)
        g.savefig(save_jpg_path)
        try:
            __IPYTHON__
            if self.config.close_figure_in_ipython:
                plt.close(g.figure)
        except NameError:
            plt.close(g.figure)

    def _symlogY_and_fill_handler(self, ax, linthresh=10):
        # log scale，but linear region around 0
        ax.set_yscale('symlog', linthresh=linthresh)
        # fill the linear region around 0
        ax.axhspan(-linthresh, linthresh, color='lightgray', alpha=0.3)


class SingleStarVisualizer(BaseHDF5Visualizer):
    @log_time(logger)
    def create_mass_distance_plot_density(
            self, single_df_at_t, simu_name,
            ):
        """创建质量-距离关系图"""
        self._create_jointplot_density(
            df_at_t=single_df_at_t,
            simu_name=simu_name,
            x_col='Distance_to_cluster_center[pc]',
            y_col='M',
            log_scale=(True, True),
            filename_var_part='mass_vs_distance_loglog',
        )
    
    @log_time(logger)
    def create_vx_x_plot_density(
            self, single_df_at_t, simu_name,
            ):
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
    
    @log_time(logger)
    def create_CMD_plot_density(
            self, single_df_at_t, simu_name,
            ):
        def _custom_decorator(ax, df):
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
    
    @log_time(logger)
    def create_position_plot_jpg(
            self, single_df_at_t, simu_name,
            filename_suffix: str | None = None,
            extra_data_handler: Callable | None = None,
            extra_ax_handler: Callable | None = None,
            custom_ax_decorator: Callable | None = None,
            uniform_color_and_size: bool = False
            ):
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
        """创建位置散点图"""
        if self.config.skip_existing_plot and os.path.exists(save_jpg_path):
            logger.debug(f"Skip existing plot: {save_jpg_path}")
            return
        color_rgb_darker = self.teff_to_rgb_converter.get_rgb(single_df_at_t['Teff*'].values) * \
            self.luminosity_to_plot_alpha(single_df_at_t['L*'].values)[:, np.newaxis]
        if not uniform_color_and_size:
            size = np.sqrt(single_df_at_t['R*'])
            color = color_rgb_darker
        else: # mostly for tidal tail visualization
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

    @log_time(logger)
    def create_position_plot_wide_pc_jpg(
            self, single_df_at_t, simu_name,
            ):
        def _set_wide_pos_lim_pc(ax):
            ax.set_xlim(*self.config.limits['position_pc_lim_MAX'])
            ax.set_ylim(*self.config.limits['position_pc_lim_MAX'])

        self.create_position_plot_jpg(
            single_df_at_t=single_df_at_t,
            simu_name=simu_name,
            filename_suffix='wide_pc',
            custom_ax_decorator=_set_wide_pos_lim_pc,
            uniform_color_and_size=True
        )

    @log_time(logger)
    def create_position_plot_hightlight_compact_objects_jpg(
            self, single_df_at_t, simu_name, 
            filename_suffix: str | None = None,
            extra_data_handler: Callable | None = None,
            extra_ax_handler: Callable | None = None,
            custom_ax_decorator: Callable | None = None
            ):
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

        fig, ax = plt.subplots() # Normal white background by default

        '''
            10: "HeWD",
            11: "COWD",
            12: "ONeWD",
            13: "NS",
            14: "BH",
        '''

        # Filter data
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
                s=np.sqrt(df_others['R*']), # Dynamic size based on R*
                color='gray',
                ax=ax
            )

        # Define plot configurations for different types of compact objects
        # Each tuple: (DataFrame, marker_style, size, color, label, alpha, edge_color)
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
                    'label': label, # Note: legend is removed later in your code
                    'alpha': 0.7,
                    'edgecolors': edgecolors,
                    'lw': lw,
                    'ax': ax
                }
                # if edgecolors:
                #     plot_kwargs['edgecolors'] = edgecolors
                
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

    @log_time(logger)
    def create_position_plot_hightlight_compact_objects_wide_pc_jpg(
            self, single_df_at_t, simu_name):
        def _set_wide_pos_lim_pc(ax):
            ax.set_xlim(*self.config.limits['position_pc_lim_MAX'])
            ax.set_ylim(*self.config.limits['position_pc_lim_MAX'])
        self.create_position_plot_hightlight_compact_objects_jpg(
            single_df_at_t=single_df_at_t,
            simu_name=simu_name,
            filename_suffix='wide_pc',
            custom_ax_decorator=_set_wide_pos_lim_pc
        )

    @log_time(logger)
    def create_color_CMD_jpg(
            self, single_df_at_t, simu_name,
            extra_data_handler: Callable | None = None,
            extra_ax_handler: Callable | None = None
            ):
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
        """创建彩色HR图"""
        ax = sns.scatterplot(
            data=single_df_at_t, 
            x='Teff*', 
            y='L*', 
            hue='Stellar Type',
            hue_order=all_stellar_types_sorted,
            style='Stellar Type',
            palette=self.config.st_verbose_to_color,
            s=20,
            # edgecolors='none',
            linewidths=0.8, # default=1.5
            markers=self.config.star_type_verbose_to_marker,
            legend='full',
            # palette=sns.color_palette("husl", 17),
            # color=self.teff_to_rgb_converter.get_rgb(single_df_at_t['Teff*'].values), 
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

        ##解决因为部分stellar type时有时无 导致legend长度抽出难以观看
        ##思路：展示完整的legend，但对于当前不存在的st使用透明占位符legend
        # 创建自定义图例句柄
        legend_handles = []
        # 占位符标记的颜色 (全透明)
        placeholder_marker_color = (0, 0, 0, 0.0) 
        # scatterplot中的s=20大致对应Line2D中的markersize=sqrt(20)
        legend_marker_size = np.sqrt(20) 
        legend_marker_edge_width = 0.8 # 与scatterplot中的linewidths一致

        current_stellar_types = set(single_df_at_t['Stellar Type'].unique())
        for st_type in all_stellar_types_sorted:
            # 从配置中获取标记形状，如果找不到则默认为'+'
            marker_shape = self.config.star_type_verbose_to_marker.get(st_type, '+') 

            if st_type in current_stellar_types:
                label = st_type
                # 从配置中获取颜色，如果找不到则默认为黑色
                color = self.config.st_verbose_to_color.get(st_type, 'black')
                mec = color
            else:
                # 对于不存在的类型，使用占位符
                label = " "*8 # 长度=最长的可能长度（12:ONeWD)，避免图例宽度抽搐
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

        # with mpl.rc_context(**self.config.fixed_width_font_context):
        #     plt.legend(title='Stellar Type', fontsize=11, title_fontsize=11, bbox_to_anchor=(1, 1), loc='upper left')
        ax.invert_xaxis()
        ax.figure.savefig(save_jpg_path)
        try:
            __IPYTHON__
            if self.config.close_figure_in_ipython:
                plt.close(ax.figure)
        except NameError:
            plt.close(ax.figure)


class BinaryStarVisualizer(BaseHDF5Visualizer):
    def _create_base_jpg_plot_compact_object_only(
        self,
        binary_df_at_t, 
        simu_name: str,
        x_col: str,
        y_col: str,
        log_scale: tuple[bool, bool],
        filename_var_part: str,
        xlim_key: str = None,
        ylim_key: str = None,
        extra_ax_handler: Optional[Callable[[plt.Axes], None]] = None,
        custom_ax_decorator: Optional[Callable[[plt.Axes, pd.DataFrame], None]] = None,
    ):
        """
        辅助函数，用于创建基本的JPG散点图或线图。

        参数:
            binary_df_at_t
            simu_name: 模拟名称。
            x_col: DataFrame中用作x轴的列名。
            y_col: DataFrame中用作y轴的列名。
            log_scale: 一个元组 (bool, bool)，指示x轴和y轴是否使用对数刻度。
            filename_var_part: 用于构成保存文件名的变量部分。
            xlim_key: self.config.limits中x轴限制的键名。缺省 = x_col
            ylim_key: self.config.limits中y轴限制的键名。缺省 = y_col
            extra_ax_handler: 可选的回调函数，用于在主要绘图后自定义轴。
            custom_ax_decorator: 可选的回调函数，用于在decorate_jointfig之后，保存之前，
                                       对ax进行更具体的绘图操作 (例如添加辅助线、文本等)。
                                       接收 ax 和处理后的 df_at_t 作为参数。
        """
        if xlim_key is None:
            xlim_key = x_col
        if ylim_key is None:
            ylim_key = y_col
        ttot = binary_df_at_t['TTOT'].iloc[0]
        tmyr = binary_df_at_t['Time[Myr]'].iloc[0]
        t_over_tcr0 = binary_df_at_t['TTOT/TCR0'].iloc[0]
        t_over_trh0 = binary_df_at_t['TTOT/TRH0'].iloc[0]

        save_jpg_path = f"{self.config.plot_dir}/jpg/{self.config.figname_prefix[simu_name]}output_ttot_{ttot}_{filename_var_part}.jpg"
        if self.config.skip_existing_plot and os.path.exists(save_jpg_path):
            logger.debug(f"Skip existing plot: {save_jpg_path}")
            return
        
        binary_df_at_t_cbo = self.binary_df_compact_object_filter(binary_df_at_t)
        vc = binary_df_at_t_cbo['Stellar Type'].value_counts()
        # convert series to df, index as a column
        vc_df = vc.reset_index()

        fig, ax = plt.subplots()
        for _st in binary_df_at_t_cbo['Stellar Type'].unique():
            sns.lineplot(
                data=binary_df_at_t_cbo[binary_df_at_t_cbo['Stellar Type'] == _st],
                x=x_col,
                y=y_col,
                lw=0,
                **self.get_binary_cookie_dict(
                    _st.split('-')[0], _st.split('-')[1]
                ),
                legend=False,
                ax=ax
            )
        if log_scale[0]:
            ax.set(xscale='log')
        if log_scale[1]:
            ax.set(yscale='log')
        
        legend_elements = []
        if len(binary_df_at_t_cbo) > 0: # not empty plot
            # add table with binary type counts
            _table = ax.table(
                cellText=vc_df.values, colLabels=vc_df.columns, 
                colWidths=[0.18, 0.1],
                colLoc='right',
                loc='lower left',
                # bbox=mpl.transforms.Bbox.from_bounds(1.02, 0, 0.4, 0.3)
                )

            # Add legend manually
            stellar_kws = np.unique(np.concatenate((binary_df_at_t_cbo['Bin KW1'].unique(), binary_df_at_t_cbo['Bin KW2'].unique())))
            # Create legend elements for each binary type
            for _kw in stellar_kws:
                legend_elements.append(
                    plt.Line2D([0], [0], marker='o', lw=0,
                                markerfacecolor=self.config.palette_st[_kw+1],
                                markeredgecolor='black',
                                markersize=10, 
                                alpha=0.6,
                                label=self.config.kw_to_stellar_type[_kw])
                )
        else:
            pass # no data, no table, empty plot
        # 无论是否有没有legend，追加一个空的最长的legend元素，避免图例宽度抽搐
        _emptylabel = " "*5 # 长度=最长的可能长度（ONeWD)
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', lw=0,
                        markerfacecolor='none',
                        markeredgecolor='none',
                        markersize=10, 
                        alpha=0.6,
                        label=_emptylabel)
        )
        with mpl.rc_context(**self.config.fixed_width_font_context):
            ax.legend(handles=legend_elements, loc='upper left', frameon=True, 
                        title='Stellar Types', bbox_to_anchor=(1, 1))

        if extra_ax_handler is not None:
            extra_ax_handler(ax)
        
        self.decorate_jointfig(
            ax,
            binary_df_at_t_cbo,
            x_col,
            y_col,
            self.config.limits[xlim_key],
            self.config.limits[ylim_key],
            simu_name,
            ttot, tmyr, t_over_tcr0, t_over_trh0
        )

        if custom_ax_decorator is not None:
            custom_ax_decorator(ax, binary_df_at_t_cbo) # 传递处理后的DataFrame

        add_grid(ax) # 假设 add_grid 是一个全局可用的函数
        fig.savefig(save_jpg_path)
        try:
            __IPYTHON__
            if self.config.close_figure_in_ipython:
                plt.close(ax.figure)
        except NameError:
            plt.close(fig)

    @log_time(logger)
    def binary_df_compact_object_filter(self, df):
        if 'KW' in df.columns:
            compact_object_mask = df['KW'].isin(self.config.compact_object_KW)
        elif 'Bin KW1' in df.columns and 'Bin KW2' in df.columns:
            compact_object_mask = df['Bin KW1'].isin(self.config.compact_object_KW) | df['Bin KW2'].isin(self.config.compact_object_KW)
        else:
            raise ValueError("DataFrame does not contain 'KW' or 'Bin KW1/Bin KW2' columns. Columns: " + str(df.columns))
        return df[compact_object_mask]
    
    @log_time(logger)
    def get_binary_cookie_dict(self, st1, st2):
        return self.get_binary_cookie_dict_num(
            self.config.stellar_type_to_kw[st1], 
            self.config.stellar_type_to_kw[st2])

    @log_time(logger)
    def get_binary_cookie_dict_num(self, kw1, kw2):
        cookie_tastes = self.config.palette_st
        sts = self.config.kw_to_stellar_type[kw1] + '-' + self.config.kw_to_stellar_type[kw2]
        fs = 'top' if sts in self.config.wow_binary_st_list else 'left'
        ms = 15 if sts in self.config.wow_binary_st_list else 10
        return dict(marker=self.config.marker_fill_list[kw1 + 1], 
                    markerfacecolor=cookie_tastes[kw1 + 1],
                    markerfacecoloralt=cookie_tastes[kw2 + 1],
                    markeredgecolor='black',
                    fillstyle=fs, ms=ms, alpha=0.6,
                    )

    @log_time(logger)
    def create_mass_ratio_m1_plot_density(
            self, binary_df_at_t, simu_name):
        """创建质量比图"""
        self._create_jointplot_density(
            df_at_t=binary_df_at_t,
            simu_name=simu_name,
            x_col='primary_mass[solar]',
            y_col='mass_ratio',
            log_scale=(True, True),
            xlim_key='M',
            filename_var_part='mass_ratio_vs_primary_mass_loglog',
        )

    @log_time(logger)
    def create_mass_ratio_m1_plot_jpg_compact_object_only(
            self, binary_df_at_t, simu_name):
        self._create_base_jpg_plot_compact_object_only(
            binary_df_at_t,
            simu_name=simu_name,
            x_col='primary_mass[solar]',
            y_col='mass_ratio',
            log_scale=(True, False),
            xlim_key='M',
            filename_var_part='mass_ratio_vs_primary_mass_loglog_compact_objects_only'
        )

    def _decorator_semi_m1(self, ax, df): # df is processed_df from helper
        ax.axhline(0.00465, color='darkred', linestyle='--', label='Solar radius')
        ax.text(
            ax.get_xlim()[-1], 
            0.00465, 
            'Solar radius', 
            color='darkred', 
            fontsize=10, 
            horizontalalignment='right'
        )

    @log_time(logger)
    def create_semi_m1_plot_density(
            self, binary_df_at_t, simu_name):
        """创建半长轴-主星质量图"""
        self._create_jointplot_density(
            df_at_t=binary_df_at_t,
            simu_name=simu_name,
            x_col='primary_mass[solar]',
            y_col='Bin A[au]',
            log_scale=(True, True),
            xlim_key='M',
            filename_var_part='a_vs_primary_mass_loglog',
            custom_ax_joint_decorator=self._decorator_semi_m1
        )

    @log_time(logger)
    def create_semi_m1_plot_jpg_compact_object_only(
            self, binary_df_at_t, simu_name):
        self._create_base_jpg_plot_compact_object_only(
            binary_df_at_t,
            simu_name=simu_name,
            x_col='primary_mass[solar]',
            y_col='Bin A[au]',
            log_scale=(True, True),
            xlim_key='M',
            filename_var_part='a_vs_primary_mass_loglog_compact_objects_only',
            custom_ax_decorator=self._decorator_semi_m1
        )

    def _decorator_ebin_semi(self, ax, df): # df is processed_df from helper
        ax.axhline(y=1, linestyle='--', color='darkred', linewidth=1.5)
        # calc hard and soft binary based on the potentially filtered/modified df
        hard_num = np.sum(df['is_hard_binary'])
        soft_num = len(df) - hard_num
        hard_frac, soft_frac = (hard_num / len(df), soft_num / len(df)) if len(df) > 0 else (0.0, 0.0)
        
        xmax = ax.get_xlim()[1]
        with mpl.rc_context(**self.config.fixed_width_font_context):
            ax.text(xmax, 1.0, f'{hard_num}\n{hard_frac:.1%} hard', color='darkred', ha='right', va='bottom')
            ax.text(xmax, 0.9, f'{soft_frac:.1%} soft\n{soft_num}', color='darkred', ha='right', va='top')

    @log_time(logger)
    def create_ebind_semi_plot_density(
            self, binary_df_at_t, simu_name):
        """创建绑定能-半长轴图"""

        self._create_jointplot_density(
            df_at_t=binary_df_at_t,
            simu_name=simu_name,
            x_col='Bin A[au]',
            y_col='Ebind/kT',
            log_scale=(True, True),
            filename_var_part='ebind_vs_a_loglog',
            custom_ax_joint_decorator=self._decorator_ebin_semi
        )

    @log_time(logger)
    def create_ebind_semi_plot_jpg_compact_object_only(
            self, binary_df_at_t, simu_name):

        self._create_base_jpg_plot_compact_object_only(
            binary_df_at_t,
            simu_name=simu_name,
            x_col='Bin A[au]',
            y_col='Ebind/kT',
            log_scale=(True, True),
            filename_var_part='ebind_vs_a_loglog_compact_objects_only',
            custom_ax_decorator=self._decorator_ebin_semi
        )
    
    @log_time(logger)
    def create_ecc_semi_plot_density(
            self, binary_df_at_t, simu_name):
        """创建偏心率-半长轴图"""
        self._create_jointplot_density(
            df_at_t=binary_df_at_t,
            simu_name=simu_name,
            x_col='Bin A[au]',
            y_col='Bin ECC',
            log_scale=(True, False),
            filename_var_part='ecc_vs_a', 
        )
    
    @log_time(logger)
    def create_ecc_semi_plot_jpg_compact_object_only(
            self, binary_df_at_t, simu_name):
        self._create_base_jpg_plot_compact_object_only(
            binary_df_at_t,
            simu_name=simu_name,
            x_col='Bin A[au]',
            y_col='Bin ECC',
            log_scale=(True, False),
            filename_var_part='ecc_vs_a_compact_objects_only',
        )
    
    @log_time(logger)
    def create_ecc_semi_plot_jpg_compact_object_only_loglog(
            self, binary_df_at_t, simu_name):
        self._create_base_jpg_plot_compact_object_only(
            binary_df_at_t,
            simu_name=simu_name,
            x_col='Bin A[au]',
            y_col='Bin ECC',
            log_scale=(True, True),
            filename_var_part='ecc_vs_a_loglog_compact_objects_only',
        )

    @log_time(logger)
    def create_taugw_semi_plot_jpg_compact_object_only(
            self, binary_df_at_t, simu_name):
        self._create_base_jpg_plot_compact_object_only(
            binary_df_at_t,
            simu_name=simu_name,
            x_col='Bin A[au]',
            y_col='tau_gw[Myr]',
            log_scale=(True, True),
            filename_var_part='taugw_vs_a_compact_objects_only',
        )

    @log_time(logger)
    def create_mtot_distance_plot_density(
            self, binary_df_at_t, simu_name):
        """创建总质量-距离关系密度图"""
        self._create_jointplot_density(
            df_at_t=binary_df_at_t,
            simu_name=simu_name,
            x_col='Distance_to_cluster_center[pc]',
            y_col='total_mass[solar]',
            log_scale=(True, True),
            ylim_key='M',
            filename_var_part='mtot_vs_distance_loglog',
        )

    @log_time(logger)
    def create_mtot_distance_plot_jpg_compact_object_only(
            self, binary_df_at_t, simu_name):
        """创建仅包含致密天体的总质量-距离关系图"""
        self._create_base_jpg_plot_compact_object_only(
            binary_df_at_t,
            simu_name=simu_name,
            x_col='Distance_to_cluster_center[pc]',
            y_col='total_mass[solar]',
            log_scale=(True, True),
            ylim_key='M',
            filename_var_part='mtot_vs_distance_loglog_compact_objects_only',
        )

    @log_time(logger)
    def create_semi_distance_plot_density(
            self, binary_df_at_t, simu_name):
        """创建半长轴-距离关系密度图"""
        self._create_jointplot_density(
            df_at_t=binary_df_at_t,
            simu_name=simu_name,
            x_col='Distance_to_cluster_center[pc]',
            y_col='Bin A[au]',
            log_scale=(True, True),
            filename_var_part='a_vs_distance',
        )
    
    @log_time(logger)
    def create_semi_distance_plot_jpg_compact_object_only(
            self, binary_df_at_t, simu_name):
        """创建仅包含致密天体的半长轴-距离关系图"""
        self._create_base_jpg_plot_compact_object_only(
            binary_df_at_t,
            simu_name=simu_name,
            x_col='Distance_to_cluster_center[pc]',
            y_col='Bin A[au]',
            log_scale=(True, True),
            filename_var_part='a_vs_distance_compact_objects_only',
        )

    @log_time(logger)
    def create_bin_vx_x_plot_density(
            self, binary_df_at_t, simu_name):
        """创建二元星系统的vx-x关系密度图"""
        self._create_jointplot_density(
            df_at_t=binary_df_at_t,
            simu_name=simu_name,
            x_col='Bin cm X [pc]',
            y_col='Bin cm V1',
            log_scale=(False, False),
            filename_var_part='bin_vx_vs_x',
            xlim_key='position_pc_lim',
            ylim_key='velocity_kmps_lim',
        )

    @log_time(logger)
    def create_bin_vx_x_plot_jpg_compact_object_only(
            self, binary_df_at_t, simu_name):
        """创建仅包含致密天体的vx-x关系图"""
        self._create_base_jpg_plot_compact_object_only(
            binary_df_at_t,
            simu_name=simu_name,
            x_col='Bin cm X [pc]',
            y_col='Bin cm V1',
            log_scale=(False, False),
            xlim_key='position_pc_lim',
            ylim_key='velocity_kmps_lim',
            filename_var_part='bin_vx_vs_x_compact_objects_only',
        )
    
    # def _ax_handler_peri_r1r2(self, ax):
    #     # 画一条x=y的线，在

    def create_bin_peri_r1r2_plot_density(
            self, binary_df_at_t, simu_name):
        """创建近星点距离-半径之和关系密度图"""
        self._create_jointplot_density(
            df_at_t=binary_df_at_t,
            simu_name=simu_name,
            x_col='sum_of_radius[au]',
            y_col='peri[au]',
            log_scale=(True, True),
            filename_var_part='peri_vs_r1r2_loglog',
        )


class HDF5Visualizer():
    def __init__(self, config_manager):
        self.single = SingleStarVisualizer(config_manager)
        self.binary = BinaryStarVisualizer(config_manager)


class BaseContinousFileVisualizer(BaseVisualizer):
    pass


class LagrVisualizer(BaseContinousFileVisualizer):
    '''
    所有lagr图默认每次都重绘，不考虑skip_existing_plot参数
    '''
    def __init__(self, config_manager):
        super().__init__(config_manager)
        self.metric_to_plot_label = {
            'rlagr': 'Lagrangian radii [pc]',
            'rlagr_s': 'Lagrangian radii of single stars [pc]',
            'rlagr_b': 'Lagrangian radii of binary stars [pc]',
            'avmass': 'Average mass [Msolar]',
            'nshell': 'Number of stars',
            'vx': 'Mass weighted X velocity [km/s]',
            'vy': 'Mass weighted Y velocity [km/s]',
            'vz': 'Mass weighted Z velocity [km/s]',
            'v': 'Mass weighted velocity [km/s]',
            'vr': 'Mass weighted radial velocity [km/s]',
            'vt': 'Mass weighted tangential velocity [km/s]',
            'sigma2': 'Mass weighted\nvelocity dispersion squared [${km}^2~s^{-2}$]',
            'sigma': 'Mass weighted velocity dispersion [km/s]',
            'sigma_r2': 'Mass weighted\nradial velocity dispersion squared [${km}^2~s^{-2}$]',
            'sigma_r': 'Mass weighted radial velocity dispersion [km/s]',
            'sigma_t2': 'Mass weighted\ntangential velocity dispersion squared [${km}^2~s^{-2}$]',
            'sigma_t': 'Mass weighted tangential velocity dispersion [km/s]',
            'vrot': 'Mass weighted\nrotational velocity [km/s]',
        }

    def create_lagr_plot_base(
            self, l7df_sns, simu_name, metric='rlagr', 
            filename_suffix: str | None = None,
            extra_ax_handler: Callable | None = None):
        if filename_suffix is None:
            save_pdf_path = f"{self.config.plot_dir}/{self.config.figname_prefix[simu_name]}_{metric}.pdf"
        else:
            save_pdf_path = f"{self.config.plot_dir}/{self.config.figname_prefix[simu_name]}_{metric}_{filename_suffix}.pdf"
        l7df_sns_selected_metric = l7df_sns[(l7df_sns['Metric'] == metric) & (l7df_sns['%'].isin(self.config.selected_lagr_percent))]
        # remove t=0 data
        l7df_sns_selected_metric = l7df_sns_selected_metric[l7df_sns_selected_metric['Time[Myr]'] > 0]
        ax = sns.lineplot(data=l7df_sns_selected_metric, 
                    x='Time[Myr]', y='Value', hue='%')
        ax.set(
            yscale='log',
            ylabel=self.metric_to_plot_label[metric],
            title=simu_name
            )
        if extra_ax_handler is not None:
            extra_ax_handler(ax)
        
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        add_grid(ax)
        ax.figure.savefig(save_pdf_path)
        try:
            __IPYTHON__
            if self.config.close_figure_in_ipython:
                plt.close(ax.figure)
        except NameError:
            plt.close(ax.figure)

    def _extra_ax_handler_rlagr(self, ax):
        ax.set(ylim=(ax.get_ylim()[0], 10.05))

    def _extra_ax_handler_logx(self, ax):
        ax.set_xscale('log')

    def _extra_ax_handler_rlagr_logx(self, ax):
        self._extra_ax_handler_rlagr(ax)
        self._extra_ax_handler_logx(ax)

    def create_lagr_radii_plot(self, l7df_sns, simu_name):
        self.create_lagr_plot_base(
            l7df_sns, simu_name, metric='rlagr',
            extra_ax_handler=self._extra_ax_handler_rlagr)
        self.create_lagr_plot_base(
            l7df_sns, simu_name, metric='rlagr', filename_suffix='loglog',
            extra_ax_handler=self._extra_ax_handler_rlagr_logx)
    
    def create_lagr_avmass_plot(self, l7df_sns, simu_name):
        self.create_lagr_plot_base(l7df_sns, simu_name, metric='avmass')
        self.create_lagr_plot_base(l7df_sns, simu_name, metric='avmass', filename_suffix='loglog', extra_ax_handler=self._extra_ax_handler_logx)
    
    def create_lagr_velocity_dispersion_plot(self, l7df_sns, simu_name):
        self.create_lagr_plot_base(l7df_sns, simu_name, metric='sigma')
        self.create_lagr_plot_base(l7df_sns, simu_name, metric='sigma', filename_suffix='loglog', extra_ax_handler=self._extra_ax_handler_logx)


class CollCoalVisualizer(BaseContinousFileVisualizer):
    def two_bh_filter(self, df):
        return df[(df['primary_stellar_type'] == 14) & (df['secondary_stellar_type'] == 14)]
    
    def two_cbo_fileter(self, df):
        return df[(df['primary_stellar_type'].isin(self.config.compact_object_KW)) & 
                  (df['secondary_stellar_type'].isin(self.config.compact_object_KW))]
    
    def create_mass_ratio_primary_plot_cbo(self, df, simu_name):
        df_cbo = self.two_cbo_fileter(df)

        ax = sns.scatterplot(data=df_cbo, x='primary_mass[solar]', y='mass_ratio', hue='Stellar Type', style='Merger_type', alpha=0.5)

        # put time of each point below the point
        marker_size = mpl.rcParams['lines.markersize']
        for i, row in df_cbo.iterrows():
            ax.text(row['primary_mass[solar]'], row['mass_ratio'] - 0.002*marker_size, f"{row['Time[Myr]']:.2f}", fontsize=8, ha='center', va='top')
        # put a note on the bottom right corner to show what is the text
        if len(df_cbo) > 0:
            ax.text(1.01, 0, 'Marked text is time[Myr]\n  of each event in simulations', 
                    transform=ax.transAxes, fontsize=10, ha='left', va='bottom', 
                    color='black', fontstyle='italic')

        gwtc_df = load_GWTC_catalog()
        # errorbar 函数需要误差的绝对值（对于 lower error）
        # yerr 可以是 [yerr_lower_absolute, yerr_upper_absolute]
        y_err_lower_abs = np.abs(gwtc_df['mass_ratio_lower'])
        y_err_upper_abs = gwtc_df['mass_ratio_upper']

        ax.errorbar(
            x=gwtc_df['mass_1_source'], y=gwtc_df['mass_ratio'],
            yerr=[y_err_lower_abs, y_err_upper_abs],
            fmt='o', markersize=5,
            capsize=3, 
            alpha=0.2, color='gray',
            label='GWTC Data'
        )

        # fill x = 40 - 150 with pink
        ax.fill_betweenx(y=ax.get_ylim(), x1=40, x2=150,  color='pink', alpha=0.2)

        ax.set(xscale='log', xlim=(2.5, 500), ylim=(0, 1.05))
        add_grid(ax)

        ax.legend(bbox_to_anchor=(1, 1), loc='upper left')

        ax.figure.savefig(f"{self.config.plot_dir}/{self.config.figname_prefix[simu_name]}_merger_mass_ratio_vs_primary_mass_2cbo.pdf")
        try:
            __IPYTHON__
            if self.config.close_figure_in_ipython:
                plt.close(ax.figure)
        except NameError:
            plt.close(ax.figure)


class SimulationPlotter:
    """模拟处理类，管理整个模拟处理流程"""
    def __init__(self, config_manager):
        self.config = config_manager
        self.hdf5_file_processor = HDF5FileProcessor(config_manager)
        self.lagr_file_processor = LagrFileProcessor(config_manager)
        self.hdf5_visualizer = HDF5Visualizer(config_manager)
        self.lagr_visualizer = LagrVisualizer(config_manager)

    def plot_hdf5_snapshots(self, hdf5_snap_path, simu_name):
        """处理单个快照文件"""
        # 获取快照时间
        t_nbody_in_filename = self.hdf5_file_processor.get_snapshot_time(hdf5_snap_path)
        if t_nbody_in_filename < self.config.skip_until_of[simu_name]:
            logger.debug("skipped")
            return
        
        # 加载数据
        df_dict = self.hdf5_file_processor.read_file(hdf5_snap_path, simu_name)
        
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
                logger.info(f"Warning: {simu_name} {hdf5_snap_path} {ttot=} data validation failed, skipping")
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

    def plot_lagr(self, simu_name):
        l7df_sns = self.lagr_file_processor.load_sns_friendly_data(simu_name)
        self.lagr_visualizer.create_lagr_radii_plot(l7df_sns, simu_name)
        self.lagr_visualizer.create_lagr_avmass_plot(l7df_sns, simu_name)
        self.lagr_visualizer.create_lagr_velocity_dispersion_plot(l7df_sns, simu_name)
        plt.close('all')
        gc.collect()


    def plot_all_simulations(self):
        """处理所有模拟"""
        for simu_name in self.config.pathof.keys():
            # 先画lagr
            self.plot_lagr(simu_name)

            # 获取所有快照文件
            hdf5_snap_files = sorted(
                glob(self.config.pathof[simu_name] + '/**/*.h5part'), 
                key=lambda x: int(x.split('snap.40_')[1].split('.h5part')[0])
            )
            # 如果最后一个快照的修改时间在24小时之内，则从列表中删除，避免读取一个正在跑的模拟
            WAIT_SNAPSHOT_AGE_HOUR = 24
            if hdf5_snap_files and os.path.getmtime(hdf5_snap_files[-1]) > time.time() - WAIT_SNAPSHOT_AGE_HOUR * 3600:
                logger.info(f"Last snapshot of {simu_name}: {hdf5_snap_files.pop()} is created within {WAIT_SNAPSHOT_AGE_HOUR}h, skipping it.")
            
            # 创建带固定参数的部分函数
            process_file_partial = functools.partial(
                self.plot_hdf5_snapshots,
                simu_name=simu_name
            )
            
            # 使用进程池并行处理
            with multiprocessing.Pool(
                processes=self.config.processes_count, 
                maxtasksperchild=self.config.tasks_per_child
            ) as pool:
                list(
                    tqdm(
                        pool.imap(process_file_partial, hdf5_snap_files), 
                        total=len(hdf5_snap_files), 
                        desc=f'{simu_name} HDF5 Snap#'
                    )
                )


def main():
    try:
        long_options = ["skip-until=", '--debug'] 
        opts, args = getopt.getopt(sys.argv[1:], "k:", long_options)
        if '--debug' in dict(opts):
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
    except getopt.GetoptError as err:
        print(err) 
        sys.exit(2)

    config = ConfigManager(opts)
    
    # 初始化处理器
    plotter = SimulationPlotter(config)
    
    # 处理所有模拟
    plotter.plot_all_simulations()

if __name__ == "__main__":
    main()
