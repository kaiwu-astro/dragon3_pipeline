#!/usr/bin/env python3
# Standard library
import datetime
import gzip
import io
import os
import pickle as pk
import re
import signal
import subprocess
import sys
import time
import warnings
from contextlib import contextmanager
from functools import lru_cache, wraps
from glob import glob
from subprocess import Popen, call, run

# Third-party libraries
import astropy
import astropy.constants as constants
import astropy.units as u
from astropy.units.quantity import Quantity
import h5py
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import colour
from pandas.api.types import is_list_like
from scipy.interpolate import interp1d
from scipy.io import FortranFile
from tqdm.auto import tqdm
try:
    from colour.colorimetry import SpectralDistribution, msds_to_XYZ, planck_law
    from colour.models import XYZ_to_sRGB
except:
    pass

pc_to_AU = constants.pc.to(u.AU).value

def save(datadir:str, vars: list, fname='save.pkl'):
    '''datadir可以直接是pkl的路径, 此时fname不生效'''
    if datadir.endswith('.pkl'):
        path = datadir
    else:
        path = datadir + "/" + fname
    if path.endswith('.gz'):
        with gzip.open(path, "wb") as f:
            pk.dump(vars, f)
    else:
        with open(path, 'wb') as f:
            pk.dump(vars, f)
    print(f"saved {path} uses {os.path.getsize(path)/1024**2:.2f}MB")
    

def read(datadir:str, fname='save.pkl') -> list:
    '''datadir可以直接是pkl的路径, 此时fname不生效'''
    if datadir.endswith('.pkl'):
        path = datadir
    else:
        path = datadir + "/" + fname
    if path.endswith('.gz'):
        with gzip.open(path, "rb") as f:
            return pk.load(f)
    else:
        with open(path , 'rb') as f:
            return pk.load(f)

def get_output(cmd:str, raise_error=False) -> list:
    '''
    get output of a shell cmd. ignore error by default
    return: list of strings (one line per item, no \\n in the end)
    '''
    return str(run(cmd, shell=True, check=raise_error, capture_output=True).stdout, encoding='utf-8').split("\n")[:-1]

def can_convert_to_float(s: str) -> bool:
    """
    检查字符串是否可以转换为浮点数。
    
    参数:
        s (str): 要检查的字符串。
        
    返回:
        bool: 如果可以转换为浮点数，返回True；否则返回False。
    """
    try:
        float(s)
        return True
    except ValueError:
        return False

def get_scale_dict(stdout_path):
    scaling_line = get_output(
            f'grep "PHYSICAL SCALING" {stdout_path}')
    scaling_splitted = scaling_line[0].split(":")[-1].split()
    scaling_dict = {}
    for i in range(0,len(scaling_splitted),3):
        scaling_dict[scaling_splitted[i]] = float(scaling_splitted[i+2])
    # rename key of the dict: R* -> r, V* -> v, M* -> m, T* -> t
    scaling_dict['r'] = scaling_dict.pop('R*')
    scaling_dict['v'] = scaling_dict.pop('V*')
    scaling_dict['m'] = scaling_dict.pop('M*')
    scaling_dict['t'] = scaling_dict.pop('T*')
    return scaling_dict

def get_scale_dict_from_hdf5_df(scalar_df: pd.DataFrame):
    '''scalar_df is from dataframes_from_hdf5_file() function'''
    return {
        'r': scalar_df['RBAR'].values[0],
        'v': scalar_df['VSTAR'].values[0],
        'm': scalar_df['ZMBAR'].values[0],
        't': scalar_df['TSCALE'].values[0]
    }

def load_snapshot_data(hdf5_file_path, step_key):
    """从HDF5文件加载指定时间步的数据，并分类组织"""
    with h5py.File(hdf5_file_path, 'r') as f:
        step_group = f[step_key]
        
        # 加载标量数据
        scalar_data = {k: step_group['000 Scalars'][i] for i, k in 
                      enumerate(['TTOT', 'NPAIRS', 'RBAR', 'ZMBAR', 'N', 'TSTAR', 'RDENS(1)', 'RDENS(2)', 'RDENS(3)', 'TTOT/TCR0', 'TSCALE', 'VSTAR', 'RC', 'NC', 'VC', 'RHOM', 'CMAX', 'RSCALE', 'RSMIN', 'DMIN1', 'RG(1)', 'RG(2)', 'RG(3)', 'VG(1)', 'VG(2)', 'VG(3)', 'TIDAL(1)', 'TIDAL(2)', 'TIDAL(3)', 'TIDAL(4)', 'GMG', 'OMEGA', 'DISK', 'A', 'B', 'ZMET', 'ZPARS(1)', 'ZPARS(2)', 'ZPARS(3)', 'ZPARS(4)', 'ZPARS(5)', 'ZPARS(6)', 'ZPARS(7)', 'ZPARS(8)', 'ZPARS(9)', 'ZPARS(10)', 'ZPARS(11)', 'ZPARS(12)', 'ZPARS(13)', 'ZPARS(14)', 'ZPARS(15)', 'ZPARS(16)', 'ZPARS(17)', 'ZPARS(18)', 'ZPARS(19)', 'ZPARS(20)', 'ETAI', 'ETAR', 'ETAU', 'ECLOSE', 'DTMIN', 'RMIN', 'GMIN', 'GMAX', 'SMAX', 'NNBOPT', 'EPOCH0', 'N_SINGLE', 'N_BINARY', 'N_MERGER']
                                )}
        time_value = scalar_data['TTOT']

        # 加载单星数据
        single_cols = ['001 X1', '002 X2', '003 X3', '004 V1', '005 V2', '006 V3', '007 A1', '008 A2', '009 A3', '010 AD1', '011 AD2', '012 AD3', '013 D21', '014 D22', '015 D23', '016 D31', '017 D32', '018 D33', '019 STEP', '020 STEPR', '021 T0', '022 T0R', '023 M', '024 NB-Sph', '025 POT', '026 R*', '027 L*', '028 Teff*', '029 RC*', '030 MC*', '031 KW', '032 Name', '033 Type', '035 ASPN', '036 TEV', '037 TEV0', '038 EPOCH']
        single_data = {col.split(' ', 1)[1]: np.array(step_group[col]) 
                      for col in single_cols if col in step_group}
        
        # 同样加载双星和merger数据
        binary_cols = ['101 Bin cm X1', '102 Bin cm X2', '103 Bin cm X3', '104 Bin cm V1', '105 Bin cm V2', '106 Bin cm V3', '107 Bin cm A1', '108 Bin cm A2', '109 Bin cm A3', '110 Bin cm AD1', '111 Bin cm AD2', '112 Bin cm AD3', '113 Bin cm D21', '114 Bin cm D22', '115 Bin cm D23', '116 Bin cm D31', '117 Bin cm D32', '118 Bin cm D33', '119 Bin cm STEP', '120 Bin cm STEPR', '121 Bin cm T0', '122 Bin cm T0R', '123 Bin M1*', '124 Bin M2*', '125 Bin rel X1', '126 Bin rel X2', '127 Bin rel X3', '128 Bin rel V1', '129 Bin rel V2', '130 Bin rel V3', '131 Bin rel A1', '132 Bin rel A2', '133 Bin rel A3', '134 Bin rel AD1', '135 Bin rel AD2', '136 Bin rel AD3', '137 Bin rel D21', '138 Bin rel D22', '139 Bin rel D23', '140 Bin rel D31', '141 Bin rel D32', '142 Bin rel D33', '143 Bin POT', '144 Bin RS1*', '145 Bin L1*', '146 Bin Teff1*', '147 Bin RS2*', '148 Bin L2*', '149 Bin Teff2*', '150 Bin RC1*', '151 Bin MC1*', '152 Bin RC2*', '153 Bin MC2*', '154 Bin A[au]', '155 Bin ECC', '156 Bin P[d]', '157 Bin G', '158 Bin KW1', '159 Bin KW2', '160 Bin cm KW', '161 Bin Name1', '162 Bin Name2', '163 Bin cm Name', '164 ASPN1', '165 ASPN2', '166 TEV1', '167 TEV2', '168 TEV01', '169 TEV02', '170 EPOCH1', '171 EPOCH2', '176 Bin Label', '176 Bin cm Name']
        binary_data = {col.split(' ', 1)[1]: np.array(step_group[col]) 
                      for col in binary_cols if col in step_group}
        # '176 Bin cm Name' 是为了将就Rainer中间引入的bug...
        if "176 Bin cm Name" in binary_data:
            binary_data["176 Bin Label"] = binary_data.pop("176 Bin cm Name")
        
        merger_cols = ['201 Mer XC1', '202 Mer XC2', '203 Mer XC3', '204 Mer VC1', '205 Mer VC2', '206 Mer VC3', '207 Mer M1', '208 Mer M2', '209 Mer M3', '210 Mer XR01', '211 Mer XR02', '212 Mer XR03', '213 Mer VR01', '214 Mer VR02', '215 Mer VR03', '216 Mer XR11', '217 Mer XR12', '218 Mer XR13', '219 Mer VR11', '220 Mer VR12', '221 Mer VR13', '222 Mer POT', '223 Mer RS1', '224 Mer L1', '225 Mer TE1', '226 Mer RS2', '227 Mer L2', '228 Mer TE2', '229 Mer RS3', '230 Mer L3', '231 Mer TE3', '232 Mer RC1', '233 Mer MC1', '234 Mer RC2', '235 Mer MC2', '236 Mer RC3', '237 Mer MC3', '238 Mer A0[au]', '239 Mer ECC0', '240 Mer P0[d]', '241 Mer A1[au]', '242 Mer ECC1', '243 Mer P1[d]', '244 Mer KW1', '245 Mer KW2', '246 Mer KW3', '247 Mer KWC', '248 Mer NAM1', '249 Mer NAM2', '250 Mer NAM3', '251 Mer NAMC']
        merger_data = {col.split(' ', 1)[1]: np.array(step_group[col]) 
                      for col in merger_cols if col in step_group}
        
        return {
            'ttot': time_value,
            'scalars': scalar_data,
            'singles': single_data,
            'binaries': binary_data,
            'mergers': merger_data
        }

def dataframes_from_hdf5_file(hdf5_file_path):
    """构建三个数据集：单星、双星和merger的时间序列"""
    with h5py.File(hdf5_file_path, 'r') as f:
        step_keys = sorted([k for k in f.keys() if k.startswith('Step#')])
    
    # 为三类天体分别创建时间序列数据集
    singles_dataframes = []
    binaries_dataframes = []
    mergers_dataframes = []
    scalar_data = []

    # 有时因为模拟重复跑导致snapshot重复。出现时保留前面一个snapshot
    presented_ttots = []
    
    for step_key in step_keys:
        data = load_snapshot_data(hdf5_file_path, step_key)
        if data['ttot'] in presented_ttots:
            continue
        else:
            presented_ttots.append(data['ttot'])
        
        # 记录标量数据
        scalar_data.append({**{'TTOT': data['ttot']}, **data['scalars']})
        
        # 单星数据处理
        if data['singles']:
            df_single = pd.DataFrame(data['singles'])
            df_single['TTOT'] = data['ttot']
            singles_dataframes.append(df_single)
        
        # 双星数据处理
        if data['binaries']:
            df_binary = pd.DataFrame(data['binaries'])
            df_binary['TTOT'] = data['ttot']
            binaries_dataframes.append(df_binary)
        
        # merger数据处理
        if data['mergers']:
            df_merger = pd.DataFrame(data['mergers'])
            df_merger['TTOT'] = data['ttot']
            mergers_dataframes.append(df_merger)
    
    df_scalar = pd.DataFrame(scalar_data).set_index('TTOT', drop=False)
    df_scalar.attrs['data_source'] = hdf5_file_path
    
    # 合并
    if singles_dataframes:
        df_singles = pd.concat(singles_dataframes)
        df_singles.attrs['data_source'] = hdf5_file_path
    else:
        df_singles = None
        
    if binaries_dataframes:
        df_binaries = pd.concat(binaries_dataframes)
        df_binaries.attrs['data_source'] = hdf5_file_path
    else:
        df_binaries = None
        
    if mergers_dataframes:
        df_mergers = pd.concat(mergers_dataframes)
        df_mergers.attrs['data_source'] = hdf5_file_path
    else:
        df_mergers = None
    
    return {
        'scalars': df_scalar,
        'singles': df_singles,
        'binaries': df_binaries,
        'mergers': df_mergers
    }

def merge_multiple_hdf5_dataframes(hdf5_pandas_dataframes_dict_list):
    """
    整合多个dataframes_from_hdf5_file处理后的数据集
    
    参数:
    hdf5_pandas_dataframes_dict_list - 包含多个数据集字典的列表，每个字典包含'scalars', 'singles', 'binaries', 'mergers'
    
    返回:
    合并后的数据集字典
    """
    
    # 初始化结果容器
    merged_datasets = {
        'scalars': None,
        'singles': None,
        'binaries': None,
        'mergers': None
    }
    
    # 整合标量数据
    scalar_datasets = [ds['scalars'] for ds in hdf5_pandas_dataframes_dict_list if ds['scalars'] is not None]
    if scalar_datasets:
        merged_datasets['scalars'] = pd.concat(scalar_datasets)
    
    # 整合单星数据（Dask DataFrame格式）
    singles_dfs = [ds['singles'] for ds in hdf5_pandas_dataframes_dict_list if ds['singles'] is not None]
    if singles_dfs:
        merged_datasets['singles'] = pd.concat(singles_dfs)
    
    # 整合双星数据
    binaries_dfs = [ds['binaries'] for ds in hdf5_pandas_dataframes_dict_list if ds['binaries'] is not None]
    if binaries_dfs:
        merged_datasets['binaries'] = pd.concat(binaries_dfs)
    
    # 整合merger数据
    mergers_dfs = [ds['mergers'] for ds in hdf5_pandas_dataframes_dict_list if ds['mergers'] is not None]
    if mergers_dfs:
        merged_datasets['mergers'] = pd.concat(mergers_dfs)
    
    return merged_datasets

class BlackbodyColorConverter:
    """将黑体辐射温度转换为RGB颜色的工具类"""
    
    def __init__(self, cache_path='/p/home/jusers/wu13/juwels/project/intermediate_data/teff_to_rgb.pkl'):
        self.cache_path = cache_path
        self.prepare_rgb_interpolator()
    
    def get_blackbody_rgb_df(self):
        """计算不同温度黑体辐射的RGB值"""
        # 定义温度范围
        temperatures = np.concatenate([
            np.arange(0, 20000, 10),
            np.arange(20000, 1000000, 1000)])
        # 波长范围（可见光谱：380-780 nm）
        wavelengths = np.arange(380, 781, 5)
        # 创建存储结果的列表
        results = []
        # 对每个温度计算RGB值
        for temp in tqdm(temperatures):
            # 计算黑体辐射谱
            spd_data = {}
            for wavelength in wavelengths:
                spd_data[wavelength] = planck_law(wavelength * 1e-9, temp)
            # 创建光谱分布对象
            spd = SpectralDistribution(spd_data)
            # 正规化光谱，使其具有合适的亮度
            spd.normalise()
            # 将光谱转换为XYZ三刺激值
            XYZ = msds_to_XYZ(spd, method='integration')
            # 将XYZ转换为sRGB
            RGB_linear = XYZ_to_sRGB(XYZ / 100)
            # 将线性RGB值限制在[0,1]范围内
            RGB_linear = np.clip(RGB_linear, 0, 1)
            # 应用sRGB伽马校正
            RGB = colour.cctf_encoding(RGB_linear)
            # 将结果添加到列表中
            results.append([temp] + RGB.tolist())
        return pd.DataFrame(results, columns=['Teff', 'R', 'G', 'B'])
    
    def prepare_rgb_interpolator(self):
        """
        创建从温度到RGB的插值函数
        
        参数:
            df: 包含温度和对应RGB值的DataFrame
        
        返回:
            插值函数，接收温度值返回RGB值
        """
        if os.path.exists(self.cache_path):
            self.r_interp, self.g_interp, self.b_interp = read(self.cache_path)
        else:
            df = self.get_blackbody_rgb_df()
            df_sorted = df.sort_values('Teff')
            
            # 创建三个插值函数，分别用于R、G、B
            r_interp = interp1d(df_sorted['Teff'], df_sorted['R'], kind='cubic', 
                            fill_value='extrapolate')
            g_interp = interp1d(df_sorted['Teff'], df_sorted['G'], kind='cubic', 
                            fill_value='extrapolate')
            b_interp = interp1d(df_sorted['Teff'], df_sorted['B'], kind='cubic', 
                            fill_value='extrapolate')
            save(self.cache_path, [r_interp, g_interp, b_interp])
            self.r_interp, self.g_interp, self.b_interp = r_interp, g_interp, b_interp
    
    def get_rgb(self, teff):
        """
        获取指定温度对应的RGB颜色
        
        参数:
            teff: 温度值或温度数组
        
        返回:
            RGB值，形状为 (n, 3) 或 (3,)
        """
        r = np.clip(self.r_interp(teff), 0, 1)
        g = np.clip(self.g_interp(teff), 0, 1)
        b = np.clip(self.b_interp(teff), 0, 1)
        return np.array([r, g, b]).T
    
    def plot_colorbar(self, teff_min=10, teff_max=12000, step=100):
        """
        可视化温度到RGB颜色的映射
        
        参数:
            teff_min: 最小温度
            teff_max: 最大温度
            step: 温度步长
        """
        teff = np.arange(teff_min, teff_max, step)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(teff, np.ones(len(teff)), 
                    c=self.get_rgb(teff), 
                    s=100, marker='s')
        ax.set_xlabel('Teff (K)')
        ax.set_yticks([])  # 隐藏y轴刻度
        ax.set_title('Black body temperature - RGB Color Mapping')
        ax.grid(True, axis='x')
        return fig, ax

def decode_bytes_columns_inplace(df):
    """
    解码DataFrame中指定列的字节数据为字符串，并去除空格
    
    参数:
        df: 要处理的DataFrame
        columns: 要解码的列名列表
    
    返回:
        解码后的DataFrame
    """
    _col_decoded = []
    for col in df.columns:
        if isinstance(df[col].iloc[0], bytes):
            df[col] = df[col].str.decode('utf-8').str.strip()
            _col_decoded.append(col)
    print('Decoded columns:', _col_decoded)


def tau_gw(a: float | Quantity, e: float | Quantity, mu: float | Quantity, M: float | Quantity, G=None, c=None) -> float | Quantity:
    """
    Sobolenko, Berczik, Spurzem 2021 eqn (3)
    https://doi.org/10.1051/0004-6361/202039859
    计算双黑洞系统因引力波辐射而合并的时间尺度 τ_gw（秒）。

    参数:
        a  (float): 轨道半长轴，单位为米。
        e  (float): 离心率（0 <= e < 1）。
        mu (float): 约化质量 μ，单位为千克。
        M  (float): 总质量 M，单位为千克。
        也可以全使用astropy.Quantity

    返回:
        float: 合并时标 τ_gw，单位为秒。
    """
    if G is None:
        if isinstance(a, float):
            G = constants.G.value 
        elif isinstance(a, Quantity):
            G = constants.G
    if c is None:
        if isinstance(a, float):
            c = constants.c.value
        elif isinstance(a, Quantity):
            c = constants.c

    # 公式分子和分母
    num = 5.0 * c**5 * a**4
    den = 64.0 * G**3 * mu * M**2

    # 离心率修正因子
    F_e = (1 - e**2)**3.5 / (1 + 73.0*e**2/24.0 + 37.0*e**4/96.0)
    # print(f"tau_gw: a={a}, e={e}, mu={mu}, M={M}, G={G}, c={c}, num={num}, den={den}, F_e={F_e}")

    return (num / den) * F_e

def load_GWTC_catalog(csvpath: str='/p/project1/madnuc/wu13/intermediate_data/GWTC_catalog.csv', reload=False) -> pd.DataFrame:
    '''csv下载地址：https://gwosc.org/eventapi/html/GWTC/'''
    if not reload and os.path.exists(os.path.splitext(csvpath)[0] + '.pkl'):
        return pd.read_pickle(os.path.splitext(csvpath)[0] + '.pkl')

    gwtc_df = pd.read_csv(csvpath)

    gwtc_df['mass_ratio'] = gwtc_df['mass_2_source'] / gwtc_df['mass_1_source']

    m1_abs_min = gwtc_df['mass_1_source'] + gwtc_df['mass_1_source_lower'] # lower 是负值
    m1_abs_max = gwtc_df['mass_1_source'] + gwtc_df['mass_1_source_upper']
    m2_abs_min = gwtc_df['mass_2_source'] + gwtc_df['mass_2_source_lower'] 
    m2_abs_max = gwtc_df['mass_2_source'] + gwtc_df['mass_2_source_upper']

    # mass_ratio 的最小和最大可能值
    # q_min = m2_min / m1_max
    # q_max = m2_max / m1_min
    mass_ratio_val_min = m2_abs_min / m1_abs_max
    mass_ratio_val_max = m2_abs_max / m1_abs_min

    # mass_ratio 的误差（作为与中心值的偏差）
    gwtc_df['mass_ratio_lower'] = mass_ratio_val_min - gwtc_df['mass_ratio']
    gwtc_df['mass_ratio_upper'] = mass_ratio_val_max - gwtc_df['mass_ratio']

    gwtc_df.to_pickle(os.path.splitext(csvpath)[0] + '.pkl')

    return gwtc_df

@lru_cache
def get_valueStr_of_namelist_key(path: str, key: str) -> str:
    """
    从namelist输入文件中提取初始参数。带缓存
    
    参数:
        path (str): namelist输入文件的路径。
        namelist_key (str): 要提取的namelist键。
        
    返回:
        value (str)
    """
    # 方法：在文件全局搜key=，返回value
    with open(path, 'r') as f:
        content = f.read()
    
    # 使用正则表达式搜索 key=value 模式
    # 允许key前后有空格，value可能包含数字、小数点、科学记数法等
    pattern = rf'\s*{re.escape(key)}\s*=\s*([^,\s\n]+)'
    match = re.search(pattern, content, re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
    else:
        raise KeyError(f"Key '{key}' not found in {path}")

def read_bwdat(filename):
    return pd.read_csv(filename, skiprows=(0,), sep=r'\s+')

def read_coll_13(path):
    return pd.read_csv(path, sep=r'\s+', skiprows=(0, 1, 2, 3, 5))

def read_coal_24(path):
    return pd.read_csv(path, sep=r'\s+', skiprows=4)

def make_l7header():
    # length: 284
    baselist = ['1.00E-03', '3.00E-03', '5.00E-03', '1.00E-02', '3.00E-02', '5.00E-02',
    '1.00E-01', '2.00E-01', '3.00E-01', '4.00E-01', '5.00E-01', '6.00E-01',
    '7.00E-01', '8.00E-01', '9.00E-01', '9.50E-01', '9.90E-01', '1.00E+00',
    '<RC']
    baselist2 = ['1.00E-03', '3.00E-03', '5.00E-03', '1.00E-02', '3.00E-02', '5.00E-02',
    '1.00E-01', '2.00E-01', '3.00E-01', '4.00E-01', '5.00E-01', '6.00E-01',
    '7.00E-01', '8.00E-01', '9.00E-01', '9.50E-01', '9.90E-01', '1.00E+00',]
    eachnames = ['rlagr', 'rlagr_s', 'rlagr_b', 'avmass', 'nshell', 'vx', 'vy', 'vz', 'v', 'vr', 'vt', 'sigma2', 'sigma_r2', 'sigma_t2', 'vrot']
    l7header = ['Time[NB]'] + [eachnames[0] + n2 for n2 in baselist] + [n1 + n2 for n1 in eachnames[1:3] for n2 in baselist2] + [n1 + n2 for n1 in eachnames[3:] for n2 in baselist]
    return l7header

def read_lagr_7(n6resultdir='.', fname="lagr.7.txt") -> pd.DataFrame:
    path = n6resultdir + f"/{fname}" if os.path.isdir(n6resultdir) else n6resultdir # 第一个参数可直接指定路径
    if not os.path.exists(path):
        raise IOError(f"{path} not found")
    l7header = make_l7header()
    l7file_ncol = int(get_output("tail " + path + " | awk '{print NF; exit}'")[0])
    if len(l7header) != l7file_ncol:
        raise IOError(f"{path} has ncolumn={l7file_ncol} != {len(l7header)=}. lagr.7 file in source code may have been changed.")
        
    return pd.read_csv(path, sep=r'\s+', names=l7header, skiprows=(0, 1))

def l7df_to_physical_units(df, scale_dict):
    """
    Convert the lagr.7 DataFrame to physical units using the provided scaling factors.
    scale_dict: {'r': rscale, 'm': mscale, 'v': vscale, 't': tscale}
    Time[NB]舍弃，换为 Time[Myr]
    其他的，根据colname name开头的字符串，乘以对应的scale_dict
    """
    converted_df = df.copy()
    converted_df['Time[Myr]'] = converted_df['Time[NB]'] * scale_dict['t']
    converted_df = converted_df.drop(columns=['Time[NB]'])

    metric_prefix_categories = {
           'rlagr': scale_dict['r'],
         'rlagr_s': scale_dict['r'],
         'rlagr_b': scale_dict['r'],
          'avmass': scale_dict['m'],
          'nshell': 1,
              'vx': scale_dict['v'],
              'vy': scale_dict['v'],
              'vz': scale_dict['v'],
               'v': scale_dict['v'],
              'vr': scale_dict['v'],
              'vt': scale_dict['v'],
          'sigma2': scale_dict['v']**2,
        'sigma_r2': scale_dict['v']**2,
        'sigma_t2': scale_dict['v']**2,
            'vrot': scale_dict['v'],
    }
    for col in converted_df.columns:
        if col.startswith(tuple(metric_prefix_categories.keys())):
            prefix = col.split('.')[0][:-1].split('<')[0]
            if prefix in metric_prefix_categories:
                converted_df[col] = converted_df[col].astype(float) * metric_prefix_categories[prefix]
            else:
                raise ValueError(f"Unknown prefix '{prefix}' in column '{col}'.")

    return converted_df

def transform_l7df_to_sns_friendly(df_physical_units):
    """
    Transform the lagr.7 DataFrame to a long format suitable for seaborn plotting.
    This function assumes the DataFrame has been converted to physical units using l7df_to_physical_units.
    The output DataFrame will have the following columns:
    - Time[Myr]
    - Percentage: 1%, 3%, ..., 99%, <RC
    - Metric: rlagr, rlagr_s, rlagr_b, avmass, nshell, vx, vy, vz, vrot, vr, vt, sigma_r2, sigma_t2, sigma2, v
    - Value
    """
    if not 'Time[Myr]' in df_physical_units.columns:
        raise ValueError(
            "Input DataFrame must be converted to physicsl units and contain 'Time[Myr]' column.")
    df = df_physical_units
    # 使用melt将宽格式转换为长格式，保留Time作为标识符
    melted_df = pd.melt(df, id_vars=['Time[Myr]'], var_name='variable', value_name='Value')

    # 从变量名中提取指标名称和百分比
    eachnames = ['rlagr_s', 'rlagr_b', 'rlagr', 'avmass', 'nshell', 
                    'vx', 'vy', 'vz', 'vrot', 'vr', 'vt', 
                    'sigma_r2', 'sigma_t2', 'sigma2', 'v']

    # 函数用于提取指标名称和百分比
    def extract_metric_and_percentage(variable):
        for name in eachnames:
            if variable.startswith(name):
                return name, variable[len(name):]
        return None, None

    # 应用提取函数
    metrics_and_percentages = melted_df['variable'].apply(extract_metric_and_percentage)
    melted_df['Metric'] = metrics_and_percentages.apply(lambda x: x[0])
    melted_df['Percentage'] = metrics_and_percentages.apply(lambda x: x[1])

    # 删除原始的变量列
    melted_df = melted_df.drop('variable', axis=1)

    # 重新排序列
    melted_df = melted_df[['Time[Myr]', 'Percentage', 'Metric', 'Value']]

    def percentage_present(x):
        """
        Convert a float to a percentage string with one decimal place.
        If the value is '<RC', return it as is.
        """
        if x == '<RC':
            return x
        elif can_convert_to_float(x):
            if float(x) < 0.01:
                return f"{float(x):.1%}"
            else:
                return f"{float(x):.0%}"
        else:
            raise ValueError(f"Cannot convert {x} to float for percentage calculation. Is there additional columns in the original l7df?")
        
    melted_df['%'] = melted_df['Percentage'].apply(percentage_present)
    
    return melted_df

def log_time(logger):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.debug(f"Function {func.__name__} started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.debug(f"Function {func.__name__} finished at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}, took {end_time - start_time:.4f} seconds")
            return result
        return wrapper
    return decorator