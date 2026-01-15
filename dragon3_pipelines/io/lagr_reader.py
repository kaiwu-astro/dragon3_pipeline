"""Lagr file reading and processing"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

from dragon3_pipelines.utils import get_output

logger = logging.getLogger(__name__)


class ContinousFileProcessor:
    def __init__(self, config_manager, file_basename: str):
        self.config = config_manager
        self.file_basename = file_basename
        self.file_path = None
        self.firstjobof: Dict[str, str] = {}
        self.scale_dict_of: Dict[str, Dict[str, float]] = {}

    def concat_file(self, simu_name: str) -> None:
        gather_file_cmd = f'cd {self.config.pathof[simu_name]};' + \
        f'''tmpf=`mktemp --suffix=.{self.file_basename}`; find . -name '{self.file_basename}*' | xargs ls | xargs cat > $tmpf; echo $tmpf'''
        self.file_path = get_output(gather_file_cmd)[0]
        logger.debug(f'Gathered {self.file_basename} of {simu_name} files into {self.file_path}')
    
    def read_file(self, simu_name: str):
        self.concat_file(simu_name)
        logger.debug(f'Loading gathered self.file_basename at {self.file_path}')
        raise NotImplementedError("子类必须实现此方法")
    
    def clean_data(self, df: pd.DataFrame, timecol: str = 'TIME[NB]') -> pd.DataFrame:
        """
        可能因为模拟重跑而造成某个star反复输出
        在类似[1.0, 2.1, 3.2, 4.3, 5.7, 3.5, 4.6, 5.9, 4.7, 4.8, 7.1]的数据里去掉 3.5， 4.6， 4.7，4.8
        """
        is_forwarding = np.array([df[timecol][:i+1].max() == v for i, v in df[timecol].items()])
        if not is_forwarding.all():
            logger.warning(f"[{self.file_basename}] Warning: Found {len(is_forwarding) - is_forwarding.sum()} descending entries in {timecol}, removing")
        return df[is_forwarding].reset_index(drop=True)

    def firstjobhere(self, simu_name: str) -> str:
        '''同shell命令，返回jobid。自带缓存机制'''
        if simu_name not in self.firstjobof.keys():
            get_firstj_cmd = f'cd {self.config.pathof[simu_name]};' + \
            r'''ls | grep -E '^[0-9]+$' | sort -n | head -n 1'''
            self.firstjobof[simu_name] = get_output(get_firstj_cmd)[-1]
        return self.firstjobof[simu_name]
    
    firstj = firstjobhere
    
    def get_scale_dict_from_stdout(self, simu_name: str) -> Dict[str, float]:
        """
        从stdout中提取缩放字典。自带缓存机制
        """
        from glob import glob
        from dragon3_pipelines.io.text_parsers import get_scale_dict
        
        if simu_name not in self.scale_dict_of:
            first_output_file_path = glob(self.config.pathof[simu_name] + '/' + self.firstj(simu_name) + '/N*out')[0]
            self.scale_dict_of[simu_name] = get_scale_dict(first_output_file_path)
            print(f'Got {self.scale_dict_of[simu_name]} from {first_output_file_path}')
        return self.scale_dict_of[simu_name]


class LagrFileProcessor(ContinousFileProcessor):
    """读取和画图前预处理lagr.7"""
    def __init__(self, config_manager):
        super().__init__(config_manager, file_basename='lagr.7')
    
    def read_file(self, simu_name: str) -> pd.DataFrame:
        from dragon3_pipelines.io.text_parsers import read_lagr_7, l7df_to_physical_units
        
        self.concat_file(simu_name)
        logger.debug(f'Loading gathered {self.file_basename} of {simu_name} at {self.file_path}')
        l7df = read_lagr_7(self.file_path)
        l7df = self.clean_data(l7df)
        l7df = l7df_to_physical_units(l7df, self.get_scale_dict_from_stdout(simu_name))
        return l7df
    
    def clean_data(self, l7df: pd.DataFrame) -> pd.DataFrame:
        """
        1) 丢弃包含非数值型数据的行（应全为 int/float）
        2) 处理 'Time[NB]' 的重复：保留最后一次出现（避免中途中断导致的不完整行）。
        """
        numeric_df = l7df.apply(pd.to_numeric, errors='coerce')
        non_numeric_mask = (numeric_df.isna() & l7df.notna()).any(axis=1)
        if non_numeric_mask.any():
            if 'Time[NB]' in l7df.columns:
                bad_times = np.unique(l7df.loc[non_numeric_mask, 'Time[NB]'].values)
                logger.warning(f"[lagr.7] Warning: Found non-numeric entries; dropping {non_numeric_mask.sum()} rows at Time[NB]={bad_times}")
            else:
                logger.warning(f"[lagr.7] Warning: Found non-numeric entries; dropping {non_numeric_mask.sum()} rows (no 'Time[NB]' column)")
        l7df = numeric_df.loc[~non_numeric_mask].copy()

        if 'Time[NB]' in l7df.columns:
            duplicated_times = l7df['Time[NB]'].duplicated(keep=False)
            if duplicated_times.any():
                dup_vals = np.unique(l7df.loc[duplicated_times, 'Time[NB]'].values)
                logger.warning(f"[lagr.7] Warning: Duplicate 'Time[NB]' detected at {dup_vals}; using the last occurrence")
                l7df = l7df.loc[l7df['Time[NB]'].duplicated(keep='last') | ~duplicated_times]
        else:
            logger.warning("[lagr.7] Warning: 'Time[NB]' column not found when de-duplicating.")
        return l7df
    
    def load_sns_friendly_data(self, simu_name: str) -> pd.DataFrame:
        from dragon3_pipelines.io.text_parsers import transform_l7df_to_sns_friendly
        
        l7df_sns = transform_l7df_to_sns_friendly(self.read_file(simu_name))
        metrics_to_transform = ['sigma2', 'sigma_r2', 'sigma_t2']
        new_rows = []
        for metric_old in metrics_to_transform:
            df_subset = l7df_sns[l7df_sns['Metric'] == metric_old].copy()
            if not df_subset.empty:
                df_subset['Value'] = np.sqrt(df_subset['Value'])
                metric_new = metric_old[:-1]
                df_subset['Metric'] = metric_new
                new_rows.append(df_subset)
        if new_rows:
            l7df_sns = pd.concat([l7df_sns, ] + new_rows, ignore_index=True)

        return l7df_sns
