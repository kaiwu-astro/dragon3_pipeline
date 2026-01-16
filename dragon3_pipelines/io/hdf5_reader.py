"""HDF5 file reading and processing"""

import logging
import warnings
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import astropy.constants as constants
import astropy.units as u

from dragon3_pipelines.utils import log_time

logger = logging.getLogger(__name__)


class HDF5FileProcessor:
    """Read and preprocess HDF5 data for plotting"""
    def __init__(self, config_manager):
        self.config = config_manager
    
    @log_time(logger)
    def read_file(self, hdf5_path: str, simu_name: Optional[str] = None, N0: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Load and preprocess HDF5 data. Extract multiple DataFrames from a single HDF5 file containing snapshots at multiple times.
        
        Note: One HDF5 file (.h5part) contains MULTIPLE snapshots (typically 8) at different time points.
        Each snapshot represents the simulation state at one specific TTOT value.
        
        Args:
            hdf5_path: Path to HDF5 file
            simu_name: Used to get initial condition file path and read N0
            N0: Initial particle count (required if simu_name is None)
        
        Returns:
            df_dict: Dictionary containing 'scalars', 'singles', 'binaries', 'mergers'
                     Each DataFrame contains data for ALL snapshots in this HDF5 file
        """
        from dragon3_pipelines.io.text_parsers import get_valueStr_of_namelist_key, get_scale_dict_from_hdf5_df, dataframes_from_hdf5_file
        from dragon3_pipelines.io.text_parsers import tau_gw
        
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
        single_df_all['mod_velocity[kmps]'] = np.sqrt(
            single_df_all['V1']**2 + single_df_all['V2']**2 + single_df_all['V3']**2
        )
        ## 虽然星团质心已知有偏移，但不知质心速度如何，这里验算一下质心速度。如果质心速度的mod超过0.1km/s，就警告
        vcm_x = (single_df_all['V1'] * single_df_all['M']).sum() / single_df_all['M'].sum()
        vcm_y = (single_df_all['V2'] * single_df_all['M']).sum() / single_df_all['M'].sum()
        vcm_z = (single_df_all['V3'] * single_df_all['M']).sum() / single_df_all['M'].sum()
        vcm_mod = np.sqrt(vcm_x**2 + vcm_y**2 + vcm_z**2)
        TOO_HIGH_CLUSTER_CM_VELOCITY_THRESHOLD_KMPS = 1.0
        if vcm_mod > TOO_HIGH_CLUSTER_CM_VELOCITY_THRESHOLD_KMPS:
            logger.warning(f"[{hdf5_path}] Warning: Cluster center of mass velocity = ({vcm_x:.3f}, {vcm_y:.3f}, {vcm_z:.3f}) km/s, mod={vcm_mod:.3f} km/s, seems high")
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
        
        pc_to_AU = constants.pc.to(u.AU).value
        binary_df_all['Ebind_abs_NBODY'] = \
             binary_df_all['Bin M1*'] / scale_dict['m']         \
                * binary_df_all['Bin M2*'] / scale_dict['m']     \
            / (2 * binary_df_all['Bin A[au]'] / pc_to_AU / scale_dict['r']) \
            / (binary_df_all['Bin M1*'] / scale_dict['m'] + binary_df_all['Bin M2*'] / scale_dict['m'])
        binary_df_all['Ebind/kT'] = \
            binary_df_all['Ebind_abs_NBODY'] / self.config.ECLOSE_INPUT
        binary_df_all['is_hard_binary'] = binary_df_all['Ebind/kT'] >= 1
        
        with warnings.catch_warnings():
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

        return df_dict
    
    def get_hdf5_file_time_from_filename(self, hdf5_path: str) -> float:
        """Extract approximate snapshot time from HDF5 filename
        
        Note: This extracts the time encoded in the filename, which represents
        the approximate time of snapshots contained in this HDF5 file.
        Each HDF5 file contains multiple snapshots at different times.
        """
        return float(hdf5_path.split('snap.40_')[-1].split('.h5part')[0])
    
    @log_time(logger)
    def get_snapshot_at_t(self, df_dict: Dict[str, pd.DataFrame], ttot: float) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], bool]:
        """
        Get snapshot data at a specific time point
        
        Note: This extracts data for ONE snapshot (one moment in time) from the 
        df_dict that contains multiple snapshots from an HDF5 file.
        
        Args:
            df_dict: Dictionary containing 'scalars', 'singles', 'binaries', 'mergers'
                     (obtained from read_file, contains MULTIPLE snapshots)
            ttot: Time point to retrieve (one specific snapshot)
        
        Returns:
            single_df: Single star DataFrame at this time
            binary_df: Binary star DataFrame at this time
            is_valid: True/False, validates if lengths match scalar counts
        """
        single_df = df_dict['singles'][df_dict['singles']['TTOT'] == ttot].copy()
        binary_df = df_dict['binaries'][df_dict['binaries']['TTOT'] == ttot].copy()
        
        # 获取单星和双星的数量
        N_SINGLE = df_dict['scalars'].loc[ttot, 'N_SINGLE']
        N_BINARY = df_dict['scalars'].loc[ttot, 'N_BINARY']
        
        # 验证数据完整性
        if not isinstance(N_SINGLE, (float, np.float64, np.float32)) or not isinstance(N_BINARY, (float, np.float64, np.float32)):
            return None, None, False
        
        return single_df, binary_df, True
    
    def get_compact_object_mask(self, df: pd.DataFrame) -> pd.Series:
        """Get mask for binaries containing compact objects"""
        if 'KW' in df.columns:
            compact_object_mask = df['KW'].isin(self.config.compact_object_KW)
        elif 'Bin KW1' in df.columns and 'Bin KW2' in df.columns:
            compact_object_mask = df['Bin KW1'].isin(self.config.compact_object_KW) | df['Bin KW2'].isin(self.config.compact_object_KW)
        else:
            raise ValueError("DataFrame does not contain 'KW' or 'Bin KW1/Bin KW2' columns. Columns: " + str(df.columns))
        return compact_object_mask

    def mark_funny_star_single(self, single_df: pd.DataFrame) -> None:
        """Mark 'interesting' single star targets (writes tag_* columns and aggregates to is_funny)"""
        df = single_df
        low, high = self.config.IMBH_mass_range_msun

        mask_imbh = (df['KW'] == 14) & df['M'].between(low, high)
        if 'tag_IMBH' not in df.columns:
            df['tag_IMBH'] = False
        df.loc[mask_imbh, 'tag_IMBH'] = True

        # NEW: NS
        mask_ns = (df['KW'] == 13)
        if 'tag_NS' not in df.columns:
            df['tag_NS'] = False
        df.loc[mask_ns, 'tag_NS'] = True

        # NEW: high_velocity_halo_star: r>20pc & v>50 km/s
        mask_hv_halo = (df['Distance_to_cluster_center[pc]'] > 20) & (df['mod_velocity[kmps]'] > 50)
        if 'tag_high_velocity_halo_star' not in df.columns:
            df['tag_high_velocity_halo_star'] = False
        df.loc[mask_hv_halo, 'tag_high_velocity_halo_star'] = True

        if 'is_funny' not in df.columns:
            df['is_funny'] = False
        df.loc[mask_imbh | mask_ns | mask_hv_halo, 'is_funny'] = True

    def mark_funny_star_binary(self, binary_df: pd.DataFrame) -> None:
        """Mark 'interesting' binary star targets (writes tag_* columns and aggregates to is_funny)"""
        df = binary_df
        low, high = self.config.IMBH_mass_range_msun
        gap_low, gap_high = self.config.PISNe_mass_gap

        mask_bh1 = df['Bin KW1'] == 14
        mask_bh2 = df['Bin KW2'] == 14

        mask_imbh = (
            (mask_bh1 & df['Bin M1*'].between(low, high)) |
            (mask_bh2 & df['Bin M2*'].between(low, high))
        )
        if 'tag_IMBH' not in df.columns:
            df['tag_IMBH'] = False
        df.loc[mask_imbh, 'tag_IMBH'] = True

        mask_bh_bh = mask_bh1 & mask_bh2
        mask_emr = mask_bh_bh & (df['mass_ratio'] < self.config.extreme_mass_ratio_upper)
        if 'tag_EMR_BHBH' not in df.columns:
            df['tag_EMR_BHBH'] = False
        df.loc[mask_emr, 'tag_EMR_BHBH'] = True

        mask_high_e = mask_bh_bh & (df['Bin ECC'] > self.config.high_BH_ecc_lower)
        if 'tag_high_e_BHBH' not in df.columns:
            df['tag_high_e_BHBH'] = False
        df.loc[mask_high_e, 'tag_high_e_BHBH'] = True

        mask_pisne = (
            (mask_bh1 & df['Bin M1*'].between(gap_low, gap_high)) |
            (mask_bh2 & df['Bin M2*'].between(gap_low, gap_high))
        )
        if 'tag_PISNe_mass_gap_BH' not in df.columns:
            df['tag_PISNe_mass_gap_BH'] = False
        df.loc[mask_pisne, 'tag_PISNe_mass_gap_BH'] = True

        # NEW: BH-NS binary
        mask_ns1 = df['Bin KW1'] == 13
        mask_ns2 = df['Bin KW2'] == 13
        mask_bh_ns = (mask_bh1 & mask_ns2) | (mask_bh2 & mask_ns1)
        if 'tag_BH_NS' not in df.columns:
            df['tag_BH_NS'] = False
        df.loc[mask_bh_ns, 'tag_BH_NS'] = True

        # NEW: any-NS binary
        mask_ns_any = mask_ns1 | mask_ns2
        if 'tag_NS_any' not in df.columns:
            df['tag_NS_any'] = False
        df.loc[mask_ns_any, 'tag_NS_any'] = True

        # NEW: Bin cm KW odd/even flags (robust to missing column)
        if 'Bin cm KW' in df.columns:
            bcmkw = df['Bin cm KW']
            mask_mass_transferring = (bcmkw > 10) & ((bcmkw.astype(int) % 2) == 1)
            mask_circularized = (bcmkw >= 10) & ((bcmkw.astype(int) % 2) == 0)
        else:
            logger.debug("[mark_funny_star_binary] Column 'Bin cm KW' not found; skip mass_transferring/circularized tags.")
            mask_mass_transferring = False
            mask_circularized = False

        if 'tag_mass_transferring_binary' not in df.columns:
            df['tag_mass_transferring_binary'] = False
        df.loc[mask_mass_transferring, 'tag_mass_transferring_binary'] = True

        if 'tag_circularized_binary' not in df.columns:
            df['tag_circularized_binary'] = False
        df.loc[mask_circularized, 'tag_circularized_binary'] = True

        combined_mask = (
            mask_imbh |
            mask_emr |
            mask_high_e |
            mask_pisne |
            mask_bh_ns |
            mask_ns_any |
            mask_mass_transferring |
            mask_circularized
        )
        if 'is_funny' not in df.columns:
            df['is_funny'] = False
        df.loc[combined_mask, 'is_funny'] = True

    def _compute_binding_energy(self, m_bin: float, m3: float, r: float, 
                                v_rel_x: float, v_rel_y: float, v_rel_z: float) -> float:
        """
        Compute binding energy for a triple system
        
        Args:
            m_bin: Binary total mass [Msun]
            m3: Third body mass [Msun]
            r: Distance from binary center of mass to third body [pc]
            v_rel_x, v_rel_y, v_rel_z: Relative velocity components [km/s]
        
        Returns:
            E_bind: Binding energy [Msun * (km/s)^2], negative value indicates bound
        """
        mu = (m_bin * m3) / (m_bin + m3)
        v_rel = np.sqrt(v_rel_x**2 + v_rel_y**2 + v_rel_z**2)
        E_kin = 0.5 * mu * v_rel**2
        G_PCKMSSQ_MSUN = 4.302e-3
        E_pot = -G_PCKMSSQ_MSUN * m_bin * m3 / r
        E_bind = E_kin + E_pot
        return E_bind

    @log_time(logger)
    def get_triples_from_hdf5(self, df_dict: Dict[str, pd.DataFrame], ttot: float) -> pd.DataFrame:
        """
        Identify hierarchical triple systems
        
        Args:
            df_dict: Dictionary containing 'singles', 'binaries', 'scalars'
            ttot: Current TTOT value
        
        Returns:
            triples_df: DataFrame containing triple system information
        """
        pc_to_AU = constants.pc.to(u.AU).value
        
        single_df_at_t, binary_df_at_t, is_valid = self.get_snapshot_at_t(df_dict, ttot)
        
        if not is_valid or binary_df_at_t.empty or single_df_at_t.empty:
            logger.debug(f"No valid data for triples detection at ttot={ttot}")
            return pd.DataFrame()
        
        single_positions = single_df_at_t[['X [pc]', 'Y [pc]', 'Z [pc]']].copy().values
        single_names = single_df_at_t['Name'].values
        tree = cKDTree(single_positions)
        
        triples_list = []
        
        for idx, binary_row in binary_df_at_t.iterrows():
            bin_name1 = binary_row['Bin Name1']
            bin_name2 = binary_row['Bin Name2']
            bin_cm_name = binary_row['Bin cm Name']
            bin_pos = np.array([
                binary_row['Bin cm X [pc]'],
                binary_row['Bin cm Y [pc]'],
                binary_row['Bin cm Z [pc]']
            ])
            bin_vel = np.array([
                binary_row['Bin cm V1'],
                binary_row['Bin cm V2'],
                binary_row['Bin cm V3']
            ])
            m_bin = binary_row['total_mass[solar]']
            a_bin = binary_row['Bin A[au]']
            
            r_max_pc = a_bin / pc_to_AU * 1000
            candidate_indices = tree.query_ball_point(bin_pos, r=r_max_pc)
            
            n_triples_found_for_this_binary = 0
            for cand_idx in candidate_indices:
                third_name = single_names[cand_idx]
                
                if third_name == bin_name1 or third_name == bin_name2:
                    continue
                n_triples_found_for_this_binary += 1
                
                third_row = single_df_at_t.iloc[cand_idx]
                third_pos = single_positions[cand_idx]
                third_vel = np.array([
                    third_row['V1'],
                    third_row['V2'],
                    third_row['V3']
                ])
                m3 = third_row['M']
                
                r_vec = third_pos - bin_pos
                r = np.linalg.norm(r_vec)
                v_rel = third_vel - bin_vel
                
                E_bind = self._compute_binding_energy(
                    m_bin, m3, r,
                    v_rel[0], v_rel[1], v_rel[2]
                )
                
                if E_bind < 0:
                    triple_info = {
                        'TTOT': ttot,
                        'Bin cm Name': bin_cm_name,
                        'Bin Name1': bin_name1,
                        'Bin Name2': bin_name2,
                        'Third_body_Name': third_name,
                        'cm_distance_bin_to_3rd[pc]': r,
                        'E_bind[Msun*(km/s)^2]': E_bind,
                        'Bin total_mass[solar]': m_bin,
                        'Third_mass[solar]': m3,
                        'Bin A[au]': a_bin,
                        'Bin ECC': binary_row['Bin ECC'],
                        'Bin KW1': binary_row['Bin KW1'],
                        'Bin KW2': binary_row['Bin KW2'],
                        'Third_KW': third_row['KW'],
                    }
                    triples_list.append(triple_info)

            if n_triples_found_for_this_binary > 1:
                logger.warning(f"[Multiples] Found {n_triples_found_for_this_binary} stars with Ebind<0 for binary 1,2,cm={bin_name1},{bin_name2},{bin_cm_name} at ttot={ttot}, indicating possible higher-order multiples.")

        if triples_list:
            triples_df = pd.DataFrame(triples_list)
            logger.info(f"Found {len(triples_df)} hierarchical triples at ttot={ttot}")
            return triples_df
        else:
            logger.debug(f"No hierarchical triples found at ttot={ttot}")
            return pd.DataFrame()
