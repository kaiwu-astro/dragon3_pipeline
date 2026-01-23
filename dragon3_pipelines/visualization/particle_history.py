"""Particle history visualization tools for tracking individual particle evolution"""

import logging
import os
from typing import Any, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from dragon3_pipelines.analysis.physics import (
    compute_binary_orbit_relative_positions,
    compute_individual_orbit_params,
)
from dragon3_pipelines.utils import log_time
from dragon3_pipelines.visualization.base import BaseVisualizer, add_grid

logger = logging.getLogger(__name__)


class ParticleHistoryVisualizer(BaseVisualizer):
    """Visualizer for particle evolution history

    This class provides visualization tools for tracking individual particles
    through their evolution in the simulation, including both single star and
    binary states.
    """

    def __init__(
        self,
        config_manager: Any,
        simu_name: Optional[str] = None,
        particle_name: Optional[int] = None,
    ) -> None:
        """
        Initialize the ParticleHistoryVisualizer.

        Args:
            config_manager: Configuration manager instance
            simu_name: Optional simulation name for output path construction
            particle_name: Optional particle name/ID for output path construction
        """
        super().__init__(config_manager)
        self.simu_name = simu_name
        self.particle_name = particle_name

    def _filter_by_time(
        self,
        history_df: pd.DataFrame,
        time: Union[str, float, Tuple[float, float]],
    ) -> pd.DataFrame:
        """
        Filter history DataFrame by time specification.

        Args:
            history_df: DataFrame containing particle history with 'TTOT' column
            time: Time filter specification:
                - 'all': Return all rows
                - float: Return row with TTOT closest to this value
                - (t_start, t_end): Return rows where t_start <= TTOT <= t_end

        Returns:
            Filtered DataFrame
        """
        if history_df.empty or "TTOT" not in history_df.columns:
            return history_df

        if time == "all":
            return history_df

        if isinstance(time, (int, float)):
            # Find row with TTOT closest to the specified value
            idx = (history_df["TTOT"] - time).abs().idxmin()
            return history_df.loc[[idx]]

        if isinstance(time, (tuple, list)) and len(time) == 2:
            t_start, t_end = time
            mask = (history_df["TTOT"] >= t_start) & (history_df["TTOT"] <= t_end)
            return history_df[mask]

        logger.warning(f"Invalid time specification: {time}. Returning all data.")
        return history_df

    def _get_marker_color(self, kw: int, teff: float) -> Union[str, Tuple[float, float, float]]:
        """
        Get marker color based on stellar type and effective temperature.

        Args:
            kw: Stellar type code (KW)
            teff: Effective temperature in Kelvin

        Returns:
            Color specification (string or RGB tuple)
        """
        if kw == 13:  # Neutron Star
            return "red"
        if kw == 14:  # Black Hole
            return "black"
        # Use blackbody color for other stellar types
        try:
            rgb = self.teff_to_rgb_converter.get_rgb(teff)
            if hasattr(rgb, "shape") and len(rgb.shape) > 1:
                rgb = rgb[0]  # Handle array case
            return tuple(rgb)
        except Exception:
            return "gray"

    def _plot_main_axes(self, ax: plt.Axes, row: pd.Series) -> None:
        """
        Plot the main axes showing stellar positions and orbits.

        Args:
            ax: Matplotlib axes object
            row: Single row from history DataFrame
        """
        # Set up axis limits and scale
        bin_a_limits = self.config.limits.get("Bin A[au]", (0.001, 100000.0))
        ax.set_xlim(-bin_a_limits[1], bin_a_limits[1])
        ax.set_ylim(-bin_a_limits[1], bin_a_limits[1])
        ax.set_xscale("symlog", linthresh=bin_a_limits[0])
        ax.set_yscale("symlog", linthresh=bin_a_limits[0])
        ax.set_xlabel("X [au]")
        ax.set_ylabel("Y [au]")

        # Determine if this is a single star or binary state
        state = row.get("state", "single")
        is_binary = state == "binary" and pd.notna(row.get("Bin cm X1", np.nan))

        if not is_binary:
            # Single star: plot at origin
            self._plot_single_star(ax, row)
        else:
            # Binary: plot both stars with orbits
            self._plot_binary_system(ax, row)

        add_grid(ax)

    def _plot_single_star(self, ax: plt.Axes, row: pd.Series) -> None:
        """
        Plot a single star at the origin.

        Args:
            ax: Matplotlib axes object
            row: Single row from history DataFrame
        """
        # Get stellar properties
        r_star = row.get("R*", 1.0)
        teff = row.get("Teff*", 5778.0)
        kw = int(row.get("KW", 1))

        # Calculate marker size based on stellar radius
        base_size = 100
        marker_size = base_size * np.log10(r_star + 1) if r_star > 0 else base_size

        # Get color
        color = self._get_marker_color(kw, teff)

        # Plot the star at origin
        ax.scatter([0], [0], s=marker_size, c=[color], edgecolors="white", linewidths=0.5, zorder=5)

        # Annotate with stellar type
        stellar_type = self.config.kw_to_stellar_type.get(kw, str(kw))
        ax.annotate(
            stellar_type,
            (0, 0),
            ha="center",
            va="center",
            fontsize=8,
            color="white" if kw == 14 else "black",
            zorder=6,
        )

    def _plot_binary_system(self, ax: plt.Axes, row: pd.Series) -> None:
        """
        Plot a binary system with both stars and their orbits.

        Args:
            ax: Matplotlib axes object
            row: Single row from history DataFrame
        """
        # Get binary properties
        m1 = row.get("Bin M1*", 1.0)
        m2 = row.get("Bin M2*", 1.0)
        a_bin = row.get("Bin A[au]", 1.0)
        ecc_bin = row.get("Bin ECC", 0.0)

        # Get relative position (if available, otherwise estimate from semi-major axis)
        rel_x = row.get("Bin rel X1", a_bin)  # Use semi-major axis as approximation
        rel_y = row.get("Bin rel X2", 0.0)
        rel_z = row.get("Bin rel X3", 0.0)

        # Compute positions relative to center of mass
        (x1, y1, z1), (x2, y2, z2) = compute_binary_orbit_relative_positions(
            m1, m2, rel_x, rel_y, rel_z
        )

        # Get stellar properties for each component
        kw1 = int(row.get("Bin KW1", 1))
        kw2 = int(row.get("Bin KW2", 1))
        teff1 = row.get("Bin Teff1*", 5778.0)
        teff2 = row.get("Bin Teff2*", 5778.0)
        r1 = row.get("Bin R1*", 1.0)
        r2 = row.get("Bin R2*", 1.0)

        # Calculate marker sizes
        base_size = 100
        size1 = base_size * np.log10(r1 + 1) if r1 > 0 else base_size
        size2 = base_size * np.log10(r2 + 1) if r2 > 0 else base_size

        # Get colors
        color1 = self._get_marker_color(kw1, teff1)
        color2 = self._get_marker_color(kw2, teff2)

        # Plot both stars
        ax.scatter([x1], [y1], s=size1, c=[color1], edgecolors="white", linewidths=0.5, zorder=5)
        ax.scatter([x2], [y2], s=size2, c=[color2], edgecolors="white", linewidths=0.5, zorder=5)

        # Annotate with stellar types
        st1 = self.config.kw_to_stellar_type.get(kw1, str(kw1))
        st2 = self.config.kw_to_stellar_type.get(kw2, str(kw2))
        ax.annotate(
            st1,
            (x1, y1),
            ha="center",
            va="center",
            fontsize=8,
            color="white" if kw1 == 14 else "black",
            zorder=6,
        )
        ax.annotate(
            st2,
            (x2, y2),
            ha="center",
            va="center",
            fontsize=8,
            color="white" if kw2 == 14 else "black",
            zorder=6,
        )

        # Compute individual orbit parameters
        (a1, e1), (a2, e2) = compute_individual_orbit_params(a_bin, ecc_bin, m1, m2)

        # Draw orbital ellipses (centered at origin = center of mass)
        # For each star, semi-major axis is a_i, semi-minor axis is a_i * sqrt(1 - e^2)
        b1 = a1 * np.sqrt(1 - e1**2) if e1 < 1 else a1
        b2 = a2 * np.sqrt(1 - e2**2) if e2 < 1 else a2

        # Draw orbital ellipses
        ellipse1 = Ellipse(
            (0, 0),
            width=2 * a1,
            height=2 * b1,
            angle=0,
            fill=False,
            linestyle="--",
            edgecolor=color1 if isinstance(color1, str) else "gray",
            linewidth=1,
            alpha=0.6,
        )
        ellipse2 = Ellipse(
            (0, 0),
            width=2 * a2,
            height=2 * b2,
            angle=0,
            fill=False,
            linestyle="--",
            edgecolor=color2 if isinstance(color2, str) else "gray",
            linewidth=1,
            alpha=0.6,
        )
        ax.add_patch(ellipse1)
        ax.add_patch(ellipse2)

    def _annotate_info(self, ax: plt.Axes, row: pd.Series) -> None:
        """
        Add text annotations with stellar/binary information.

        Args:
            ax: Matplotlib axes object
            row: Single row from history DataFrame
        """

        def _fmt_val(val, fmt_spec: str, suffix: str = "") -> str:
            """Safely format a value, returning 'N/A' for NaN."""
            if pd.isna(val):
                return f"{'N/A':>{len(fmt_spec.split('.')[0])}}{suffix}"
            return f"{val:{fmt_spec}}{suffix}"

        state = row.get("state", "single")
        is_binary = state == "binary" and pd.notna(row.get("Bin cm X1", np.nan))

        info_lines = []

        if not is_binary:
            # Single star information
            m = row.get("M", np.nan)
            r = row.get("R*", np.nan)
            teff = row.get("Teff*", np.nan)
            kw = int(row.get("KW", 0)) if pd.notna(row.get("KW")) else 0

            color = self._get_marker_color(kw, teff if pd.notna(teff) else 5778.0)

            info_lines.append(f"M*     = {_fmt_val(m, '8.3f', ' Msun')}")
            info_lines.append(f"R*     = {_fmt_val(r, '8.3f', ' Rsun')}")
            info_lines.append(f"Teff*  = {_fmt_val(teff, '8.0f', ' K')}")
            info_lines.append(f"KW     = {kw:8d}")
        else:
            # Binary star information
            m1 = row.get("Bin M1*", np.nan)
            m2 = row.get("Bin M2*", np.nan)
            teff1 = row.get("Bin Teff1*", np.nan)
            teff2 = row.get("Bin Teff2*", np.nan)
            a_bin = row.get("Bin A[au]", np.nan)
            ecc = row.get("Bin ECC", np.nan)
            peri = a_bin * (1 - ecc) if pd.notna(a_bin) and pd.notna(ecc) else np.nan
            ebind = row.get("Ebind/kT", np.nan)
            tau_gw = row.get("tau_gw[Myr]", np.nan)

            info_lines.append(f"M1     = {_fmt_val(m1, '8.3f', ' Msun')}")
            info_lines.append(f"M2     = {_fmt_val(m2, '8.3f', ' Msun')}")
            info_lines.append(f"Teff1  = {_fmt_val(teff1, '8.0f', ' K')}")
            info_lines.append(f"Teff2  = {_fmt_val(teff2, '8.0f', ' K')}")
            info_lines.append("")
            info_lines.append(f"a      = {_fmt_val(a_bin, '8.3f', ' au')}")
            info_lines.append(f"e      = {_fmt_val(ecc, '8.4f', '')}")
            info_lines.append(f"peri   = {_fmt_val(peri, '8.3f', ' au')}")
            info_lines.append(f"Eb/kT  = {_fmt_val(ebind, '8.2f', '')}")
            info_lines.append(f"tau_gw = {_fmt_val(tau_gw, '8.1f', ' Myr')}")

            color = "black"

        # Add text annotation in upper left corner
        info_text = "\n".join(info_lines)
        ax.text(
            0.02,
            0.98,
            info_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontfamily="monospace",
            fontsize=9,
            color=color if isinstance(color, str) else "black",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Set figure title
        ttot = row.get("TTOT", 0.0)
        tmyr = row.get("Time[Myr]", 0.0)
        particle_name = row.get("Name", self.particle_name or "Unknown")
        simu_name = self.simu_name or ""

        title = f"{simu_name} | Particle {particle_name}\nTTOT = {ttot:.3f} NB = {tmyr:.2f} Myr"
        ax.figure.suptitle(title, fontsize=12)

    def _plot_inset_axes(self, ax: plt.Axes, row: pd.Series) -> None:
        """
        Add an inset axes showing the particle's position in the cluster.

        Args:
            ax: Main matplotlib axes object
            row: Single row from history DataFrame
        """
        # Create inset axes
        ax_inset = inset_axes(ax, width="33%", height="33%", loc="lower left", borderpad=0.2)
        ax_inset.patch.set_alpha(0.8)

        # Get position
        state = row.get("state", "single")
        is_binary = state == "binary" and pd.notna(row.get("Bin cm X1", np.nan))

        if is_binary:
            x_pc = row.get("Bin cm X [pc]", row.get("Bin cm X1", 0.0))
            y_pc = row.get("Bin cm Y [pc]", row.get("Bin cm X2", 0.0))
        else:
            x_pc = row.get("X [pc]", row.get("X1", 0.0))
            y_pc = row.get("Y [pc]", row.get("X2", 0.0))

        # Get color based on stellar type
        if is_binary:
            kw = int(row.get("Bin KW1", 1))
            teff = row.get("Bin Teff1*", 5778.0)
        else:
            kw = int(row.get("KW", 1))
            teff = row.get("Teff*", 5778.0)

        color = self._get_marker_color(kw, teff)

        # Plot position
        ax_inset.scatter([x_pc], [y_pc], s=30, c=[color], edgecolors="white", linewidths=0.5)
        ax_inset.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
        ax_inset.axvline(0, color="gray", linewidth=0.5, alpha=0.5)

        # Set limits
        pos_limits = self.config.limits.get("position_pc_lim", (-15.0, 15.0))
        ax_inset.set_xlim(*pos_limits)
        ax_inset.set_ylim(*pos_limits)

        # Configure tick parameters - place labels inside
        ax_inset.tick_params(axis="both", direction="in", pad=-12, labelsize=7, length=3)
        ax_inset.set_xlabel("X [pc]", fontsize=7, labelpad=-20)
        ax_inset.set_ylabel("Y [pc]", fontsize=7, labelpad=-25)

    @log_time(logger)
    def plot(
        self,
        history_df: pd.DataFrame,
        time: Union[str, float, Tuple[float, float]] = "all",
        simu_name: Optional[str] = None,
        particle_name: Optional[int] = None,
    ) -> None:
        """
        Generate visualization plots for particle history.

        Creates a figure for each time point showing the particle's state,
        including position, stellar properties, and orbital information (if in binary).

        Args:
            history_df: DataFrame containing particle evolution history
            time: Time filter specification:
                - 'all': Generate plots for all time points
                - float: Generate plot for the time point closest to this value
                - (t_start, t_end): Generate plots for time points in this range
            simu_name: Optional simulation name (overrides instance value)
            particle_name: Optional particle name (overrides instance value)
        """
        if history_df.empty:
            logger.warning("Empty history DataFrame provided")
            return

        # Use provided values or fall back to instance values
        simu_name = simu_name or self.simu_name
        particle_name = particle_name or self.particle_name

        if particle_name is None:
            # Try to extract from DataFrame
            particle_name = (
                int(history_df["Name"].iloc[0])
                if "Name" in history_df.columns and not history_df["Name"].isna().all()
                else "unknown"
            )

        # Filter by time
        filtered_df = self._filter_by_time(history_df, time)

        if filtered_df.empty:
            logger.warning(f"No data after filtering by time={time}")
            return

        # Create output directory
        plot_base_dir = self.config.plot_dir
        output_dir = os.path.join(plot_base_dir, "particle_history", str(particle_name))
        os.makedirs(output_dir, exist_ok=True)

        # Generate plot for each row
        for idx, row in filtered_df.iterrows():
            ttot = row.get("TTOT", -1.0)
            output_path = os.path.join(output_dir, f"{particle_name}_{ttot:.2f}.jpg")

            if self.config.skip_existing_plot and os.path.exists(output_path):
                logger.debug(f"Skipping existing plot: {output_path}")
                continue

            # Create figure
            fig, ax = plt.subplots(figsize=(8, 8))

            # Plot components
            self._plot_main_axes(ax, row)
            self._annotate_info(ax, row)
            self._plot_inset_axes(ax, row)

            # Save figure
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.debug(f"Saved plot: {output_path}")

            # Close figure to free memory
            try:
                __IPYTHON__
                if self.config.close_figure_in_ipython:
                    plt.close(fig)
            except NameError:
                plt.close(fig)

        logger.info(
            f"Generated {len(filtered_df)} plots for particle {particle_name} " f"in {output_dir}"
        )
