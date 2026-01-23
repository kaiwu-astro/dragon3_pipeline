"""Physics calculations and formulae for astrophysical simulations"""

from typing import Tuple

# Re-export tau_gw from io module where it was already migrated
from dragon3_pipelines.io.text_parsers import tau_gw

__all__ = [
    "tau_gw",
    "compute_binary_orbit_relative_positions",
    "compute_individual_orbit_params",
]


def compute_binary_orbit_relative_positions(
    m1: float, m2: float, rel_x: float, rel_y: float, rel_z: float
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Compute the positions of two stars relative to their common center of mass.

    Given the relative position vector (r_rel = r_2 - r_1), this function computes
    the positions of each star relative to the center of mass:
        r_1 = -M_2 / (M_1 + M_2) * r_rel
        r_2 = M_1 / (M_1 + M_2) * r_rel

    Args:
        m1: Mass of the primary star [any unit, same as m2]
        m2: Mass of the secondary star [any unit, same as m1]
        rel_x: X component of relative position vector (r_2 - r_1)
        rel_y: Y component of relative position vector (r_2 - r_1)
        rel_z: Z component of relative position vector (r_2 - r_1)

    Returns:
        Tuple of two tuples:
            - (x1, y1, z1): Position of primary star relative to center of mass
            - (x2, y2, z2): Position of secondary star relative to center of mass
    """
    total_mass = m1 + m2
    if total_mass == 0:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

    # Primary star position relative to center of mass
    frac1 = -m2 / total_mass
    x1 = frac1 * rel_x
    y1 = frac1 * rel_y
    z1 = frac1 * rel_z

    # Secondary star position relative to center of mass
    frac2 = m1 / total_mass
    x2 = frac2 * rel_x
    y2 = frac2 * rel_y
    z2 = frac2 * rel_z

    return (x1, y1, z1), (x2, y2, z2)


def compute_individual_orbit_params(
    a_bin: float, ecc_bin: float, m1: float, m2: float
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Compute the orbital parameters for each star around the center of mass.

    For a binary system with semi-major axis a and eccentricity e, each star
    orbits the center of mass with:
        a_1 = a * M_2 / (M_1 + M_2)
        a_2 = a * M_1 / (M_1 + M_2)
        e_1 = e_2 = e  (eccentricity is the same for both orbits)

    Args:
        a_bin: Semi-major axis of the binary [any unit]
        ecc_bin: Eccentricity of the binary orbit
        m1: Mass of the primary star [any unit, same as m2]
        m2: Mass of the secondary star [any unit, same as m1]

    Returns:
        Tuple of two tuples:
            - (a1, e1): Semi-major axis and eccentricity for primary star
            - (a2, e2): Semi-major axis and eccentricity for secondary star
    """
    total_mass = m1 + m2
    if total_mass == 0:
        return (0.0, ecc_bin), (0.0, ecc_bin)

    a1 = a_bin * m2 / total_mass
    a2 = a_bin * m1 / total_mass

    return (a1, ecc_bin), (a2, ecc_bin)
