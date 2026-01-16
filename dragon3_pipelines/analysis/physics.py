"""Physics calculations and formulae for astrophysical simulations"""

# Re-export tau_gw from io module where it was already migrated
from dragon3_pipelines.io.text_parsers import tau_gw

__all__ = ['tau_gw']
