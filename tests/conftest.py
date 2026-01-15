"""
Shared pytest fixtures for dragon3_pipelines tests
"""
import pytest
import tempfile
import os
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary for testing"""
    return {
        "paths": {
            "simulations": {
                "test_sim": "/path/to/test/simulation"
            }
        },
        "plot_dir": "/path/to/plots",
        "processes_count": 4
    }
