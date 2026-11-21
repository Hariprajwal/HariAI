"""Tests for the automatic local inference engine setup module."""
from unittest.mock import patch, MagicMock
from hari_ai.runtime.setup_engine import (
    is_engine_installed,
    get_recommended_wheel_index,
    install_inference_engine,
    CPU_WHEEL_INDEX,
)
from hari_ai.cli.main import cmd_setup


def test_is_engine_installed():
    # llama_cpp is installed in the test environment
    assert isinstance(is_engine_installed(), bool)


def test_get_recommended_wheel_index_cpu():
    url = get_recommended_wheel_index(force_cpu=True)
    assert url == CPU_WHEEL_INDEX


def test_install_inference_engine_already_installed():
    lines = []
    success = install_inference_engine(emit=lines.append)
    assert success is True
    assert any("already installed" in line.lower() for line in lines)


def test_cmd_setup_runs():
    lines = []
    cmd_setup(emit=lines.append)
    assert any("already installed" in line.lower() for line in lines)


def test_install_inference_engine_simulated_install():
    # Test installation logic path when engine is not installed
    with patch("hari_ai.runtime.setup_engine.is_engine_installed", side_effect=[False, True]):
        with patch("hari_ai.runtime.setup_engine._pip_install_engine", return_value=True):
            lines = []
            res = install_inference_engine(emit=lines.append)
            assert res is True
            assert any("setting up" in line.lower() for line in lines)
            # Success message is "✓ Engine ready!"
            assert any("engine ready" in line.lower() or "ready" in line.lower() for line in lines)

