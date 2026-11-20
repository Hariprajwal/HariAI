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
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.stdout.readline.side_effect = [
                "Collecting llama-cpp-python\n",
                "Downloading prebuilt wheel\n",
                "Successfully installed llama-cpp-python\n",
                "",
            ]
            mock_proc.wait.return_value = 0
            mock_popen.return_value = mock_proc

            lines = []
            res = install_inference_engine(emit=lines.append)
            assert res is True
            assert any("setting up" in line.lower() for line in lines)
            assert any("verified successfully" in line.lower() for line in lines)
