"""Unit tests for HariAI Hardware Accelerator Detection & GPU offload."""

import unittest
from hari_ai.runtime.hardware import detect_hardware, HardwareProfile


class TestHardwareDetection(unittest.TestCase):
    def test_detect_hardware_returns_profile(self):
        profile = detect_hardware(refresh=True)
        self.assertIsInstance(profile, HardwareProfile)
        self.assertIn(profile.device_type, ["cuda", "directml", "mps", "gpu_offload", "cpu"])
        self.assertGreater(profile.cpu_cores, 0)
        self.assertGreater(profile.optimal_threads, 0)
        self.assertIsInstance(profile.summary(), str)
        self.assertGreater(len(profile.summary()), 0)

    def test_profile_summary_gpu_vs_cpu(self):
        gpu_profile = HardwareProfile(
            device_type="cuda",
            device_name="NVIDIA GeForce RTX 4090",
            gpu_available=True,
            n_gpu_layers=-1,
            total_memory_gb=24.0,
            cpu_cores=16,
            optimal_threads=8,
        )
        self.assertIn("GPU (NVIDIA GeForce RTX 4090 · 24.0 GB VRAM)", gpu_profile.summary())

        cpu_profile = HardwareProfile(
            device_type="cpu",
            device_name="x86_64 8 cores",
            gpu_available=False,
            n_gpu_layers=0,
            cpu_cores=8,
            optimal_threads=4,
        )
        self.assertIn("CPU (x86_64 8 cores · 4 threads)", cpu_profile.summary())


if __name__ == "__main__":
    unittest.main()
