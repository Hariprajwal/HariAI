"""Unit tests for the HariAI training infrastructure."""

import os
import tempfile
import unittest
import torch
from torch.utils.data import TensorDataset

from hari_ai.model.config import UAIANanoConfig
from hari_ai.model.transformer import UAIANanoModel
from hari_ai.trainer.optimizer import CosineWarmupScheduler, configure_optimizers
from hari_ai.trainer.trainer import TrainerConfig, UAIATrainer


class TestTrainerInfrastructure(unittest.TestCase):
    def setUp(self):
        self.micro_config = UAIANanoConfig(
            vocab_size=64,
            d_model=32,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_ff=64,
            max_seq_len=32,
            n_actions=4,
            tie_word_embeddings=True,
        )
        self.model = UAIANanoModel(self.micro_config)

    def test_optimizer_parameter_grouping(self):
        optimizer = configure_optimizers(self.model, lr=1e-3, weight_decay=0.1)
        self.assertEqual(len(optimizer.param_groups), 2)

        decay_group = optimizer.param_groups[0]
        no_decay_group = optimizer.param_groups[1]

        self.assertEqual(decay_group["weight_decay"], 0.1)
        self.assertEqual(no_decay_group["weight_decay"], 0.0)

        # Ensure all decay params are 2D+
        for p in decay_group["params"]:
            self.assertGreaterEqual(p.dim(), 2)

        # Ensure all no-decay params are 1D
        for p in no_decay_group["params"]:
            self.assertEqual(p.dim(), 1)

    def test_cosine_warmup_scheduler(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_steps=100,
            max_steps=1000,
            max_lr=1e-3,
            min_lr=1e-4,
        )

        # At step 0: linear warmup fraction
        lr_0 = scheduler.get_lr(0)
        self.assertAlmostEqual(lr_0, 1e-3 / 100, places=5)

        # At warmup_steps: should reach max_lr
        lr_warmup = scheduler.get_lr(100)
        self.assertAlmostEqual(lr_warmup, 1e-3, places=5)

        # At max_steps: should reach min_lr
        lr_max = scheduler.get_lr(1000)
        self.assertAlmostEqual(lr_max, 1e-4, places=5)

        # After max_steps: stays at min_lr
        lr_after = scheduler.get_lr(1500)
        self.assertEqual(lr_after, 1e-4)

    def test_training_loss_decrease_on_synthetic_data(self):
        # Create a small repetitive dataset to verify learning convergence
        seq_len = 8
        vocab_size = self.micro_config.vocab_size

        # Fixed token sequence repeated
        pattern = torch.tensor([[i % vocab_size for i in range(seq_len + 1)]] * 32)
        input_ids = pattern[:, :-1]
        labels = pattern[:, 1:]

        dataset = TensorDataset(input_ids, labels)

        class DictDataset(torch.utils.data.Dataset):
            def __init__(self, ds):
                self.ds = ds

            def __len__(self):
                return len(self.ds)

            def __getitem__(self, idx):
                x, y = self.ds[idx]
                return {"input_ids": x, "labels": y}

        dict_ds = DictDataset(dataset)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer_cfg = TrainerConfig(
                output_dir=tmpdir,
                batch_size=4,
                grad_accum_steps=1,
                learning_rate=3e-3,
                max_steps=30,
                warmup_steps=5,
                device="cpu",
            )

            trainer = UAIATrainer(
                model=self.model,
                config=trainer_cfg,
                train_dataset=dict_ds,
            )

            history = trainer.train()

            # Verify that training loss decreased
            initial_loss = history[0]["loss"]
            final_loss = history[-1]["loss"]
            self.assertLess(final_loss, initial_loss)

    def test_checkpoint_save_and_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer_cfg = TrainerConfig(output_dir=tmpdir, device="cpu")
            trainer = UAIATrainer(
                model=self.model,
                config=trainer_cfg,
                train_dataset=TensorDataset(torch.zeros(1)),  # dummy
            )

            ckpt_path = trainer.save_checkpoint(step=42)
            self.assertTrue(os.path.exists(ckpt_path))

            # Perturb model weights
            with torch.no_grad():
                self.model.token_embeddings.weight.fill_(99.0)

            # Reload
            trainer.load_checkpoint(ckpt_path)
            self.assertEqual(trainer.global_step, 42)
            # Weights should no longer be 99.0
            self.assertNotEqual(float(self.model.token_embeddings.weight[0, 0].item()), 99.0)


if __name__ == "__main__":
    unittest.main()
