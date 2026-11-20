import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from configs.model import ModelConfig
    from experiments.pretrain_fda import adversarial_step
    from models.discriminator import ImageDiscriminator
    from models.frequency import SpectralTransfer, fda_transfer
    from utils.checkpoints import load_checkpoint, save_checkpoint
    from utils.fda_checkpoint import load_spectral_transfer, save_spectral_checkpoint


@unittest.skipIf(torch is None, "PyTorch is required for spectral-transfer training tests")
class SpectralTransferTests(unittest.TestCase):
    def test_transfer_is_differentiable_and_zero_weight_preserves_source(self):
        torch.manual_seed(42)
        source = torch.rand(2, 3, 32, 32) * 0.2 + 0.1
        style = torch.rand(2, 3, 32, 32) * 0.2 + 0.7
        unchanged = fda_transfer(source, style, source.new_tensor(0))
        self.assertTrue(torch.allclose(unchanged, source, atol=1e-5))
        transfer = SpectralTransfer()
        transfer(source, style).mean().backward()
        gradient = transfer.low_frequency_logit.grad
        self.assertTrue(torch.isfinite(gradient))
        self.assertGreater(abs(gradient.item()), 0)

    def test_adversarial_loss_updates_spectral_parameter(self):
        torch.manual_seed(42)
        transfer = SpectralTransfer()
        discriminator = ImageDiscriminator(width=4)
        # A nonuniform intensity pair gives the generator an observable style gradient.
        source = torch.rand(2, 3, 32, 32) * 0.2 + 0.1
        style = torch.rand(2, 3, 32, 32) * 0.2 + 0.7
        original = transfer.low_frequency_logit.detach().clone()
        discriminator_before = next(discriminator.parameters()).detach().clone()
        generator_loss, discriminator_loss = adversarial_step(
            transfer, discriminator, source, style,
            torch.optim.SGD(transfer.parameters(), lr=1), torch.optim.SGD(discriminator.parameters(), lr=0.1))
        self.assertFalse(torch.equal(original, transfer.low_frequency_logit.detach()))
        self.assertFalse(torch.equal(discriminator_before, next(discriminator.parameters()).detach()))
        self.assertGreater(generator_loss, 0)
        self.assertGreater(discriminator_loss, 0)
        self.assertTrue(all(parameter.requires_grad for parameter in discriminator.parameters()))

    def test_standalone_and_embedded_fda_checkpoints_restore_frozen_weights(self):
        transfer = SpectralTransfer(beta=0.2)
        transfer.pretraining = {"protocol": "da", "source_domains": ["DRIVE"], "style_domains": ["STARE"]}
        with torch.no_grad():
            transfer.low_frequency_logit.fill_(0.8)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = (root / "fda.pt", root / "stage2.pt")
            save_spectral_checkpoint(paths[0], transfer, seed=42)
            model = torch.nn.Conv2d(1, 1, 1)
            save_checkpoint(str(paths[1]), model, 2, ModelConfig(), 1, fda=transfer)
            self.assertIn("fda", load_checkpoint(str(paths[1]), 2, ModelConfig()))
            for path in paths:
                restored = load_spectral_transfer(path, "cpu")
                self.assertEqual(restored.beta, 0.2)
                self.assertTrue(torch.equal(restored.low_frequency_logit, transfer.low_frequency_logit))
                self.assertFalse(restored.low_frequency_logit.requires_grad)
                self.assertEqual(restored.pretraining["protocol"], "da")


if __name__ == "__main__":
    unittest.main()
