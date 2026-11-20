import unittest

try:
    import torch
    from torch import nn
    from models.network import ConditionalDiffusionSegmenter
    from configs.model import ModelConfig
    from models.diffusion import GaussianDiffusion
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch and SAM packages are required")
class ModelShapeTests(unittest.TestCase):
    def test_two_branches_and_noise_shape(self):
        class FakeEncoder(nn.Module):
            img_size = 256

            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(nn.Conv2d(3, 256, 3, stride=4, padding=1), nn.GELU())

            def forward(self, image):
                return self.net(image)

        config = ModelConfig(task="odoc")
        model = ConditionalDiffusionSegmenter(config, FakeEncoder(), FakeEncoder())
        self.assertFalse(any(p.requires_grad for p in model.segmentation_encoder.parameters()))
        image = torch.rand(1, 3, 256, 256)
        mask = torch.rand(1, 2, 256, 256)
        t = torch.tensor([50])
        features = model.encode_condition(image)
        self.assertEqual([tuple(feature.shape[-2:]) for feature in features], [(64, 64), (32, 32), (16, 16)])
        self.assertEqual(tuple(model(mask, t, features).shape), tuple(mask.shape))
        loss = GaussianDiffusion(config).training_loss(model, mask, image)
        self.assertEqual(loss.ndim, 0)


if __name__ == "__main__":
    unittest.main()
