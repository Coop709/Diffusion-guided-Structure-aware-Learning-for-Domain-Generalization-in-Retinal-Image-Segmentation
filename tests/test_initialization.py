import unittest

try:
    import torch
    from configs.model import ModelConfig
    from models.diffusion import GaussianDiffusion
    from models.initialization import initialize_mask
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for initialization tests")
class InitializationTests(unittest.TestCase):
    def test_joint_uses_the_same_noise_field_for_both_terms(self):
        coarse = torch.full((2, 1, 16, 16), 0.75)
        channel = torch.full_like(coarse, 0.25)
        draw = lambda mode: initialize_mask(coarse, channel, mode, torch.Generator().manual_seed(42))
        self.assertTrue(torch.allclose(draw("joint"), draw("sam_only") + draw("channel_only")))
        self.assertTrue(torch.equal(draw("gaussian"), draw("additional_condition")))

    def test_structure_initializer_uses_convolution(self):
        from models.initialization import convolve_structure

        noise = torch.zeros(1, 1, 7, 7)
        cue = torch.zeros_like(noise)
        noise[0, 0, 3, 3] = 1
        cue[0, 0, 3, 4] = 1
        result = convolve_structure(noise, cue)
        self.assertEqual(float(result[0, 0, 3, 4]), 1.0)
        self.assertEqual(int(torch.count_nonzero(result)), 1)

    def test_feature_capture_uses_the_last_denoising_invocation(self):
        class ToyDenoiser:
            cue_conditioner = None

            def encode_condition(self, image):
                return image

            def __call__(self, sample, t, condition, use_pff=True, return_features=False):
                epsilon = torch.zeros_like(sample)
                return (epsilon, t.clone()) if return_features else epsilon

        image = torch.ones(1, 3, 4, 4)
        mask = torch.ones(1, 1, 4, 4)
        probabilities, final_t = GaussianDiffusion(ModelConfig(diffusion_steps=20)).ddim_sample(
            ToyDenoiser(), image, mask, image, steps=10, return_features=True)
        self.assertEqual(final_t.item(), 0)
        self.assertEqual(tuple(probabilities.shape), tuple(mask.shape))


if __name__ == "__main__":
    unittest.main()
