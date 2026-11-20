import unittest

try:
    from infer import load_stage2_for_original_inference
except ImportError:
    load_stage2_for_original_inference = None


@unittest.skipIf(load_stage2_for_original_inference is None,
                 "PyTorch and model dependencies are required")
class OriginalEncoderLoadingTests(unittest.TestCase):
    class FakeModel:
        def __init__(self):
            self.loaded = None

        def state_dict(self):
            return {
                "segmentation_encoder.weight": None,
                "condition_encoder.weight": None,
                "decoder.weight": None,
            }

        def load_state_dict(self, state, strict=False):
            self.loaded = state
            expected = set(self.state_dict())
            return sorted(expected - set(state)), sorted(set(state) - expected)

    def test_stage2_weights_leave_original_encoder_untouched(self):
        model = self.FakeModel()
        checkpoint = {
            "segmentation_encoder.weight": "stage1-transfer",
            "condition_encoder.weight": "stage2-condition",
            "decoder.weight": "stage2-decoder",
        }

        load_stage2_for_original_inference(model, checkpoint)

        self.assertNotIn("segmentation_encoder.weight", model.loaded)
        self.assertEqual(model.loaded["condition_encoder.weight"], "stage2-condition")
        self.assertEqual(model.loaded["decoder.weight"], "stage2-decoder")


if __name__ == "__main__":
    unittest.main()
