import unittest

from datasets.pairing import paired_batches


class DomainPairingTests(unittest.TestCase):
    def test_style_batches_cycle_without_dropping_source_batches(self):
        self.assertEqual(list(paired_batches([1, 2, 3, 4, 5], ["a", "b"])),
                         [(1, "a"), (2, "b"), (3, "a"), (4, "b"), (5, "a")])

    def test_target_masks_are_not_needed_for_pairing(self):
        source = [{"image": "source", "mask": "source_label"}]
        target = [{"image": "target"}]
        batch, style = next(paired_batches(source, target))
        self.assertEqual(batch["mask"], "source_label")
        self.assertNotIn("mask", style)

    def test_empty_style_loader_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            list(paired_batches([1], []))


if __name__ == "__main__":
    unittest.main()
