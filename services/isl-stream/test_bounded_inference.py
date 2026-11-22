"""Optional PyTorch tests for bounded causal CTC window inference."""
import unittest

try:
    import torch
except ImportError:  # Training dependencies are intentionally optional in lightweight CI.
    torch = None

if torch is not None:
    import numpy as np
    from training.inference import iter_bounded_causal_predictions


@unittest.skipIf(torch is None, "PyTorch training dependency is not installed")
class BoundedInferenceTests(unittest.TestCase):
    class TokenEchoModel:
        causal = True

        def __call__(self, features, lengths):
            ids = features[0, :, 0].long()
            logits = torch.full((len(ids), 1, 4), -10.0, device=features.device)
            logits[torch.arange(len(ids), device=features.device), 0, ids] = 10.0
            return logits.log_softmax(dim=-1), lengths

    def test_overlap_is_not_decoded_twice_and_ctc_state_crosses_windows(self):
        ids = np.array([1, 1, 0, 2, 2, 0, 3, 3], dtype=np.float32)
        chunks = [ids[:2, None], ids[2:6, None], ids[6:, None]]
        emitted = list(iter_bounded_causal_predictions(
            chunks,
            self.TokenEchoModel(),
            torch.device("cpu"),
            {0: "", 1: "HELLO", 2: "WORLD", 3: "YES"},
            max_frames=4,
            overlap_frames=1,
        ))
        self.assertEqual(emitted, [["HELLO", "WORLD"], ["YES"], []])

    def test_requires_causal_model(self):
        model = self.TokenEchoModel()
        model.causal = False
        with self.assertRaisesRegex(ValueError, "causal checkpoint"):
            list(iter_bounded_causal_predictions(
                [np.ones((2, 1), dtype=np.float32)],
                model,
                torch.device("cpu"),
                {0: "", 1: "HELLO"},
                max_frames=4,
            ))


if __name__ == "__main__":
    unittest.main()
