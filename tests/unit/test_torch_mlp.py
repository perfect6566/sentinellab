from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest


class TorchMLPTests(unittest.TestCase):
    def test_fit_reinitializes_deterministically_and_returns_probabilities(self) -> None:
        code = textwrap.dedent(
            """
            import numpy as np
            from ml.torch_mlp import TorchMLPClassifier
            rng = np.random.default_rng(7)
            features = rng.normal(size=(80, 6)).astype(np.float32)
            labels = (features[:, 0] + features[:, 1] > 0).astype(np.int8)
            classifier = TorchMLPClassifier(
                hidden_dims=(8,), max_epochs=2, batch_size=16, dropout=0.0, random_state=99
            )
            classifier.fit(features, labels)
            first = classifier.predict_proba(features)
            classifier.fit(features, labels)
            second = classifier.predict_proba(features)
            np.testing.assert_allclose(first, second, atol=1e-7)
            np.testing.assert_allclose(first.sum(axis=1), 1.0, atol=1e-7)
            assert first.shape == (80, 2)
            """
        )
        completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
