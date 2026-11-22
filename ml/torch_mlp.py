"""A small sklearn-compatible PyTorch binary classifier."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(previous, width), nn.ReLU(), nn.Dropout(dropout)])
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(1)


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """Deterministic CPU MLP that reinitializes on every ``fit`` call.

    Early stopping is intentionally omitted. Formal CV therefore uses a fixed
    epoch budget and never consults the final hold-out or validation fold while
    fitting. Each sklearn clone/fold constructs fresh model and optimizer state.
    """

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (128, 64),
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 20,
        dropout: float = 0.1,
        random_state: int = 20261028,
        device: str = "cpu",
        verbose: bool = False,
    ) -> None:
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.dropout = dropout
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    @staticmethod
    def _dense_float32(values: Any) -> np.ndarray:
        if sparse.issparse(values):
            values = values.toarray()
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError("TorchMLPClassifier expects a 2D feature matrix")
        if not np.isfinite(array).all():
            raise ValueError("TorchMLPClassifier received non-finite features")
        return array

    def fit(self, X: Any, y: Any) -> TorchMLPClassifier:
        features = self._dense_float32(X)
        labels = np.asarray(y, dtype=np.float32)
        if set(np.unique(labels)) != {0.0, 1.0}:
            raise ValueError("TorchMLPClassifier requires both binary classes 0 and 1")

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        torch.use_deterministic_algorithms(True)
        self.classes_ = np.asarray([0, 1], dtype=np.int8)
        self.n_features_in_ = features.shape[1]
        self.model_ = _MLP(self.n_features_in_, tuple(self.hidden_dims), self.dropout).to(
            self.device
        )
        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        positives = float(labels.sum())
        negatives = float(len(labels) - positives)
        positive_weight = torch.tensor(
            [negatives / positives], dtype=torch.float32, device=self.device
        )
        loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight)

        dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
        generator = torch.Generator().manual_seed(self.random_state)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        self.training_loss_ = []
        self.model_.train()
        for epoch in range(self.max_epochs):
            total_loss = 0.0
            for batch_features, batch_labels in loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model_(batch_features)
                loss = loss_function(logits, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * len(batch_labels)
            epoch_loss = total_loss / len(dataset)
            self.training_loss_.append(epoch_loss)
            if self.verbose:
                print(f"epoch={epoch + 1} loss={epoch_loss:.6f}")
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, ["model_", "classes_", "n_features_in_"])
        features = self._dense_float32(X)
        if features.shape[1] != self.n_features_in_:
            raise ValueError("Feature count differs from fitted MLP")
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.from_numpy(features).to(self.device))
            malware = torch.sigmoid(logits).cpu().numpy()
        return np.column_stack([1.0 - malware, malware])

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int8)
