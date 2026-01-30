"""Temporal Fusion Transformer model for volatility prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from .base import BaseVolModel

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit layer."""

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(torch.sigmoid(self.fc1(x)) * self.fc2(x))


class GateAddNorm(nn.Module):
    """Gate-Add-Norm block."""

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.glu = GatedLinearUnit(input_size, hidden_size, dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(self.glu(x) + skip)


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network for feature importance."""

    def __init__(
        self,
        input_size: int,
        num_features: int,
        hidden_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features

        # Feature transformations
        self.feature_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_size),
                nn.ReLU(),
            )
            for _ in range(num_features)
        ])

        # Variable selection
        self.selection_weights = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_features),
            nn.Softmax(dim=-1),
        )

        self.glu = GatedLinearUnit(hidden_size, hidden_size, dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch, seq_len, num_features)
        batch_size, seq_len, _ = x.shape

        # Transform each feature
        transformed = []
        for i, transform in enumerate(self.feature_transforms):
            feat = x[:, :, i : i + 1]  # (batch, seq_len, 1)
            transformed.append(transform(feat))  # (batch, seq_len, hidden_size)

        transformed = torch.stack(transformed, dim=-1)  # (batch, seq_len, hidden_size, num_features)

        # Calculate selection weights
        weights = self.selection_weights(x)  # (batch, seq_len, num_features)
        weights = weights.unsqueeze(2)  # (batch, seq_len, 1, num_features)

        # Weighted sum of transformed features
        selected = (transformed * weights).sum(dim=-1)  # (batch, seq_len, hidden_size)

        return self.glu(selected), weights.squeeze(2)


class TemporalFusionTransformerNetwork(nn.Module):
    """Simplified Temporal Fusion Transformer network."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_layers: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        # Variable selection network
        self.vsn = VariableSelectionNetwork(
            input_size=input_size,
            num_features=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
        )

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Gate add norm
        self.gate_add_norm1 = GateAddNorm(hidden_size, hidden_size, dropout)
        self.gate_add_norm2 = GateAddNorm(hidden_size, hidden_size, dropout)

        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Variable selection
        selected, weights = self.vsn(x)

        # LSTM encoding
        lstm_out, _ = self.lstm(selected)

        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Gate add norm
        gated = self.gate_add_norm1(attn_out, lstm_out)

        # Take last timestep
        output = gated[:, -1, :]

        # Final prediction
        prediction = self.output_layer(output)

        return prediction, weights.mean(dim=1)  # Average weights over time


class TFTVolModel(BaseVolModel):
    """Temporal Fusion Transformer model for volatility prediction."""

    name = "tft"

    def __init__(
        self,
        sequence_length: int = 30,
        hidden_size: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 10,
        device: str | None = None,
        **kwargs: Any,
    ):
        """Initialize the TFT model.

        Args:
            sequence_length: Length of input sequences
            hidden_size: Size of hidden layers
            num_heads: Number of attention heads
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate
            batch_size: Training batch size
            epochs: Maximum training epochs
            patience: Early stopping patience
            device: Device to use
            **kwargs: Additional parameters
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is not installed. Run: pip install torch")

        super().__init__(
            sequence_length=sequence_length,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            **kwargs,
        )

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: TemporalFusionTransformerNetwork | None = None
        self.scaler_X: Any = None
        self.scaler_y: Any = None
        self.feature_weights: np.ndarray | None = None
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def _create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Create sequences for TFT input."""
        sequences = []
        targets = []

        for i in range(self.sequence_length, len(X)):
            sequences.append(X[i - self.sequence_length : i])
            if y is not None:
                targets.append(y[i])

        X_seq = np.array(sequences)
        y_seq = np.array(targets) if y is not None else None

        return X_seq, y_seq

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: pd.Series | np.ndarray | None = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> "TFTVolModel":
        """Fit the TFT model."""
        from sklearn.preprocessing import StandardScaler

        X_arr = self._validate_input(X)
        y_arr = self._validate_target(y)

        # Scale features
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        X_scaled = self.scaler_X.fit_transform(X_arr)
        y_scaled = self.scaler_y.fit_transform(y_arr.reshape(-1, 1)).flatten()

        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)

        # Initialize model
        input_size = X_arr.shape[1]
        self.model = TemporalFusionTransformerNetwork(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout,
            num_layers=self.num_layers,
        ).to(self.device)

        # Create data loaders
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).to(self.device)
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Validation data
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_arr = self._validate_input(X_val)
            y_val_arr = self._validate_target(y_val)
            X_val_scaled = self.scaler_X.transform(X_val_arr)
            y_val_scaled = self.scaler_y.transform(y_val_arr.reshape(-1, 1)).flatten()
            X_val_seq, y_val_seq = self._create_sequences(X_val_scaled, y_val_scaled)

            X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val_seq).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        all_weights = []

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            epoch_weights = []

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                output, weights = self.model(batch_X)
                output = output.squeeze()
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                epoch_weights.append(weights.detach().cpu().numpy())

            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)

            # Validation
            val_loss = train_loss
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        output, _ = self.model(batch_X)
                        output = output.squeeze()
                        loss = criterion(output, batch_y)
                        val_loss += loss.item()
                val_loss /= len(val_loader)
            self.val_losses.append(val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{self.epochs}: "
                    f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = self.model.state_dict().copy()
                all_weights = epoch_weights
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Store average feature weights
        if all_weights:
            self.feature_weights = np.mean(np.concatenate(all_weights, axis=0), axis=0)

        self.is_fitted = True
        logger.info(
            f"Fitted {self.name} with hidden_size={self.hidden_size}, "
            f"num_heads={self.num_heads}"
        )
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")

        X_arr = self._validate_input(X)
        X_scaled = self.scaler_X.transform(X_arr)

        # Create sequences
        X_seq, _ = self._create_sequences(X_scaled)

        if len(X_seq) == 0:
            return np.array([])

        self.model.eval()
        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        with torch.no_grad():
            predictions, _ = self.model(X_tensor)
            predictions = predictions.cpu().numpy()

        # Inverse scale
        predictions = self.scaler_y.inverse_transform(predictions).flatten()

        # Pad the beginning with NaN
        full_predictions = np.full(len(X_arr), np.nan)
        full_predictions[self.sequence_length :] = predictions

        return full_predictions

    def get_feature_importance(self) -> pd.Series | None:
        """Get feature importance from variable selection weights."""
        if self.feature_weights is None:
            return None

        if self.feature_names:
            return pd.Series(self.feature_weights, index=self.feature_names).sort_values(
                ascending=False
            )
        return pd.Series(self.feature_weights).sort_values(ascending=False)

    def save(self, path: str) -> None:
        """Save the model to disk."""
        import joblib
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "params": self.params,
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names,
            "scaler_X": self.scaler_X,
            "scaler_y": self.scaler_y,
            "feature_weights": self.feature_weights,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

        if self.model is not None:
            model_data["model_state"] = self.model.state_dict()
            model_data["input_size"] = self.model.input_size

        joblib.dump(model_data, path)
        logger.info(f"Saved model to {path}")

    @classmethod
    def load(cls, path: str) -> "TFTVolModel":
        """Load a model from disk."""
        import joblib

        model_data = joblib.load(path)

        instance = cls(**model_data["params"])
        instance.is_fitted = model_data["is_fitted"]
        instance.feature_names = model_data["feature_names"]
        instance.scaler_X = model_data["scaler_X"]
        instance.scaler_y = model_data["scaler_y"]
        instance.feature_weights = model_data.get("feature_weights")
        instance.train_losses = model_data.get("train_losses", [])
        instance.val_losses = model_data.get("val_losses", [])

        if "model_state" in model_data:
            instance.model = TemporalFusionTransformerNetwork(
                input_size=model_data["input_size"],
                hidden_size=instance.hidden_size,
                num_heads=instance.num_heads,
                dropout=instance.dropout,
                num_layers=instance.num_layers,
            ).to(instance.device)
            instance.model.load_state_dict(model_data["model_state"])

        logger.info(f"Loaded model from {path}")
        return instance
