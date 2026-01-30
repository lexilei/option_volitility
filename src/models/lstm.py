"""LSTM model for volatility prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from .base import BaseVolModel

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class LSTMNetwork(nn.Module):
    """LSTM neural network for time series prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        """Initialize the LSTM network.

        Args:
            input_size: Number of input features
            hidden_size: Size of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            Output tensor of shape (batch, 1)
        """
        # LSTM output
        lstm_out, _ = self.lstm(x)

        # Take the last timestep output
        out = lstm_out[:, -1, :]

        # Fully connected layers
        out = self.fc(out)
        return out


class LSTMVolModel(BaseVolModel):
    """LSTM model for volatility prediction."""

    name = "lstm"

    def __init__(
        self,
        sequence_length: int = 30,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 10,
        device: str | None = None,
        **kwargs: Any,
    ):
        """Initialize the LSTM model.

        Args:
            sequence_length: Length of input sequences
            hidden_size: Size of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
            learning_rate: Learning rate for optimizer
            batch_size: Training batch size
            epochs: Maximum training epochs
            patience: Early stopping patience
            device: Device to use ('cuda', 'cpu', or None for auto)
            **kwargs: Additional parameters
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is not installed. Run: pip install torch")

        super().__init__(
            sequence_length=sequence_length,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            **kwargs,
        )

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience

        # Auto-select device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: LSTMNetwork | None = None
        self.scaler_X: Any = None
        self.scaler_y: Any = None
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def _create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Create sequences for LSTM input.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values (optional)

        Returns:
            Tuple of (X_sequences, y_sequences) or (X_sequences, None)
        """
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
    ) -> "LSTMVolModel":
        """Fit the LSTM model.

        Args:
            X: Feature matrix
            y: Target values
            X_val: Validation features
            y_val: Validation targets
            verbose: Whether to print progress
            **kwargs: Additional fitting arguments

        Returns:
            Self
        """
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
        self.model = LSTMNetwork(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
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

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                output = self.model(batch_X).squeeze()
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)

            # Validation
            val_loss = train_loss
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        output = self.model(batch_X).squeeze()
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
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.is_fitted = True
        logger.info(f"Fitted {self.name} with {self.num_layers} layers, hidden_size={self.hidden_size}")
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Predicted values
        """
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
            predictions = self.model(X_tensor).cpu().numpy()

        # Inverse scale
        predictions = self.scaler_y.inverse_transform(predictions).flatten()

        # Pad the beginning with NaN (due to sequence creation)
        full_predictions = np.full(len(X_arr), np.nan)
        full_predictions[self.sequence_length :] = predictions

        return full_predictions

    def predict_single_sequence(self, X: np.ndarray) -> float:
        """Predict from a single sequence.

        Args:
            X: Sequence of shape (sequence_length, n_features)

        Returns:
            Predicted value
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.scaler_X.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            prediction = self.model(X_tensor).cpu().numpy()

        return self.scaler_y.inverse_transform(prediction).flatten()[0]

    def save(self, path: str) -> None:
        """Save the model to disk.

        Args:
            path: Path to save the model
        """
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
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

        # Save PyTorch model state separately
        if self.model is not None:
            model_data["model_state"] = self.model.state_dict()
            model_data["input_size"] = self.model.lstm.input_size

        joblib.dump(model_data, path)
        logger.info(f"Saved model to {path}")

    @classmethod
    def load(cls, path: str) -> "LSTMVolModel":
        """Load a model from disk.

        Args:
            path: Path to the saved model

        Returns:
            Loaded model instance
        """
        import joblib

        model_data = joblib.load(path)

        # Reconstruct model
        instance = cls(**model_data["params"])
        instance.is_fitted = model_data["is_fitted"]
        instance.feature_names = model_data["feature_names"]
        instance.scaler_X = model_data["scaler_X"]
        instance.scaler_y = model_data["scaler_y"]
        instance.train_losses = model_data.get("train_losses", [])
        instance.val_losses = model_data.get("val_losses", [])

        if "model_state" in model_data:
            instance.model = LSTMNetwork(
                input_size=model_data["input_size"],
                hidden_size=instance.hidden_size,
                num_layers=instance.num_layers,
                dropout=instance.dropout,
                bidirectional=instance.bidirectional,
            ).to(instance.device)
            instance.model.load_state_dict(model_data["model_state"])

        logger.info(f"Loaded model from {path}")
        return instance
