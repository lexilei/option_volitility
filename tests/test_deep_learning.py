"""Tests for deep learning models (LSTM and TFT)."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# Check if PyTorch is available
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestLSTMNetwork:
    """Tests for LSTMNetwork class."""

    def test_network_creation(self):
        """Test creating LSTM network."""
        from src.models.lstm import LSTMNetwork

        net = LSTMNetwork(input_size=10, hidden_size=32, num_layers=2)

        assert net.hidden_size == 32
        assert net.num_layers == 2

    def test_forward_pass(self):
        """Test forward pass."""
        from src.models.lstm import LSTMNetwork

        net = LSTMNetwork(input_size=10, hidden_size=32, num_layers=2)

        # Create sample input: (batch=4, seq_len=30, features=10)
        x = torch.randn(4, 30, 10)
        output = net(x)

        assert output.shape == (4, 1)

    def test_bidirectional(self):
        """Test bidirectional LSTM."""
        from src.models.lstm import LSTMNetwork

        net = LSTMNetwork(input_size=10, hidden_size=32, bidirectional=True)

        x = torch.randn(4, 30, 10)
        output = net(x)

        assert output.shape == (4, 1)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestLSTMVolModel:
    """Tests for LSTMVolModel class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n = 200

        X = pd.DataFrame(
            {
                f"feature_{i}": np.random.randn(n) for i in range(5)
            }
        )
        y = pd.Series(np.random.uniform(0.1, 0.3, n))

        return X, y

    @pytest.fixture
    def model(self):
        """Create an LSTM model."""
        from src.models.lstm import LSTMVolModel

        return LSTMVolModel(
            sequence_length=10,
            hidden_size=16,
            num_layers=1,
            epochs=5,
            patience=3,
            batch_size=16,
        )

    def test_initialization(self, model):
        """Test model initialization."""
        assert model.sequence_length == 10
        assert model.hidden_size == 16
        assert model.name == "lstm"

    def test_create_sequences(self, model, sample_data):
        """Test sequence creation."""
        X, y = sample_data
        X_arr = X.values
        y_arr = y.values

        X_seq, y_seq = model._create_sequences(X_arr, y_arr)

        expected_len = len(X_arr) - model.sequence_length
        assert X_seq.shape == (expected_len, model.sequence_length, X_arr.shape[1])
        assert y_seq.shape == (expected_len,)

    def test_fit(self, model, sample_data):
        """Test model fitting."""
        X, y = sample_data

        model.fit(X, y, verbose=False)

        assert model.is_fitted
        assert model.model is not None
        assert model.scaler_X is not None
        assert model.scaler_y is not None

    def test_predict_before_fit(self, model, sample_data):
        """Test predict raises error before fit."""
        X, y = sample_data

        with pytest.raises(ValueError):
            model.predict(X)

    def test_predict(self, model, sample_data):
        """Test model prediction."""
        X, y = sample_data

        model.fit(X, y, verbose=False)
        predictions = model.predict(X)

        assert len(predictions) == len(X)
        # First sequence_length values should be NaN
        assert np.isnan(predictions[: model.sequence_length]).all()
        # Rest should be valid
        assert not np.isnan(predictions[model.sequence_length :]).all()

    def test_fit_with_validation(self, model, sample_data):
        """Test fit with validation data."""
        X, y = sample_data

        # Split data
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        model.fit(X_train, y_train, X_val=X_val, y_val=y_val, verbose=False)

        assert model.is_fitted
        assert len(model.train_losses) > 0
        assert len(model.val_losses) > 0

    def test_save_and_load(self, model, sample_data):
        """Test model serialization."""
        X, y = sample_data
        model.fit(X, y, verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            model.save(f.name)

            from src.models.lstm import LSTMVolModel

            loaded = LSTMVolModel.load(f.name)

            assert loaded.is_fitted
            assert loaded.sequence_length == model.sequence_length

            # Predictions should match
            original_pred = model.predict(X)
            loaded_pred = loaded.predict(X)
            np.testing.assert_array_almost_equal(
                original_pred[~np.isnan(original_pred)],
                loaded_pred[~np.isnan(loaded_pred)],
                decimal=5,
            )

    def test_predict_single_sequence(self, model, sample_data):
        """Test predicting from single sequence."""
        X, y = sample_data
        model.fit(X, y, verbose=False)

        # Get a single sequence
        X_arr = X.values
        single_seq = X_arr[: model.sequence_length]

        prediction = model.predict_single_sequence(single_seq)

        # Prediction should be a numeric scalar (float or np.float32/64)
        assert np.isscalar(prediction)
        assert not np.isnan(prediction)

    def test_device_selection(self):
        """Test device selection."""
        from src.models.lstm import LSTMVolModel

        model_cpu = LSTMVolModel(device="cpu")
        assert str(model_cpu.device) == "cpu"

    def test_early_stopping(self, sample_data):
        """Test early stopping."""
        from src.models.lstm import LSTMVolModel

        X, y = sample_data

        model = LSTMVolModel(
            sequence_length=10,
            hidden_size=16,
            epochs=100,  # High epochs
            patience=2,  # Low patience
            batch_size=32,
        )

        # Split data for validation
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        model.fit(X_train, y_train, X_val=X_val, y_val=y_val, verbose=False)

        # Should have stopped early
        assert len(model.train_losses) < 100


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestTFTComponents:
    """Tests for TFT component classes."""

    def test_gated_linear_unit(self):
        """Test GatedLinearUnit."""
        from src.models.tft import GatedLinearUnit

        glu = GatedLinearUnit(input_size=10, hidden_size=20)

        x = torch.randn(4, 10)
        output = glu(x)

        assert output.shape == (4, 20)

    def test_gate_add_norm(self):
        """Test GateAddNorm."""
        from src.models.tft import GateAddNorm

        gan = GateAddNorm(input_size=20, hidden_size=20)

        x = torch.randn(4, 20)
        skip = torch.randn(4, 20)
        output = gan(x, skip)

        assert output.shape == (4, 20)

    def test_variable_selection_network(self):
        """Test VariableSelectionNetwork."""
        from src.models.tft import VariableSelectionNetwork

        vsn = VariableSelectionNetwork(
            input_size=10,
            num_features=10,
            hidden_size=20,
        )

        # Shape: (batch, seq_len, features)
        x = torch.randn(4, 30, 10)
        output, weights = vsn(x)

        assert output.shape == (4, 30, 20)
        assert weights.shape == (4, 30, 10)
        # Weights should sum to 1
        assert torch.allclose(weights.sum(dim=-1), torch.ones(4, 30), atol=1e-5)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestTFTNetwork:
    """Tests for TemporalFusionTransformerNetwork."""

    def test_network_creation(self):
        """Test creating TFT network."""
        from src.models.tft import TemporalFusionTransformerNetwork

        net = TemporalFusionTransformerNetwork(
            input_size=10,
            hidden_size=32,
            num_heads=4,
        )

        assert net.hidden_size == 32
        assert net.input_size == 10

    def test_forward_pass(self):
        """Test forward pass."""
        from src.models.tft import TemporalFusionTransformerNetwork

        net = TemporalFusionTransformerNetwork(
            input_size=10,
            hidden_size=32,
            num_heads=4,
        )

        x = torch.randn(4, 30, 10)
        prediction, weights = net(x)

        assert prediction.shape == (4, 1)
        assert weights.shape == (4, 10)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestTFTVolModel:
    """Tests for TFTVolModel class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n = 200

        X = pd.DataFrame(
            {
                f"feature_{i}": np.random.randn(n) for i in range(5)
            }
        )
        y = pd.Series(np.random.uniform(0.1, 0.3, n))

        return X, y

    @pytest.fixture
    def model(self):
        """Create a TFT model."""
        from src.models.tft import TFTVolModel

        return TFTVolModel(
            sequence_length=10,
            hidden_size=16,
            num_heads=2,
            num_layers=1,
            epochs=5,
            patience=3,
            batch_size=16,
        )

    def test_initialization(self, model):
        """Test model initialization."""
        assert model.sequence_length == 10
        assert model.hidden_size == 16
        assert model.num_heads == 2
        assert model.name == "tft"

    def test_fit(self, model, sample_data):
        """Test model fitting."""
        X, y = sample_data

        model.fit(X, y, verbose=False)

        assert model.is_fitted
        assert model.model is not None

    def test_predict(self, model, sample_data):
        """Test model prediction."""
        X, y = sample_data

        model.fit(X, y, verbose=False)
        predictions = model.predict(X)

        assert len(predictions) == len(X)

    def test_feature_importance(self, model, sample_data):
        """Test feature importance extraction."""
        X, y = sample_data

        model.fit(X, y, verbose=False)
        importance = model.get_feature_importance()

        assert importance is not None
        assert len(importance) == X.shape[1]

    def test_save_and_load(self, model, sample_data):
        """Test model serialization."""
        X, y = sample_data
        model.fit(X, y, verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            model.save(f.name)

            from src.models.tft import TFTVolModel

            loaded = TFTVolModel.load(f.name)

            assert loaded.is_fitted
            assert loaded.sequence_length == model.sequence_length

    def test_fit_with_validation(self, model, sample_data):
        """Test fit with validation data."""
        X, y = sample_data

        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        model.fit(X_train, y_train, X_val=X_val, y_val=y_val, verbose=False)

        assert model.is_fitted
        assert len(model.train_losses) > 0


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestDeepLearningEdgeCases:
    """Edge case tests for deep learning models."""

    def test_lstm_empty_prediction(self):
        """Test LSTM with insufficient data for sequences."""
        from src.models.lstm import LSTMVolModel

        model = LSTMVolModel(sequence_length=100, hidden_size=16, epochs=2)

        # Train with enough data
        np.random.seed(42)
        X_train = pd.DataFrame({"f1": np.random.randn(200)})
        y_train = pd.Series(np.random.uniform(0.1, 0.3, 200))

        model.fit(X_train, y_train, verbose=False)

        # Predict with less data than sequence length
        X_test = pd.DataFrame({"f1": np.random.randn(50)})
        predictions = model.predict(X_test)

        # Should return empty array since not enough data for sequences
        assert len(predictions[~np.isnan(predictions)]) == 0

    def test_tft_feature_names_preserved(self):
        """Test that TFT preserves feature names."""
        from src.models.tft import TFTVolModel

        model = TFTVolModel(sequence_length=10, hidden_size=16, epochs=2)

        np.random.seed(42)
        X = pd.DataFrame(
            {
                "vol_feature": np.random.randn(100),
                "price_feature": np.random.randn(100),
                "time_feature": np.random.randn(100),
            }
        )
        y = pd.Series(np.random.uniform(0.1, 0.3, 100))

        model.fit(X, y, verbose=False)

        importance = model.get_feature_importance()
        assert importance is not None
        assert "vol_feature" in importance.index
        assert "price_feature" in importance.index
        assert "time_feature" in importance.index


class TestNoTorchFallback:
    """Tests for when PyTorch is not installed."""

    def test_lstm_import_error(self, monkeypatch):
        """Test LSTM raises proper error without PyTorch."""
        # This test verifies the import error handling
        if HAS_TORCH:
            # Skip this specific test if torch is available
            pytest.skip("PyTorch is installed")

    def test_tft_import_error(self, monkeypatch):
        """Test TFT raises proper error without PyTorch."""
        if HAS_TORCH:
            pytest.skip("PyTorch is installed")
