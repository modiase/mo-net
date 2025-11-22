"""Tests for softmax strategy configuration."""

import pytest

from mo_net.samples.word2vec.softmax_strategy import SoftmaxConfig, SoftmaxStrategy


class TestSoftmaxStrategy:
    """Test SoftmaxStrategy enum."""

    def test_enum_values(self):
        """Test that all expected strategies exist."""
        assert SoftmaxStrategy.FULL.value == "full"
        assert SoftmaxStrategy.NEGATIVE_SAMPLING.value == "negative_sampling"
        assert SoftmaxStrategy.HIERARCHICAL.value == "hierarchical"

    def test_enum_members(self):
        """Test that enum has exactly the expected members."""
        strategies = list(SoftmaxStrategy)
        assert len(strategies) == 3
        assert SoftmaxStrategy.FULL in strategies
        assert SoftmaxStrategy.NEGATIVE_SAMPLING in strategies
        assert SoftmaxStrategy.HIERARCHICAL in strategies


class TestSoftmaxConfig:
    """Test SoftmaxConfig configuration class."""

    def test_full_softmax_config(self):
        """Test creating full softmax config."""
        config = SoftmaxConfig(strategy=SoftmaxStrategy.FULL)
        assert config.strategy == SoftmaxStrategy.FULL
        assert config.negative_samples is None

    def test_full_softmax_factory(self):
        """Test full_softmax() factory method."""
        config = SoftmaxConfig.full_softmax()
        assert config.strategy == SoftmaxStrategy.FULL
        assert config.negative_samples is None

    def test_negative_sampling_config(self):
        """Test creating negative sampling config."""
        config = SoftmaxConfig(
            strategy=SoftmaxStrategy.NEGATIVE_SAMPLING, negative_samples=5
        )
        assert config.strategy == SoftmaxStrategy.NEGATIVE_SAMPLING
        assert config.negative_samples == 5

    def test_negative_sampling_factory(self):
        """Test negative_sampling() factory method."""
        config = SoftmaxConfig.negative_sampling(k=10)
        assert config.strategy == SoftmaxStrategy.NEGATIVE_SAMPLING
        assert config.negative_samples == 10

    def test_negative_sampling_default(self):
        """Test negative_sampling() with default k=5."""
        config = SoftmaxConfig.negative_sampling()
        assert config.negative_samples == 5

    def test_hierarchical_config(self):
        """Test creating hierarchical softmax config."""
        config = SoftmaxConfig(strategy=SoftmaxStrategy.HIERARCHICAL)
        assert config.strategy == SoftmaxStrategy.HIERARCHICAL
        assert config.negative_samples is None

    def test_hierarchical_factory(self):
        """Test hierarchical_softmax() factory method."""
        config = SoftmaxConfig.hierarchical_softmax()
        assert config.strategy == SoftmaxStrategy.HIERARCHICAL
        assert config.negative_samples is None


class TestSoftmaxConfigValidation:
    """Test SoftmaxConfig validation logic."""

    def test_negative_sampling_requires_k(self):
        """Test that negative sampling requires negative_samples parameter."""
        with pytest.raises(ValueError, match="negative_samples must be specified"):
            SoftmaxConfig(strategy=SoftmaxStrategy.NEGATIVE_SAMPLING)

    def test_negative_samples_must_be_positive(self):
        """Test that negative_samples must be >= 1."""
        with pytest.raises(ValueError, match="negative_samples must be >= 1"):
            SoftmaxConfig(
                strategy=SoftmaxStrategy.NEGATIVE_SAMPLING, negative_samples=0
            )

        with pytest.raises(ValueError, match="negative_samples must be >= 1"):
            SoftmaxConfig(
                strategy=SoftmaxStrategy.NEGATIVE_SAMPLING, negative_samples=-1
            )

    def test_full_softmax_rejects_negative_samples(self):
        """Test that full softmax rejects negative_samples parameter."""
        with pytest.raises(
            ValueError, match="negative_samples should only be specified"
        ):
            SoftmaxConfig(strategy=SoftmaxStrategy.FULL, negative_samples=5)

    def test_hierarchical_rejects_negative_samples(self):
        """Test that hierarchical softmax rejects negative_samples parameter."""
        with pytest.raises(
            ValueError, match="negative_samples should only be specified"
        ):
            SoftmaxConfig(strategy=SoftmaxStrategy.HIERARCHICAL, negative_samples=5)


class TestSoftmaxConfigImmutability:
    """Test that SoftmaxConfig is immutable (frozen dataclass)."""

    def test_config_is_frozen(self):
        """Test that config attributes cannot be modified after creation."""
        config = SoftmaxConfig.full_softmax()

        with pytest.raises(Exception):  # FrozenInstanceError
            config.strategy = SoftmaxStrategy.NEGATIVE_SAMPLING  # type: ignore

        with pytest.raises(Exception):  # FrozenInstanceError
            config.negative_samples = 5  # type: ignore
