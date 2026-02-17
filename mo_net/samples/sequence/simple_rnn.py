import jax
import jax.numpy as jnp

from mo_net.functions import Tanh, sparse_cross_entropy
from mo_net.model.model import Model


def generate_sequence_data(
    n_samples: int, seq_len: int, feature_dim: int, key: jax.Array
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate synthetic sequence data for binary classification.

    Sequences are classified based on whether their sum exceeds a threshold.
    """
    key1, key2 = jax.random.split(key)
    sequences = jax.random.normal(key1, (n_samples, seq_len, feature_dim))
    sequence_sums = jnp.sum(sequences, axis=(1, 2))
    labels = (sequence_sums > jnp.median(sequence_sums)).astype(jnp.int32)
    return sequences, labels


def main():
    key = jax.random.PRNGKey(42)

    n_train = 1000
    n_val = 200
    seq_len = 10
    feature_dim = 5
    hidden_dim = 16
    output_dim = 2

    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
    X_train, Y_train = generate_sequence_data(n_train, seq_len, feature_dim, subkey1)
    X_val, Y_val = generate_sequence_data(n_val, seq_len, feature_dim, subkey2)

    model = Model.rnn_of(
        module_dimensions=[(feature_dim,), (hidden_dim,), (output_dim,)],
        activation_fn=Tanh(),
        return_sequences=False,
        key=subkey3,
    )

    print(f"Created RNN model: {model.print()}")
    print(f"Parameter count: {model.parameter_count}")
    print(f"Input shape: {X_train.shape}")
    print(f"Output shape: {model.forward_prop(X_train[:1]).shape}")

    Y_pred = model.forward_prop(X_train[:10])
    loss = sparse_cross_entropy(Y_pred, Y_train[:10])
    print(f"\nInitial loss (first 10 samples): {loss:.4f}")

    predictions = model.predict(X_val[:10])
    accuracy = jnp.mean(predictions == Y_val[:10])
    print(f"Initial accuracy (first 10 val samples): {accuracy:.2%}")


if __name__ == "__main__":
    main()
