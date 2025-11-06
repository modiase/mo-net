import io
from dataclasses import dataclass
from typing import Final, cast

import jax
import jax.numpy as jnp
import pytest

from mo_net.model.layer.base import BadLayerId
from mo_net.model.layer.embedding import Embedding, ParametersType
from mo_net.protos import Activations, D, Dimensions

key: Final = jax.random.PRNGKey(42)


@dataclass(frozen=True)
class ForwardPropTestCase:
    name: str
    input_dimensions: Dimensions
    output_dimensions: Dimensions
    vocab_size: int
    parameters: ParametersType
    input_activations: jnp.ndarray
    expected_output: jnp.ndarray


@pytest.mark.parametrize(
    "test_case",
    [
        ForwardPropTestCase(
            name="simple_embedding_lookup",
            input_dimensions=(2,),
            output_dimensions=(2, 3),
            vocab_size=5,
            parameters=Embedding.Parameters.of(
                jnp.array(
                    [
                        [1.0, 2.0, 3.0],  # token 0
                        [4.0, 5.0, 6.0],  # token 1
                        [7.0, 8.0, 9.0],  # token 2
                        [10.0, 11.0, 12.0],  # token 3
                        [13.0, 14.0, 15.0],  # token 4
                    ]
                )
            ),
            input_activations=jnp.array([1, 3]),
            expected_output=jnp.array(
                [
                    [
                        [4.0, 5.0, 6.0],  # token 1
                        [10.0, 11.0, 12.0],  # token 3
                    ]
                ]
            ),
        ),
        ForwardPropTestCase(
            name="single_token_lookup",
            input_dimensions=(1,),
            output_dimensions=(1, 2),
            vocab_size=3,
            parameters=Embedding.Parameters.of(
                jnp.array(
                    [
                        [1.0, 2.0],  # token 0
                        [3.0, 4.0],  # token 1
                        [5.0, 6.0],  # token 2
                    ]
                )
            ),
            input_activations=jnp.array([2]),
            expected_output=jnp.array([[[5.0, 6.0]]]),  # token 2
        ),
        ForwardPropTestCase(
            name="batch_embedding_lookup",
            input_dimensions=(3,),
            output_dimensions=(3, 2),
            vocab_size=4,
            parameters=Embedding.Parameters.of(
                jnp.array(
                    [
                        [1.0, 2.0],  # token 0
                        [3.0, 4.0],  # token 1
                        [5.0, 6.0],  # token 2
                        [7.0, 8.0],  # token 3
                    ]
                )
            ),
            input_activations=jnp.array([0, 2, 1]),
            expected_output=jnp.array(
                [
                    [
                        [1.0, 2.0],  # token 0
                        [5.0, 6.0],  # token 2
                        [3.0, 4.0],  # token 1
                    ]
                ]
            ),
        ),
    ],
    ids=lambda test_case: test_case.name,
)
def test_embedding_forward_prop(test_case: ForwardPropTestCase):
    layer = Embedding(
        input_dimensions=test_case.input_dimensions,
        key=key,
        output_dimensions=test_case.output_dimensions,
        parameters=test_case.parameters,
        vocab_size=test_case.vocab_size,
    )
    assert jnp.allclose(
        layer.forward_prop(input_activations=Activations(test_case.input_activations)),
        test_case.expected_output,
    )


@dataclass(frozen=True)
class BackwardPropTestCase:
    name: str
    input_dimensions: Dimensions
    output_dimensions: Dimensions
    vocab_size: int
    parameters: ParametersType
    input_activations: jnp.ndarray
    dZ: jnp.ndarray
    expected_dE: jnp.ndarray


@pytest.mark.parametrize(
    "test_case",
    [
        BackwardPropTestCase(
            name="simple_gradient_accumulation",
            input_dimensions=(2,),
            output_dimensions=(2, 3),
            vocab_size=4,
            parameters=Embedding.Parameters.of(
                jnp.array(
                    [
                        [1.0, 2.0, 3.0],  # token 0
                        [4.0, 5.0, 6.0],  # token 1
                        [7.0, 8.0, 9.0],  # token 2
                        [10.0, 11.0, 12.0],  # token 3
                    ]
                )
            ),
            input_activations=jnp.array([1, 2]),
            dZ=jnp.array(
                [
                    [
                        [1.0, 1.0, 1.0],  # token 1
                        [2.0, 2.0, 2.0],  # token 2
                    ]
                ]
            ),
            expected_dE=jnp.array(
                [
                    [0.0, 0.0, 0.0],  # token 0
                    [1.0, 1.0, 1.0],  # token 1
                    [2.0, 2.0, 2.0],  # token 2
                    [0.0, 0.0, 0.0],  # token 3
                ]
            ),
        ),
        BackwardPropTestCase(
            name="duplicate_token_gradients",
            input_dimensions=(3,),
            output_dimensions=(3, 2),
            vocab_size=3,
            parameters=Embedding.Parameters.of(
                jnp.array(
                    [
                        [1.0, 2.0],  # token 0
                        [3.0, 4.0],  # token 1
                        [5.0, 6.0],  # token 2
                    ]
                )
            ),
            input_activations=jnp.array([1, 1, 0]),
            dZ=jnp.array(
                [
                    [
                        [1.0, 1.0],  # first token 1
                        [2.0, 2.0],  # second token 1
                        [3.0, 3.0],  # token 0
                    ]
                ]
            ),
            expected_dE=jnp.array(
                [
                    [3.0, 3.0],  # token 0
                    [
                        3.0,
                        3.0,
                    ],  # token 1
                    [0.0, 0.0],  # token 2
                ]
            ),
        ),
    ],
    ids=lambda test_case: test_case.name,
)
def test_embedding_backward_prop(test_case: BackwardPropTestCase):
    layer = Embedding(
        clip_gradients=False,
        input_dimensions=test_case.input_dimensions,
        key=key,
        output_dimensions=test_case.output_dimensions,
        parameters=test_case.parameters,
        vocab_size=test_case.vocab_size,
    )
    layer.forward_prop(input_activations=Activations(test_case.input_activations))
    layer.backward_prop(dZ=cast(D[Activations], test_case.dZ))
    assert layer.cache["dP"] is not None
    dP = cast(ParametersType, layer.cache["dP"])
    assert jnp.allclose(dP.embeddings, test_case.expected_dE)


@dataclass(frozen=True)
class ParameterUpdateTestCase:
    name: str
    input_dimensions: Dimensions
    output_dimensions: Dimensions
    vocab_size: int
    initial_parameters: ParametersType
    input_activations: jnp.ndarray
    dZ: jnp.ndarray
    expected_updated_embeddings: jnp.ndarray


@pytest.mark.parametrize(
    "test_case",
    [
        ParameterUpdateTestCase(
            name="simple_parameter_update",
            input_dimensions=(1,),
            output_dimensions=(1, 2),
            vocab_size=2,
            initial_parameters=Embedding.Parameters.of(
                jnp.array(
                    [
                        [1.0, 2.0],  # token 0
                        [3.0, 4.0],  # token 1
                    ]
                )
            ),
            input_activations=jnp.array([0]),
            dZ=jnp.array([[[0.5, 0.5]]]),  # token 0
            expected_updated_embeddings=jnp.array(
                [
                    [1.5, 2.5],  # token 0
                    [3.0, 4.0],  # token 1
                ]
            ),
        ),
        ParameterUpdateTestCase(
            name="multiple_token_update",
            input_dimensions=(2,),
            output_dimensions=(2, 2),
            vocab_size=3,
            initial_parameters=Embedding.Parameters.of(
                jnp.array(
                    [
                        [1.0, 2.0],  # token 0
                        [3.0, 4.0],  # token 1
                        [5.0, 6.0],  # token 2
                    ]
                )
            ),
            input_activations=jnp.array([1, 2]),
            dZ=jnp.array(
                [
                    [
                        [1.0, 1.0],  # token 1
                        [2.0, 2.0],  # token 2
                    ]
                ]
            ),
            expected_updated_embeddings=jnp.array(
                [
                    [1.0, 2.0],  # token 0
                    [4.0, 5.0],  # token 1
                    [7.0, 8.0],  # token 2
                ]
            ),
        ),
    ],
    ids=lambda test_case: test_case.name,
)
def test_embedding_parameter_update(test_case: ParameterUpdateTestCase):
    layer = Embedding(
        clip_gradients=False,
        input_dimensions=test_case.input_dimensions,
        key=key,
        output_dimensions=test_case.output_dimensions,
        parameters=test_case.initial_parameters,
        vocab_size=test_case.vocab_size,
    )
    layer.forward_prop(input_activations=Activations(test_case.input_activations))
    layer.backward_prop(dZ=cast(D[Activations], test_case.dZ))
    layer.update_parameters()
    assert jnp.allclose(
        layer.parameters.embeddings, test_case.expected_updated_embeddings
    )


@pytest.mark.parametrize(
    "init_method,vocab_size,embedding_dim",
    [
        (Embedding.Parameters.xavier, 10, 5),
        (Embedding.Parameters.he, 8, 4),
        (Embedding.Parameters.random, 6, 3),
    ],
)
def test_embedding_initialization_methods(init_method, vocab_size, embedding_dim):
    layer = Embedding(
        input_dimensions=(vocab_size,),
        output_dimensions=(vocab_size, embedding_dim),
        vocab_size=vocab_size,
        parameters_init_fn=init_method,
        key=key,
    )
    assert layer.parameters.embeddings.shape == (vocab_size, embedding_dim)
    assert jnp.isfinite(layer.parameters.embeddings).all()


def test_embedding_cache_initialization():
    layer = Embedding(
        input_dimensions=(3,),
        output_dimensions=(3, 2),
        vocab_size=5,
        key=key,
    )
    assert layer.cache["input_indices"] is None
    assert layer.cache["output_activations"] is None
    assert layer.cache["dP"] is None


def test_embedding_forward_prop_caches_indices():
    layer = Embedding(
        input_dimensions=(2,),
        output_dimensions=(2, 3),
        vocab_size=4,
        key=key,
    )
    input_activations = jnp.array([1, 2])
    layer.forward_prop(input_activations=Activations(input_activations))
    assert layer.cache["input_indices"] is not None
    assert jnp.array_equal(layer.cache["input_indices"], jnp.array([[1, 2]]))


def test_embedding_gradient_clipping():
    layer = Embedding(
        input_dimensions=(2,),
        output_dimensions=(2, 2),
        vocab_size=3,
        clip_gradients=True,
        weight_max_norm=1.0,
        key=key,
    )
    input_activations = jnp.array([0, 1])
    layer.forward_prop(input_activations=Activations(input_activations))
    large_dZ = jnp.array([[10.0, 10.0], [10.0, 10.0]])
    layer.backward_prop(dZ=cast(D[Activations], large_dZ))
    assert layer.cache["dP"] is not None
    dP = cast(ParametersType, layer.cache["dP"])
    grad_norm = jnp.linalg.norm(dP.embeddings)
    assert grad_norm <= layer._weight_max_norm * jnp.sqrt(dP.embeddings.size)


def test_embedding_frozen_parameters():
    layer = Embedding(
        input_dimensions=(2,),
        output_dimensions=(2, 2),
        vocab_size=3,
        freeze_parameters=True,
        key=key,
    )
    original_embeddings = layer.parameters.embeddings.copy()
    input_activations = jnp.array([0, 1])
    layer.forward_prop(input_activations=Activations(input_activations))
    layer.backward_prop(dZ=cast(D[Activations], jnp.array([[1.0, 1.0], [1.0, 1.0]])))
    layer.update_parameters()
    assert jnp.allclose(layer.parameters.embeddings, original_embeddings)


def test_embedding_empty_gradient():
    layer = Embedding(
        input_dimensions=(2,),
        output_dimensions=(2, 2),
        vocab_size=3,
        key=key,
    )
    empty_grad = layer.empty_gradient()
    dP = cast(ParametersType, empty_grad)
    assert dP.embeddings.shape == layer.parameters.embeddings.shape
    assert jnp.allclose(dP.embeddings, 0.0)


def test_embedding_parameter_count():
    layer = Embedding(
        input_dimensions=(2,),
        output_dimensions=(2, 3),
        vocab_size=4,
        key=key,
    )
    expected_count = 4 * 3
    assert layer.parameter_count == expected_count


@pytest.mark.skip(reason="TODO: Fix layer name collision with appropriate fixture")
def test_embedding_serialization_deserialization():
    layer = Embedding(
        input_dimensions=(2,),
        output_dimensions=(2, 3),
        vocab_size=4,
        layer_id="test_embedding_serial_unique",
        key=key,
    )
    serialized = layer.serialize()
    deserialized = serialized.deserialize()

    assert deserialized.layer_id == layer.layer_id
    assert deserialized.input_dimensions == layer.input_dimensions
    assert deserialized.output_dimensions == layer.output_dimensions
    assert deserialized.vocab_size == layer.vocab_size
    assert jnp.allclose(deserialized.parameters.embeddings, layer.parameters.embeddings)


def test_embedding_serialize_deserialize_parameters_with_wrong_layer_id():
    layer = Embedding(
        input_dimensions=(2,),
        output_dimensions=(2, 3),
        vocab_size=4,
        layer_id="test_embedding_wrong_id",
        key=key,
    )
    layer.forward_prop(input_activations=Activations(jnp.array([0, 1])))
    layer.backward_prop(
        dZ=cast(D[Activations], jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))
    )

    buffer = io.BytesIO()
    layer.write_serialized_parameters(buffer)
    buffer.seek(0)

    other_layer = Embedding(
        input_dimensions=(2,),
        output_dimensions=(2, 3),
        vocab_size=4,
        layer_id="different_embedding",
        key=key,
    )

    with pytest.raises(BadLayerId):
        other_layer.read_serialized_parameters(buffer)


def test_embedding_error_on_backward_prop_without_forward():
    layer = Embedding(
        input_dimensions=(2,),
        output_dimensions=(2, 3),
        vocab_size=4,
        key=key,
    )
    with pytest.raises(ValueError, match="Input indices not set"):
        layer.backward_prop(
            dZ=cast(D[Activations], jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))
        )


def test_embedding_error_on_update_without_gradients():
    layer = Embedding(
        input_dimensions=(2,),
        output_dimensions=(2, 3),
        vocab_size=4,
        key=key,
    )
    with pytest.raises(ValueError, match="Gradient not set"):
        layer.update_parameters()


def test_embedding_mathematical_properties():
    layer = Embedding(
        input_dimensions=(2,),
        output_dimensions=(2, 3),
        vocab_size=4,
        key=key,
    )
    input_activations = jnp.array([0, 1])

    output = layer.forward_prop(input_activations=Activations(input_activations))
    assert output.shape == (1, 2, 3)
    assert jnp.isfinite(output).all()

    dZ = jnp.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    dX = layer.backward_prop(dZ=cast(D[Activations], dZ))
    assert cast(jnp.ndarray, dX).shape == dZ.shape
    assert jnp.isfinite(cast(jnp.ndarray, dX)).all()


def test_embedding_zero_input():
    layer = Embedding(
        input_dimensions=(2,),
        output_dimensions=(2, 3),
        vocab_size=4,
        key=key,
    )
    input_activations = jnp.array([0, 0])
    output = layer.forward_prop(input_activations=Activations(input_activations))
    expected = layer.parameters.embeddings[0:1].repeat(2, axis=0)
    assert jnp.allclose(output, expected)


def test_embedding_large_batch():
    layer = Embedding(
        input_dimensions=(100,),
        output_dimensions=(100, 5),
        vocab_size=1000,
        key=key,
    )
    input_activations = jax.random.randint(key, (100,), 0, 1000)
    output = layer.forward_prop(input_activations=Activations(input_activations))
    assert output.shape == (
        1,
        100,
        5,
    )
    assert jnp.isfinite(output).all()


def test_embedding_gradient_accumulation():
    layer = Embedding(
        input_dimensions=(3,),
        output_dimensions=(3, 2),
        vocab_size=4,
        clip_gradients=False,
        key=key,
    )
    input_activations = jnp.array([1, 1, 1])
    layer.forward_prop(input_activations=Activations(input_activations))
    dZ = jnp.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    layer.backward_prop(dZ=cast(D[Activations], dZ))

    assert layer.cache["dP"] is not None
    dP = cast(ParametersType, layer.cache["dP"])
    token_1_gradient = dP.embeddings[1]
    expected_gradient = jnp.array([3.0, 3.0])
    assert jnp.allclose(token_1_gradient, expected_gradient)


def test_embedding_embedding_shape_validation():
    with pytest.raises(ValueError, match="Embedding matrix shape"):
        Embedding(
            input_dimensions=(2,),
            output_dimensions=(2, 3),
            vocab_size=4,
            parameters=Embedding.Parameters.of(jnp.zeros((3, 3))),
            key=key,
        )


@pytest.mark.parametrize("store_output", [True, False])
def test_embedding_output_storage_option(store_output):
    layer = Embedding(
        input_dimensions=(2,),
        output_dimensions=(2, 3),
        vocab_size=4,
        store_output_activations=store_output,
        key=key,
    )
    input_activations = jnp.array([0, 1])
    output = layer.forward_prop(input_activations=Activations(input_activations))

    if store_output:
        assert layer.cache["output_activations"] is not None
        assert jnp.allclose(layer.cache["output_activations"], output)
    else:
        assert layer.cache["output_activations"] is None


def test_embedding_constructor_edge_cases():
    layer = Embedding(
        input_dimensions=(1,),
        output_dimensions=(1, 1),
        vocab_size=1,
        key=key,
    )
    assert layer.vocab_size == 1
    assert layer.parameters.embeddings.shape == (1, 1)

    layer = Embedding(
        input_dimensions=(2,),
        output_dimensions=(2, 100),
        vocab_size=10,
        key=key,
    )
    assert layer.parameters.embeddings.shape == (10, 100)


def test_embedding_gradient_operation_interface():
    layer = Embedding(
        input_dimensions=(2,),
        output_dimensions=(2, 3),
        vocab_size=4,
        key=key,
    )

    def grad_callback(grad_layer):
        assert grad_layer is layer

    layer.gradient_operation(grad_callback)


def test_embedding_reinitialise():
    def custom_init(vocab_size: int, embedding_dim: int, key: jax.Array):
        return Embedding.Parameters(
            embeddings=jax.random.normal(key, (vocab_size, embedding_dim))
        )

    layer = Embedding(
        input_dimensions=(2,),
        output_dimensions=(2, 3),
        vocab_size=4,
        parameters_init_fn=custom_init,
        key=key,
    )
    original_embeddings = layer.parameters.embeddings.copy()

    def different_init(vocab_size: int, embedding_dim: int, key: jax.Array):
        return Embedding.Parameters(
            embeddings=jax.random.normal(key, (vocab_size, embedding_dim))
        )

    layer._parameters_init_fn = different_init
    layer.reinitialise()

    assert not jnp.allclose(layer.parameters.embeddings, original_embeddings)
    assert layer.parameters.embeddings.shape == original_embeddings.shape


def test_embedding_parameter_nbytes():
    layer = Embedding(
        input_dimensions=(2,),
        output_dimensions=(2, 3),
        vocab_size=4,
        key=key,
    )
    expected_nbytes = layer.parameters.embeddings.nbytes
    assert layer.parameter_nbytes == expected_nbytes


def test_embedding_vocab_size_property():
    layer = Embedding(
        input_dimensions=(2,),
        output_dimensions=(2, 3),
        vocab_size=42,
        key=key,
    )
    assert layer.vocab_size == 42
