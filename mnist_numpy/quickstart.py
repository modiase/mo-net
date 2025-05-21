from mnist_numpy.config import TrainingParameters
from mnist_numpy.functions import ReLU
from mnist_numpy.model.block.convolution import Convolution
from mnist_numpy.model.block.norm import LayerNormOptions, Norm
from mnist_numpy.model.layer.output import SoftmaxOutputLayer
from mnist_numpy.model.layer.reshape import Reshape
from mnist_numpy.model.model import Model
from mnist_numpy.protos import NormalisationType


def mnist_mlp(training_parameters: TrainingParameters) -> Model:
    return Model.mlp_of(
        module_dimensions=((784,), (1000,), (1000,), (10,)),
        activation_fn=ReLU,
        normalisation_type=NormalisationType.LAYER,
        tracing_enabled=training_parameters.trace_logging,
    )


def mnist_cnn(training_parameters: TrainingParameters) -> Model:
    return Model(
        input_dimensions=(1, 28, 28),
        hidden=(
            (r := Reshape(input_dimensions=(784,), output_dimensions=(1, 28, 28))),
            (
                c1 := Convolution(
                    input_dimensions=r.output_dimensions,
                    n_kernels=1,
                    kernel_size=(5, 5),
                    pool_size=(5, 5),
                )
            ),
            (
                c2 := Convolution(
                    input_dimensions=c1.output_dimensions,
                    n_kernels=1,
                    kernel_size=(3, 3),
                    pool_size=(3, 3),
                    flatten_output=True,
                )
            ),
            (
                Norm(
                    input_dimensions=c2.output_dimensions,
                    output_dimensions=(10,),
                    activation_fn=ReLU,
                    options=LayerNormOptions(),
                    store_output_activations=training_parameters.trace_logging,
                )
            ),
        ),
        output=SoftmaxOutputLayer(
            input_dimensions=(10,),
        ),
    )
