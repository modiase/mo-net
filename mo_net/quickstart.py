from mo_net.config import TrainingParameters
from mo_net.functions import ReLU
from mo_net.model.layer.output import SoftmaxOutputLayer
from mo_net.model.layer.reshape import Reshape
from mo_net.model.model import Model
from mo_net.model.module.convolution import Convolution
from mo_net.model.module.norm import LayerNormOptions, Norm
from mo_net.protos import NormalisationType


def mnist_mlp(training_parameters: TrainingParameters) -> Model:
    return Model.mlp_of(
        module_dimensions=((784,), (100,), (100,), (10,)),
        activation_fn=ReLU,
        normalisation_type=NormalisationType.LAYER,
        tracing_enabled=training_parameters.trace_logging,
    )


def mnist_cnn(training_parameters: TrainingParameters) -> Model:
    return Model(
        input_dimensions=(1, 28, 28),
        hidden=(
            (r := Reshape(input_dimensions=(784,), output_dimensions=(1, 28, 28))),
            c1 := Convolution(
                input_dimensions=r.output_dimensions,
                n_kernels=8,
                kernel_size=(5, 5),
                stride=2,
            ),
            (
                c2 := Convolution(
                    input_dimensions=c1.output_dimensions,
                    n_kernels=8,
                    kernel_size=(5, 5),
                    stride=2,
                    flatten_output=True,
                )
            ),
            Norm(
                input_dimensions=c2.output_dimensions,
                output_dimensions=(10,),
                activation_fn=ReLU,
                options=LayerNormOptions(),
                store_output_activations=training_parameters.trace_logging,
            ),
        ),
        output=SoftmaxOutputLayer(
            input_dimensions=(10,),
        ),
    )
