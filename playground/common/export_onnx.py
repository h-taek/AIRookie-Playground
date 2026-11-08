
import tensorflow as tf
from tensorflow.keras import layers
import tf2onnx
import numpy as np

# ONNX 내보내기는 CPU 에서 한다.
#
# 학습은 JAX 가 GPU 메모리를 미리 크게 잡은 상태로 돈다. 체크포인트를 저장할 때마다
# 이 모듈이 케라스 모델을 만들어 순전파를 한 번 돌리는데, 그때 TensorFlow 가 같은
# GPU 를 잡으려다 메모리를 얻지 못하고 죽는다. 맥에는 CUDA 가 없어 이 충돌은 서버에서만
# 드러나며, 첫 체크포인트가 학습 시작 직후(step 0)에 저장되므로 몇 분 만에 터진다.
#
# 이 줄은 TensorFlow 가 초기화되기 전에 실행되어야 하므로 import 직후에 둔다.
tf.config.set_visible_devices([], "GPU")

def export_onnx(
    params, act_size, ppo_params, obs_size, output_path="ONNX.onnx"
):
    print(" === EXPORT ONNX === ")

    # inference_fn = make_inference_fn(params, deterministic=True)

    class MLP(tf.keras.Model):
        def __init__(
            self,
            layer_sizes,
            activation=tf.nn.relu,
            kernel_init="lecun_uniform",
            activate_final=False,
            bias=True,
            layer_norm=False,
            mean_std=None,
        ):
            super().__init__()

            self.layer_sizes = layer_sizes
            self.activation = activation
            self.kernel_init = kernel_init
            self.activate_final = activate_final
            self.bias = bias
            self.layer_norm = layer_norm

            if mean_std is not None:
                self.mean = tf.Variable(mean_std[0], trainable=False, dtype=tf.float32)
                self.std = tf.Variable(mean_std[1], trainable=False, dtype=tf.float32)
            else:
                self.mean = None
                self.std = None

            self.mlp_block = tf.keras.Sequential(name="MLP_0")
            for i, size in enumerate(self.layer_sizes):
                dense_layer = layers.Dense(
                    size,
                    activation=self.activation,
                    kernel_initializer=self.kernel_init,
                    name=f"hidden_{i}",
                    use_bias=self.bias,
                )
                self.mlp_block.add(dense_layer)
                if self.layer_norm:
                    self.mlp_block.add(
                        layers.LayerNormalization(name=f"layer_norm_{i}")
                    )
            if not self.activate_final and self.mlp_block.layers:
                if (
                    hasattr(self.mlp_block.layers[-1], "activation")
                    and self.mlp_block.layers[-1].activation is not None
                ):
                    self.mlp_block.layers[-1].activation = None

            self.submodules = [self.mlp_block]

        def call(self, inputs):
            if isinstance(inputs, list):
                inputs = inputs[0]
            if self.mean is not None and self.std is not None:
                print(self.mean.shape, self.std.shape)
                inputs = (inputs - self.mean) / self.std
            logits = self.mlp_block(inputs)
            loc, _ = tf.split(logits, 2, axis=-1)
            return tf.tanh(loc)

    def make_policy_network(
        param_size,
        mean_std,
        hidden_layer_sizes=[256, 256],
        activation=tf.nn.relu,
        kernel_init="lecun_uniform",
        layer_norm=False,
    ):
        policy_network = MLP(
            layer_sizes=list(hidden_layer_sizes) + [param_size],
            activation=activation,
            kernel_init=kernel_init,
            layer_norm=layer_norm,
            mean_std=mean_std,
        )
        return policy_network

    mean = params[0].mean["state"]
    std = params[0].std["state"]

    # Convert mean/std jax arrays to tf tensors.
    mean_std = (tf.convert_to_tensor(mean), tf.convert_to_tensor(std))

    tf_policy_network = make_policy_network(
        param_size=act_size * 2,
        mean_std=mean_std,
        hidden_layer_sizes=ppo_params.network_factory.policy_hidden_layer_sizes,
        activation=tf.nn.swish,
    )

    example_input = tf.zeros((1, obs_size))
    example_output = tf_policy_network(example_input)
    print(example_output.shape)

    def transfer_weights(jax_params, tf_model):
        """
        Transfer weights from a JAX parameter dictionary to the TensorFlow model.

        Parameters:
        - jax_params: dict
        Nested dictionary with structure {block_name: {layer_name: {params}}}.
        For example:
        {
            'CNN_0': {
            'Conv_0': {'kernel': np.ndarray},
            'Conv_1': {'kernel': np.ndarray},
            'Conv_2': {'kernel': np.ndarray},
            },
            'MLP_0': {
            'hidden_0': {'kernel': np.ndarray, 'bias': np.ndarray},
            'hidden_1': {'kernel': np.ndarray, 'bias': np.ndarray},
            'hidden_2': {'kernel': np.ndarray, 'bias': np.ndarray},
            }
        }

        - tf_model: tf.keras.Model
        An instance of the adapted VisionMLP model containing named submodules and layers.
        """
        for layer_name, layer_params in jax_params.items():
            try:
                tf_layer = tf_model.get_layer("MLP_0").get_layer(name=layer_name)
            except ValueError:
                print(f"Layer {layer_name} not found in TensorFlow model.")
                continue
            if isinstance(tf_layer, tf.keras.layers.Dense):
                kernel = np.array(layer_params["kernel"])
                bias = np.array(layer_params["bias"])
                print(
                    f"Transferring Dense layer {layer_name}, kernel shape {kernel.shape}, bias shape {bias.shape}"
                )
                tf_layer.set_weights([kernel, bias])
            else:
                print(f"Unhandled layer type in {layer_name}: {type(tf_layer)}")

        print("Weights transferred successfully.")

    def extract_policy_params(p):
        policy = getattr(p, 'policy', p)
        return policy.get('params') if isinstance(policy, dict) else None

    transfer_weights(extract_policy_params(params[1]), tf_policy_network)

    # Example inputs for the model
    test_input = [np.ones((1, obs_size), dtype=np.float32)]

    # Define the TensorFlow input signature
    spec = [
        tf.TensorSpec(shape=(1, obs_size), dtype=tf.float32, name="obs")
    ]

    tensorflow_pred = tf_policy_network(test_input)[0]
    # Build the model by calling it with example data
    print(f"Tensorflow prediction: {tensorflow_pred}")

    tf_policy_network.output_names = ["continuous_actions"]

    # opset 11 matches isaac lab.
    model_proto, _ = tf2onnx.convert.from_keras(
        tf_policy_network, input_signature=spec, opset=11, output_path=output_path
    )

    # 원본은 여기서 작업 디렉터리의 "ONNX.onnx" 에도 한 벌을 더 썼다("For Antoine :)").
    # 그 파일은 저장소 루트의 동등성 기준 정책이며 tools/reference_obs_provenance.md 가
    # SHA256 으로 못 박아 두고 있다. 체크포인트를 저장할 때마다 기준 파일이 조용히
    # 바뀌면 기준 npz 와 짝이 맞지 않게 되므로 쓰지 아니한다. 내보낸 파일은
    # output_path(체크포인트 폴더 안)에 있다.
    return
