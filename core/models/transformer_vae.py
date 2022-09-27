import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy, Reduction, MeanSquaredError
from einops.layers.keras import Rearrange as RearrangeEinops

from core.models import VAE
from core.models.utils import _KLDivergenceLayer, TransformerEncoderBlock, _Sampling
from core.constants import N_CELLS_R, N_CELLS_PHI, N_CELLS_Z, PATCH_R, PATCH_P, PATCH_Z


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


class TransformerVAE(VAE):
    def _build_encoder(self) -> Model:
        """ Based on a list of intermediate dimensions, activation function and initializers for kernel and bias builds
        the encoder.

        Returns:
             Encoder is returned as a keras.Model.

        """
        with self._strategy.scope():
            # Prepare input layer.
            x_input, e_input, angle_input, geo_input, eps_input = self._prepare_input_layers(for_encoder=True)
            num_patches = PATCH_R * PATCH_P * PATCH_Z
            feature_dim = (N_CELLS_R * N_CELLS_PHI * N_CELLS_Z) // num_patches

            # Patchify and concatenate
            _x_input = layers.Reshape((N_CELLS_R, N_CELLS_PHI, N_CELLS_Z))(x_input)
            patchified = RearrangeEinops("b (i r) (j p) (k z) -> b (i j k) (r p z)", i=PATCH_R, j=PATCH_P, k=PATCH_Z)(_x_input)
            _e_input = layers.Reshape((1, feature_dim))(layers.RepeatVector(feature_dim)(e_input))
            _angle_input = layers.Reshape((1, feature_dim))(layers.RepeatVector(feature_dim)(angle_input))
            _geo_input = layers.Reshape((2, feature_dim))(layers.RepeatVector(feature_dim)(geo_input))
            patches_combined = layers.concatenate([patchified, _e_input, _angle_input, _geo_input], axis=-2)

            # Linear projection and positional embeddings
            encoded_patches = PatchEncoder(num_patches + 4, 256)(patches_combined)

            # Transformer Encoder
            x = TransformerEncoderBlock(encoded_patches, 4, 256, [256,])
            x = layers.Dense(64)(x)
            x = TransformerEncoderBlock(x, 8, 64, [64,])
            x = layers.Dense(16)(x)
            x = TransformerEncoderBlock(x, 16, 16, [16,])

            # Handling transformer representations
            representation = layers.LayerNormalization(epsilon=1e-6)(x)
            representation = layers.Flatten()(representation)
            representation = layers.Dropout(0.2)(representation)

            # Add Dense layer to get description of multidimensional Gaussian distribution in terms of mean
            # and log(variance).
            z_mean = layers.Dense(self.latent_dim, name="z_mean")(representation)
            z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(representation)
            # Add KLDivergenceLayer responsible for calculation of KL loss.
            z_mean, z_log_var = _KLDivergenceLayer()([z_mean, z_log_var])
            # Sample a probe from the distribution.
            encoder_output = _Sampling()([z_mean, z_log_var, eps_input])
            # Create model.
            encoder = Model(inputs=[x_input, e_input, angle_input, geo_input, eps_input], outputs=encoder_output,
                            name="encoder")
        return encoder
