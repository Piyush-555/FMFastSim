import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy, Reduction, MeanSquaredError
from einops.layers.keras import Rearrange as RearrangeEinops

from core.models import ModelHandler
from utils.optimizer import OptimizerFactory
from core.models.utils import TransformerEncoderBlock
from core.models.transformer_vae import PatchEncoder
from core.constants import N_CELLS_R, N_CELLS_PHI, N_CELLS_Z, NUM_LAYERS, NUM_HEADS, \
    PROJECTION_DIM, FF_DIMS, MASKING_PERCENT, MASK_AFTER_EMBEDDING, PATCH_R, PATCH_P, PATCH_Z


class MaskedPatchEncoder(PatchEncoder):
    def __init__(self, mask_percent=0.75, mask_after_embedding=True, *args):
        super().__init__(*args)
        self.mask_percent = mask_percent
        self.mask_after_embedding = mask_after_embedding

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        mask = tf.random.uniform(shape=[self.num_patches,]) < self.mask_percent
        mask = tf.cast(tf.reshape(~mask, (-1, 1)), tf.float32)
        if not self.mask_after_embedding:
            patch = patch * mask
        encoded = self.projection(patch) + self.position_embedding(positions)
        if self.mask_after_embedding:
            patch = patch * mask
        return encoded


class TransformerMLM(ModelHandler):
    def __post_init__(self):
        self._num_layers = NUM_LAYERS
        self._num_heads = NUM_HEADS
        self._projection_dim = PROJECTION_DIM
        self._ff_dims = FF_DIMS
        super().__post_init__()

    def _get_wandb_extra_config(self):
        return {
            "num_layers": self._num_layers,
            "num_heads": self._num_heads,
            "projection_dim": self._projection_dim,
            "ff_dims": self._ff_dims
        }

    def _build_transformer(self) -> Model:
        # with self._strategy.scope(): TODO
        # Prepare input layer.
        x_input = self._prepare_input_layers()
        num_patches = PATCH_R * PATCH_P * PATCH_Z
        feature_dim = (N_CELLS_R * N_CELLS_PHI * N_CELLS_Z) / num_patches

        # Patchify
        _x_input = layers.Reshape((N_CELLS_R, N_CELLS_PHI, N_CELLS_Z))(x_input)
        patchified = RearrangeEinops("b (i r) (j p) (k z) -> b (i j k) (r p z)", i=PATCH_R, j=PATCH_P, k=PATCH_Z)(_x_input)

        # Masking, Linear projection and positional embeddings
        x = MaskedPatchEncoder(MASKING_PERCENT, MASK_AFTER_EMBEDDING, num_patches, self._projection_dim)(patchified)

        # Transformer Encoder
        for i in range(self._num_layers):
            x = TransformerEncoderBlock(x, self._num_heads[i], self._projection_dim, self._ff_dims[i])

        # Final layers
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Dense(feature_dim, activation=tf.nn.gelu)(x)
        x = layers.Dense(feature_dim, activation='sigmoid')(x)
        x = RearrangeEinops("b (i j k) (r p z) -> b (i r) (j p) (k z)",
            i=PATCH_R, j=PATCH_P, k=PATCH_Z, r=N_CELLS_R//PATCH_R, p=N_CELLS_PHI//PATCH_P, z=N_CELLS_Z//PATCH_Z)(x)
        out = layers.Reshape((-1,))(x)

        transformer = Model(inputs=[x_input,], outputs=out,
                        name="transformer")
        return transformer

    def _build_and_compile_new_model(self):
        # # Compile model within a distributed strategy.
        # with self._strategy.scope():
        # Build transformer.
        self.model = self._build_transformer()
        # Manufacture an optimizer and compile model with.
        optimizer = OptimizerFactory.create_optimizer(self._optimizer_type, self._learning_rate)
        self.model.compile(optimizer=optimizer, loss=BinaryCrossentropy(reduction=Reduction.SUM))

    def _prepare_input_layers(self):
        x_input = layers.Input(shape=self._original_dim)
        return x_input

    def _get_model_specific_data(self, dataset, e_cond, angle_cond, geo_cond):
        return [dataset,]

    def get_decoder(self):
        return self.model
