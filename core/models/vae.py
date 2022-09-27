import numpy as np
from tensorflow.keras.layers import BatchNormalization, Dense, concatenate
from tensorflow.keras.losses import BinaryCrossentropy, Reduction, MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from core.models import ModelHandler
from utils.optimizer import OptimizerFactory
from core.models.utils import _KLDivergenceLayer, _Sampling, _PhysicsLosses
from core.constants import LATENT_DIM, ACTIVATION, OUT_ACTIVATION, KERNEL_INITIALIZER, \
    BIAS_INITIALIZER, INTERMEDIATE_DIMS, INLCUDE_PHYSICS_LOSS


class VAEArch(Model):
    def get_config(self):
        config = super().get_config()
        config["encoder"] = self.encoder
        config["decoder"] = self.decoder
        return config

    def call(self, inputs, training=None, mask=None):
        _, e_input, angle_input, geo_input, _ = inputs
        z = self.encoder(inputs)
        return self.decoder([z, e_input, angle_input, geo_input])

    def __init__(self, encoder, decoder, **kwargs):
        super(VAEArch, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self._set_inputs(inputs=self.encoder.inputs, outputs=self(self.encoder.inputs))


def _Loss(y_true, y_pred):
    reconstruction_loss = BinaryCrossentropy(reduction=Reduction.SUM)(y_true, y_pred)
    loss = reconstruction_loss
    if INLCUDE_PHYSICS_LOSS:
        loss += _PhysicsLosses(y_true, y_pred)
    return loss


class VAE(ModelHandler):
    def __post_init__(self):
        self.latent_dim = LATENT_DIM
        self._intermediate_dims = INTERMEDIATE_DIMS
        self._activation = ACTIVATION
        self._out_activation = OUT_ACTIVATION
        self._kernel_initializer = KERNEL_INITIALIZER
        self._bias_initializer = BIAS_INITIALIZER
        super().__post_init__()

    def _get_wandb_extra_config(self):
        return {
            "activation": self._activation,
            "out_activation": self._out_activation,
            "intermediate_dims": self._intermediate_dims,
            "latent_dim": self.latent_dim
        }

    def _build_and_compile_new_model(self) -> None:
        # Build encoder and decoder.
        encoder = self._build_encoder()
        decoder = self._build_decoder()

        # Compile model within a distributed strategy.
        with self._strategy.scope():
            # Build VAE.
            self.model = VAEArch(encoder, decoder)
            # Manufacture an optimizer and compile model with.
            optimizer = OptimizerFactory.create_optimizer(self._optimizer_type, self._learning_rate)
            self.model.compile(optimizer=optimizer, loss=_Loss)

    def _build_encoder(self) -> Model:
        """ Based on a list of intermediate dimensions, activation function and initializers for kernel and bias builds
        the encoder.

        Returns:
             Encoder is returned as a keras.Model.

        """

        with self._strategy.scope():
            # Prepare input layer.
            x_input, e_input, angle_input, geo_input, eps_input = self._prepare_input_layers(for_encoder=True)
            x = concatenate([x_input, e_input, angle_input, geo_input])
            # Construct hidden layers (Dense and Batch Normalization).
            for intermediate_dim in self._intermediate_dims:
                x = Dense(units=intermediate_dim, activation=self._activation,
                          kernel_initializer=self._kernel_initializer,
                          bias_initializer=self._bias_initializer)(x)
                x = BatchNormalization()(x)
            # Add Dense layer to get description of multidimensional Gaussian distribution in terms of mean
            # and log(variance).
            z_mean = Dense(self.latent_dim, name="z_mean")(x)
            z_log_var = Dense(self.latent_dim, name="z_log_var")(x)
            # Add KLDivergenceLayer responsible for calculation of KL loss.
            z_mean, z_log_var = _KLDivergenceLayer()([z_mean, z_log_var])
            # Sample a probe from the distribution.
            encoder_output = _Sampling()([z_mean, z_log_var, eps_input])
            # Create model.
            encoder = Model(inputs=[x_input, e_input, angle_input, geo_input, eps_input], outputs=encoder_output,
                            name="encoder")
        return encoder

    def _build_decoder(self) -> Model:
        """ Based on a list of intermediate dimensions, activation function and initializers for kernel and bias builds
        the decoder.

        Returns:
             Decoder is returned as a keras.Model.

        """

        with self._strategy.scope():
            # Prepare input layer.
            latent_input, e_input, angle_input, geo_input = self._prepare_input_layers(for_encoder=False)
            x = concatenate([latent_input, e_input, angle_input, geo_input])
            # Construct hidden layers (Dense and Batch Normalization).
            for intermediate_dim in reversed(self._intermediate_dims):
                x = Dense(units=intermediate_dim, activation=self._activation,
                          kernel_initializer=self._kernel_initializer,
                          bias_initializer=self._bias_initializer)(x)
                x = BatchNormalization()(x)
            # Add Dense layer to get output which shape is compatible in an input's shape.
            decoder_outputs = Dense(units=self._original_dim, activation=self._out_activation)(x)
            # Create model.
            decoder = Model(inputs=[latent_input, e_input, angle_input, geo_input], outputs=decoder_outputs,
                            name="decoder")
        return decoder

    def _prepare_input_layers(self, for_encoder: bool):
        e_input = Input(shape=(1,))
        angle_input = Input(shape=(1,))
        geo_input = Input(shape=(2,))
        if for_encoder:
            x_input = Input(shape=self._original_dim)
            eps_input = Input(shape=self.latent_dim)
            return [x_input, e_input, angle_input, geo_input, eps_input]
        else:
            x_input = Input(shape=self.latent_dim)
            return [x_input, e_input, angle_input, geo_input]

    def _get_model_specific_data(self, dataset, e_cond, angle_cond, geo_cond):
        noise = np.random.normal(0, 1, size=(dataset.shape[0], self.latent_dim))
        return dataset, e_cond, angle_cond, geo_cond, noise

    def get_decoder(self):
        return self.model.decoder
