import gc
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import tensorflow as tf
import wandb
from sklearn.model_selection import KFold
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, History, Callback
from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization, Input, Dense, Layer, concatenate
from tensorflow.keras.losses import BinaryCrossentropy, Reduction, MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.python.data import Dataset
from tensorflow.python.distribute.distribute_lib import Strategy
from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy
from wandb.keras import WandbCallback
from einops.layers.keras import Rearrange as RearrangeEinops

from core.constants import ORIGINAL_DIM, LATENT_DIM, BATCH_SIZE_PER_REPLICA, EPOCHS, LEARNING_RATE, ACTIVATION, \
    OUT_ACTIVATION, OPTIMIZER_TYPE, KERNEL_INITIALIZER, GLOBAL_CHECKPOINT_DIR, EARLY_STOP, BIAS_INITIALIZER, \
    INTERMEDIATE_DIMS, SAVE_MODEL_EVERY_EPOCH, SAVE_BEST_MODEL, PATIENCE, MIN_DELTA, BEST_MODEL_FILENAME, \
    NUMBER_OF_K_FOLD_SPLITS, VALIDATION_SPLIT, WANDB_ENTITY, N_CELLS_R, N_CELLS_PHI, N_CELLS_Z, INLCUDE_PHYSICS_LOSS, \
    NUM_LAYERS, NUM_HEADS, PROJECTION_DIM, FF_DIMS, MASKING_PERCENT, MASK_AFTER_EMBEDDING, PATCH_R, PATCH_P, PATCH_Z
from utils.optimizer import OptimizerFactory, OptimizerType


class _Sampling(Layer):
    """ Custom layer to do the reparameterization trick: sample random latent vectors z from the latent Gaussian
    distribution.

    The sampled vector z is given by sampled_z = mean + std * epsilon
    """

    def __call__(self, inputs, **kwargs):
        z_mean, z_log_var, epsilon = inputs
        z_sigma = K.exp(0.5 * z_log_var)
        return z_mean + z_sigma * epsilon


# KL divergence computation
class _KLDivergenceLayer(Layer):

    def call(self, inputs, **kwargs):
        mu, log_var = inputs
        kl_loss = -0.5 * (1 + log_var - K.square(mu) - K.exp(log_var))
        kl_loss = K.mean(K.sum(kl_loss, axis=-1))
        self.add_loss(kl_loss)
        return inputs


# Physics observables
def _PhysicsLosses(y_true, y_pred):
    y_true = tf.reshape(y_true, (-1, N_CELLS_R, N_CELLS_PHI, N_CELLS_Z))
    y_pred = tf.reshape(y_pred, (-1, N_CELLS_R, N_CELLS_PHI, N_CELLS_Z))
    # longitudinal profile
    loss = MeanSquaredError(reduction=Reduction.SUM)(tf.reduce_sum(y_true, axis=(0, 1, 2)), tf.reduce_sum(y_pred, axis=(0, 1, 2))) / (BATCH_SIZE_PER_REPLICA * N_CELLS_R * N_CELLS_PHI)
    # lateral profile
    loss += MeanSquaredError(reduction=Reduction.SUM)(tf.reduce_sum(y_true, axis=(0, 2, 3)), tf.reduce_sum(y_pred, axis=(0, 2, 3))) / (BATCH_SIZE_PER_REPLICA * N_CELLS_Z * N_CELLS_PHI)
    return loss


def _Loss(y_true, y_pred):
    reconstruction_loss = BinaryCrossentropy(reduction=Reduction.SUM)(y_true, y_pred)
    loss = reconstruction_loss
    if INLCUDE_PHYSICS_LOSS:
        loss += _PhysicsLosses(y_true, y_pred)
    return loss


class VAE(Model):
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
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self._set_inputs(inputs=self.encoder.inputs, outputs=self(self.encoder.inputs))


@dataclass
class VAEHandler:
    """
    Class to handle building and training VAE models.
    """
    _wandb_project_name: str = None
    _wandb_tags: List[str] = field(default_factory=list)
    _original_dim: int = ORIGINAL_DIM
    latent_dim: int = LATENT_DIM
    _batch_size_per_replica: int = BATCH_SIZE_PER_REPLICA
    _intermediate_dims: List[int] = field(default_factory=lambda: INTERMEDIATE_DIMS)
    _learning_rate: float = LEARNING_RATE
    _epochs: int = EPOCHS
    _activation: str = ACTIVATION
    _out_activation: str = OUT_ACTIVATION
    _number_of_k_fold_splits: float = NUMBER_OF_K_FOLD_SPLITS
    _optimizer_type: OptimizerType = OPTIMIZER_TYPE
    _kernel_initializer: str = KERNEL_INITIALIZER
    _bias_initializer: str = BIAS_INITIALIZER
    _checkpoint_dir: str = GLOBAL_CHECKPOINT_DIR
    _early_stop: bool = EARLY_STOP
    _save_model_every_epoch: bool = SAVE_MODEL_EVERY_EPOCH
    _save_best_model: bool = SAVE_BEST_MODEL
    _patience: int = PATIENCE
    _min_delta: float = MIN_DELTA
    _best_model_filename: str = BEST_MODEL_FILENAME
    _validation_split: float = VALIDATION_SPLIT
    _strategy: Strategy = MirroredStrategy()

    def __post_init__(self) -> None:
        # Calculate true batch size.
        self._batch_size = self._batch_size_per_replica * self._strategy.num_replicas_in_sync
        self._build_and_compile_new_model()
        # Setup Wandb.
        if self._wandb_project_name is not None:
            self._setup_wandb()

    def _setup_wandb(self) -> None:
        config = {
            "learning_rate": self._learning_rate,
            "batch_size": self._batch_size,
            "epochs": self._epochs,
            "optimizer_type": self._optimizer_type,
            "intermediate_dims": self._intermediate_dims,
            "latent_dim": self.latent_dim
        }
        # Reinit flag is needed for hyperparameter tuning. Whenever new training is started, new Wandb run should be
        # created.
        wandb.init(project=self._wandb_project_name, entity=WANDB_ENTITY, reinit=True, config=config,
                   tags=self._wandb_tags)

    def _build_and_compile_new_model(self) -> None:
        """ Builds and compiles a new model.

        VAEHandler keep a list of VAE instance. The reason is that while k-fold cross validation is performed,
        each fold requires a new, clear instance of model. New model is always added at the end of the list of
        existing ones.

        Returns: None

        """
        # Build encoder and decoder.
        encoder = self._build_encoder()
        decoder = self._build_decoder()

        # Compile model within a distributed strategy.
        with self._strategy.scope():
            # Build VAE.
            self.model = VAE(encoder, decoder)
            # Manufacture an optimizer and compile model with.
            optimizer = OptimizerFactory.create_optimizer(self._optimizer_type, self._learning_rate)
            self.model.compile(optimizer=optimizer, loss=_Loss)

    def _prepare_input_layers(self, for_encoder: bool) -> List[Input]:
        """
        Create four Input layers. Each of them is responsible to take respectively: batch of showers/batch of latent
        vectors, batch of energies, batch of angles, batch of geometries.

        Args:
            for_encoder: Boolean which decides whether an input is full dimensional shower or a latent vector.

        Returns:
            List of Input layers (five for encoder and four for decoder).

        """
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

    def _manufacture_callbacks(self) -> List[Callback]:
        """
        Based on parameters set by the user, manufacture callbacks required for training.

        Returns:
            A list of `Callback` objects.

        """
        callbacks = []
        # If the early stopping flag is on then stop the training when a monitored metric (validation) has stopped
        # improving after (patience) number of epochs.
        if self._early_stop:
            callbacks.append(
                EarlyStopping(monitor="val_loss",
                              min_delta=self._min_delta,
                              patience=self._patience,
                              verbose=True,
                              restore_best_weights=True))
        # Save model after every epoch.
        if self._save_model_every_epoch:
            callbacks.append(ModelCheckpoint(filepath=f"{self._checkpoint_dir}/VAE_epoch_{{epoch:03}}/model_weights",
                                             monitor="val_loss",
                                             verbose=True,
                                             save_weights_only=True,
                                             mode="min",
                                             save_freq="epoch"))
        if self._save_best_model:
            callbacks.append(ModelCheckpoint(filepath=f"{self._checkpoint_dir}/VAE_best/model_weights",
                                             monitor="val_loss",
                                             verbose=True,
                                             save_best_only=True,
                                             save_weights_only=True,
                                             mode="min",
                                             save_freq="epoch"))
        # Pass metadata to wandb.
        callbacks.append(WandbCallback(
            monitor="val_loss", verbose=0, mode="auto", save_model=False))
        return callbacks

    def _get_train_and_val_data(self, dataset: np.array, e_cond: np.array, angle_cond: np.array, geo_cond: np.array,
                                noise: np.array, train_indexes: np.array, validation_indexes: np.array) \
            -> Tuple[Dataset, Dataset]:
        """
        Splits data into train and validation set based on given lists of indexes.

        """

        # Prepare training data.
        train_dataset = dataset[train_indexes, :]
        train_e_cond = e_cond[train_indexes]
        train_angle_cond = angle_cond[train_indexes]
        train_geo_cond = geo_cond[train_indexes, :]
        train_noise = noise[train_indexes, :]

        # Prepare validation data.
        val_dataset = dataset[validation_indexes, :]
        val_e_cond = e_cond[validation_indexes]
        val_angle_cond = angle_cond[validation_indexes]
        val_geo_cond = geo_cond[validation_indexes, :]
        val_noise = noise[validation_indexes, :]

        # Gather them into tuples.
        train_x = (train_dataset, train_e_cond, train_angle_cond, train_geo_cond, train_noise)
        train_y = train_dataset
        val_x = (val_dataset, val_e_cond, val_angle_cond, val_geo_cond, val_noise)
        val_y = val_dataset

        # Wrap data in Dataset objects.
        # TODO(@mdragula): This approach requires loading the whole data set to RAM. It
        #  would be better to read the data partially when needed. Also one should bare in mind that using tf.Dataset
        #  slows down training process.
        train_data = Dataset.from_tensor_slices((train_x, train_y))
        val_data = Dataset.from_tensor_slices((val_x, val_y))

        # The batch size must now be set on the Dataset objects.
        train_data = train_data.batch(self._batch_size)
        val_data = val_data.batch(self._batch_size)

        # Disable AutoShard.
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        train_data = train_data.with_options(options)
        val_data = val_data.with_options(options)

        return train_data, val_data

    def _k_fold_training(self, dataset: np.array, e_cond: np.array, angle_cond: np.array, geo_cond: np.array,
                         noise: np.array, callbacks: List[Callback], verbose: bool = True) -> List[History]:
        """
        Performs K-fold cross validation training.

        Number of fold is defined by (self._number_of_k_fold_splits). Always shuffle the dataset.

        Args:
            dataset: A matrix representing showers. Shape =
                (number of samples, ORIGINAL_DIM = N_CELLS_Z * N_CELLS_R * N_CELLS_PHI).
            e_cond: A matrix representing an energy for each sample. Shape = (number of samples, ).
            angle_cond: A matrix representing an angle for each sample. Shape = (number of samples, ).
            geo_cond: A matrix representing a geometry of the detector for each sample. Shape = (number of samples, 2).
            noise: A matrix representing an additional noise needed to perform a reparametrization trick.
            callbacks: A list of callback forwarded to the fitting function.
            verbose: A boolean which says there the training should be performed in a verbose mode or not.

        Returns: A list of `History` objects.`History.history` attribute is a record of training loss values and
        metrics values at successive epochs, as well as validation loss values and validation metrics values (if
        applicable).

        """
        # TODO(@mdragula): KFold cross validation can be parallelized. Each fold is independent from each the others.
        k_fold = KFold(n_splits=self._number_of_k_fold_splits, shuffle=True)
        histories = []

        for i, (train_indexes, validation_indexes) in enumerate(k_fold.split(dataset)):
            print(f"K-fold: {i + 1}/{self._number_of_k_fold_splits}...")
            train_data, val_data = self._get_train_and_val_data(dataset, e_cond, angle_cond, geo_cond, noise,
                                                                train_indexes, validation_indexes)

            self._build_and_compile_new_model()

            history = self.model.fit(x=train_data,
                                     shuffle=True,
                                     epochs=self._epochs,
                                     verbose=verbose,
                                     validation_data=val_data,
                                     callbacks=callbacks
                                     )
            histories.append(history)

            # Remove all unnecessary data from previous fold.
            del self.model
            del train_data
            del val_data
            tf.keras.backend.clear_session()
            gc.collect()

        return histories

    def _single_training(self, dataset: np.array, e_cond: np.array, angle_cond: np.array, geo_cond: np.array,
                         noise: np.ndarray, callbacks: List[Callback], verbose: bool = True) -> List[History]:
        """
        Performs a single training.

        A fraction of dataset (self._validation_split) is used as a validation data.

        Args:
            dataset: A matrix representing showers. Shape =
                (number of samples, ORIGINAL_DIM = N_CELLS_Z * N_CELLS_R * N_CELLS_PHI).
            e_cond: A matrix representing an energy for each sample. Shape = (number of samples, ).
            angle_cond: A matrix representing an angle for each sample. Shape = (number of samples, ).
            geo_cond: A matrix representing a geometry of the detector for each sample. Shape = (number of samples, 2).
            noise: A matrix representing an additional noise needed to perform a reparametrization trick.
            callbacks: A list of callback forwarded to the fitting function.
            verbose: A boolean which says there the training should be performed in a verbose mode or not.

        Returns: A one-element list of `History` objects.`History.history` attribute is a record of training loss
        values and metrics values at successive epochs, as well as validation loss values and validation metrics
        values (if applicable).

        """
        dataset_size, _ = dataset.shape
        permutation = np.random.permutation(dataset_size)
        split = int(dataset_size * self._validation_split)
        train_indexes, validation_indexes = permutation[split:], permutation[:split]

        train_data, val_data = self._get_train_and_val_data(dataset, e_cond, angle_cond, geo_cond, noise, train_indexes,
                                                            validation_indexes)

        history = self.model.fit(x=train_data,
                                 shuffle=True,
                                 epochs=self._epochs,
                                 verbose=verbose,
                                 validation_data=val_data,
                                 callbacks=callbacks
                                 )

        return [history]

    def train(self, dataset: np.array, e_cond: np.array, angle_cond: np.array, geo_cond: np.array,
              verbose: bool = True) -> List[History]:
        """
        For a given input data trains and validates the model.

        If the numer of K-fold splits > 1 then it runs K-fold cross validation, otherwise it runs a single training
        which uses (self._validation_split * 100) % of dataset as a validation data.

        Args:
            dataset: A matrix representing showers. Shape =
                (number of samples, ORIGINAL_DIM = N_CELLS_Z * N_CELLS_R * N_CELLS_PHI).
            e_cond: A matrix representing an energy for each sample. Shape = (number of samples, ).
            angle_cond: A matrix representing an angle for each sample. Shape = (number of samples, ).
            geo_cond: A matrix representing a geometry of the detector for each sample. Shape = (number of samples, 2).
            verbose: A boolean which says there the training should be performed in a verbose mode or not.

        Returns: A list of `History` objects.`History.history` attribute is a record of training loss values and
        metrics values at successive epochs, as well as validation loss values and validation metrics values (if
        applicable).

        """

        callbacks = self._manufacture_callbacks()

        noise = np.random.normal(0, 1, size=(dataset.shape[0], self.latent_dim))

        if self._number_of_k_fold_splits > 1:
            return self._k_fold_training(dataset, e_cond, angle_cond, geo_cond, noise, callbacks, verbose)
        else:
            return self._single_training(dataset, e_cond, angle_cond, geo_cond, noise, callbacks, verbose)


def TransformerEncoderBlock(inputs, num_heads, projection_dim, ff_dims, dropout=0.1):
    # Layer normalization 1.
    x1 = layers.LayerNormalization(epsilon=1e-6)(inputs)
    # Create a multi-head attention layer.
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=projection_dim, dropout=dropout)(x1, x1)
    # Skip connection 1.
    x2 = layers.Add()([attention_output, inputs])
    # Layer normalization 2.
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    # MLP.
    x3 = mlp(x3, hidden_units=ff_dims, dropout_rate=dropout)
    # Skip connection 2.
    return layers.Add()([x3, x2])


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class PatchEncoder(Layer):
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


class TransformerV1(VAEHandler):
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
            patches_combined = concatenate([patchified, _e_input, _angle_input, _geo_input], axis=-2)

            # Linear projection and positional embeddings
            encoded_patches = PatchEncoder(num_patches + 4, 256)(patches_combined)

            # Transformer Encoder
            x = TransformerEncoderBlock(encoded_patches, 4, 256, [256,])
            x = Dense(64)(x)
            x = TransformerEncoderBlock(x, 8, 64, [64,])
            x = Dense(16)(x)
            x = TransformerEncoderBlock(x, 16, 16, [16,])

            # Handling transformer representations
            representation = layers.LayerNormalization(epsilon=1e-6)(x)
            representation = layers.Flatten()(representation)
            representation = layers.Dropout(0.2)(representation)

            # Add Dense layer to get description of multidimensional Gaussian distribution in terms of mean
            # and log(variance).
            z_mean = Dense(self.latent_dim, name="z_mean")(representation)
            z_log_var = Dense(self.latent_dim, name="z_log_var")(representation)
            # Add KLDivergenceLayer responsible for calculation of KL loss.
            z_mean, z_log_var = _KLDivergenceLayer()([z_mean, z_log_var])
            # Sample a probe from the distribution.
            encoder_output = _Sampling()([z_mean, z_log_var, eps_input])
            # Create model.
            encoder = Model(inputs=[x_input, e_input, angle_input, geo_input, eps_input], outputs=encoder_output,
                            name="encoder")
        return encoder


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


class TransformerV2(VAEHandler):
    def __post_init__(self):
        self._num_layers = NUM_LAYERS
        self._num_heads = NUM_HEADS
        self._projection_dim = PROJECTION_DIM
        self._ff_dims = FF_DIMS
        super().__post_init__()

    def _build_transformer(self) -> Model:
        # with self._strategy.scope(): TODO
        # Prepare input layer.
        x_input, e_input, angle_input, geo_input, eps_input = self._prepare_input_layers(for_encoder=True)
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
        x = Dense(feature_dim, activation=tf.nn.gelu)(x)
        x = Dense(feature_dim, activation='sigmoid')(x)
        x = RearrangeEinops("b (i j k) (r p z) -> b (i r) (j p) (k z)",
            i=PATCH_R, j=PATCH_P, k=PATCH_Z, r=N_CELLS_R//PATCH_R, p=N_CELLS_PHI//PATCH_P, z=N_CELLS_Z//PATCH_Z)(x)
        out = layers.Reshape((-1,))(x)

        transformer = Model(inputs=[x_input, e_input, angle_input, geo_input, eps_input], outputs=out,
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
