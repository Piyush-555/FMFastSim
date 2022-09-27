import gc
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import tensorflow as tf
import wandb
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, History, Callback
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.python.data import Dataset
from tensorflow.python.distribute.distribute_lib import Strategy
from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy
from wandb.keras import WandbCallback

from core import models
from utils.optimizer import OptimizerType
from core.constants import ORIGINAL_DIM, BATCH_SIZE_PER_REPLICA, EPOCHS, LEARNING_RATE, \
    OPTIMIZER_TYPE, GLOBAL_CHECKPOINT_DIR, EARLY_STOP, SAVE_MODEL_EVERY_EPOCH, SAVE_BEST_MODEL, \
    PATIENCE, MIN_DELTA, BEST_MODEL_FILENAME, NUMBER_OF_K_FOLD_SPLITS, VALIDATION_SPLIT, WANDB_ENTITY


def ResolveModel(model_type):
    if model_type=='VAE':
        return models.VAE
    if model_type=='TransformerVAE':
        return models.TransformerVAE
    if model_type=='TransformerMLM':
        return models.TransformerMLM
    else:
        raise ValueError


@dataclass
class ModelHandler:
    """
    Class to handle building and training of models.
    """
    _wandb_project_name: str = None
    _wandb_tags: List[str] = field(default_factory=list)
    _original_dim: int = ORIGINAL_DIM
    _batch_size_per_replica: int = BATCH_SIZE_PER_REPLICA
    _learning_rate: float = LEARNING_RATE
    _epochs: int = EPOCHS
    _number_of_k_fold_splits: float = NUMBER_OF_K_FOLD_SPLITS
    _optimizer_type: OptimizerType = OPTIMIZER_TYPE
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
            "optimizer_type": self._optimizer_type
        }
        config.update(self._get_wandb_extra_config())
        # Reinit flag is needed for hyperparameter tuning. Whenever new training is started, new Wandb run should be
        # created.
        wandb.init(project=self._wandb_project_name, entity=WANDB_ENTITY, reinit=True, config=config,
                   tags=self._wandb_tags)
    
    def _get_wandb_extra_config(self):
        raise NotImplementedError

    def _build_and_compile_new_model(self) -> None:
        """ Builds and compiles a new model.

        VAEHandler keep a list of VAE instance. The reason is that while k-fold cross validation is performed,
        each fold requires a new, clear instance of model. New model is always added at the end of the list of
        existing ones.

        Returns: None

        """
        raise NotImplementedError

    def _prepare_input_layers(self, for_encoder: bool) -> List[Input]:
        """
        Create four Input layers. Each of them is responsible to take respectively: batch of showers/batch of latent
        vectors, batch of energies, batch of angles, batch of geometries.

        Args:
            for_encoder: Boolean which decides whether an input is full dimensional shower or a latent vector.

        Returns:
            List of Input layers (five for encoder and four for decoder).

        """
        raise NotImplementedError

    def get_decoder(self):
        raise NotImplementedError

    def _get_model_specific_data(self, dataset, e_cond, angle_cond, geo_cond):
        raise NotImplementedError

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
                                train_indexes: np.array, validation_indexes: np.array) -> Tuple[Dataset, Dataset]:
        """
        Splits data into train and validation set based on given lists of indexes.

        """

        # X Get data specific to each model type.
        train_x, val_x = [], []
        raw_data = self._get_model_specific_data(dataset, e_cond, angle_cond, geo_cond)
        for data in raw_data:
            train_x.append(data[train_indexes])
            val_x.append(data[validation_indexes])

        train_y = dataset[train_indexes]
        val_y = dataset[validation_indexes]

        # Convert to tuples.
        train_x, val_x = tuple(train_x), tuple(val_x)

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
                         callbacks: List[Callback], verbose: bool = True) -> List[History]:
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
            train_data, val_data = self._get_train_and_val_data(dataset, e_cond, angle_cond, geo_cond,
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
                         callbacks: List[Callback], verbose: bool = True) -> List[History]:
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

        train_data, val_data = self._get_train_and_val_data(dataset, e_cond, angle_cond, geo_cond, train_indexes,
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

        if self._number_of_k_fold_splits > 1:
            return self._k_fold_training(dataset, e_cond, angle_cond, geo_cond, callbacks, verbose)
        else:
            return self._single_training(dataset, e_cond, angle_cond, geo_cond, callbacks, verbose)
