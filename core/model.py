# TODO: Make changes wrt to PyTorch

import os
import gc
from dataclasses import dataclass, field
from typing import List, Tuple

import wandb
import numpy as np
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from core.constants import ORIGINAL_DIM, LATENT_DIM, BATCH_SIZE_PER_REPLICA, EPOCHS, LEARNING_RATE, ACTIVATION, \
    OUT_ACTIVATION, OPTIMIZER_TYPE, KERNEL_INITIALIZER, GLOBAL_CHECKPOINT_DIR, EARLY_STOP, BIAS_INITIALIZER, \
    INTERMEDIATE_DIMS, SAVE_MODEL_EVERY_EPOCH, SAVE_BEST_MODEL, PATIENCE, MIN_DELTA, BEST_MODEL_FILENAME, \
    NUMBER_OF_K_FOLD_SPLITS, VALIDATION_SPLIT, WANDB_ENTITY, N_CELLS_R, N_CELLS_PHI, N_CELLS_Z, INLCUDE_PHYSICS_LOSS, KL_WEIGHT
from utils.optimizer import OptimizerFactory, OptimizerType


def _Sampling(z_mean, z_log_var, epsilon):
    """ Custom layer to do the reparameterization trick: sample random latent vectors z from the latent Gaussian
    distribution.

    The sampled vector z is given by sampled_z = mean + std * epsilon
    """
    z_sigma = torch.exp(0.5 * z_log_var)
    return z_mean + z_sigma * epsilon


# KL divergence computation
def _KLDivergence(mu, log_var, **kwargs):
    kl_loss = -0.5 * (1 + log_var - torch.square(mu) - torch.exp(log_var))
    kl_loss = kl_loss.sum(dim=-1).mean()
    # print(kl_loss)
    return kl_loss


# Physics observables
def _PhysicsLosses(y_true, y_pred):
    y_true = torch.reshape(y_true, (-1, N_CELLS_R, N_CELLS_PHI, N_CELLS_Z))
    y_pred = torch.reshape(y_pred, (-1, N_CELLS_R, N_CELLS_PHI, N_CELLS_Z))
    # longitudinal profile
    loss = nn.MSELoss(reduction='sum')(torch.sum(y_pred, dim=(0, 1, 2)), torch.sum(y_true, dim=(0, 1, 2))) / (BATCH_SIZE_PER_REPLICA * N_CELLS_R * N_CELLS_PHI)
    # lateral profile
    loss += nn.MSELoss(reduction='sum')(torch.sum(y_pred, dim=(0, 2, 3)), torch.sum(y_true, dim=(0, 2, 3))) / (BATCH_SIZE_PER_REPLICA * N_CELLS_Z * N_CELLS_PHI)
    return loss


def _Loss(model, y_true, y_pred):
    # import pdb;pdb.set_trace()
    loss = nn.BCELoss(reduction='sum')(y_pred, y_true)
    # print(loss)
    loss += KL_WEIGHT * model.get_KL()
    if INLCUDE_PHYSICS_LOSS:
        loss += _PhysicsLosses(y_true, y_pred)
    return loss


class Encoder(nn.Module):
    def __init__(self, original_dim, intermediate_dims, latent_dim, activation, kernel_initializer, bias_initializer):
        super().__init__()
        all_dims = [original_dim + 4,] + intermediate_dims
        self.blocks = nn.Sequential(
            *[self.block(all_dims[i], all_dims[i+1], activation) for i in range(len(all_dims) - 1)]
        )
        self.mu_layer = nn.Linear(all_dims[-1], latent_dim)
        self.log_var_layer = nn.Linear(all_dims[-1], latent_dim)

    def block(self, in_dims, out_dims, activation):
        return nn.Sequential(
            nn.Linear(in_dims, out_dims),
            activation(),
            nn.BatchNorm1d(out_dims)
        )
    
    def forward(self, inputs):
        x = torch.concat(inputs, dim=1)
        for block in self.blocks:
            x = block(x)
        return self.mu_layer(x), self.log_var_layer(x)


class Decoder(nn.Module):
    def __init__(self, original_dim, intermediate_dims, latent_dim, activation, out_activation, kernel_initializer, bias_initializer):
        super().__init__()
        all_dims = [latent_dim + 4,] + intermediate_dims[::-1]
        self.blocks = nn.Sequential(
            *[self.block(all_dims[i], all_dims[i+1], activation) for i in range(len(all_dims) - 1)]
        )
        self.output_layer = nn.Sequential(
            nn.Linear(all_dims[-1], original_dim),
            out_activation()
        )

    def block(self, in_dims, out_dims, activation):
        return nn.Sequential(
            nn.Linear(in_dims, out_dims),
            activation(),
            nn.BatchNorm1d(out_dims)
        )
    
    def forward(self, inputs):
        x = torch.concat(inputs, dim=1)
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)


class VAE(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs):
        x, e_input, angle_input, geo_input, eps = inputs
        # import pdb; pdb.set_trace()
        self.mu, self.log_var = self.encoder([x, e_input, angle_input, geo_input])
        z = _Sampling(self.mu, self.log_var, eps)
        return self.decoder([z, e_input, angle_input, geo_input])
    
    def get_KL(self):
        return _KLDivergence(self.mu, self.log_var)


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
    _strategy = None  # TODO

    def __post_init__(self) -> None:
        # Calculate true batch size.
        self._batch_size = self._batch_size_per_replica  # * self._strategy.num_replicas_in_sync TODO
        self._build_model()
        # Setup Wandb.
        if self._wandb_project_name is not None:
            pass #self._setup_wandb()

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

    def _build_model(self) -> None:
        """ Builds and compiles a new model.

        VAEHandler keep a list of VAE instance. The reason is that while k-fold cross validation is performed,
        each fold requires a new, clear instance of model. New model is always added at the end of the list of
        existing ones.

        Returns: None

        """
        # Build encoder and decoder.
        encoder = Encoder(self._original_dim, self._intermediate_dims, self.latent_dim,
            self._activation, self._kernel_initializer, self._bias_initializer)
        decoder = Decoder(self._original_dim, self._intermediate_dims, self.latent_dim,
            self._activation, self._out_activation, self._kernel_initializer, self._bias_initializer)

        # Build VAE.
        self.model = VAE(encoder, decoder)
        # Manufacture an optimizer and compile model with.
        self.optimizer = OptimizerFactory.create_optimizer(self.model.parameters(), self._optimizer_type, self._learning_rate)
        self.loss_fn = _Loss

    def _fit(self, epochs, trainloader, validloader, device, verbose=True):
        min_val_loss = np.inf
        self.model.to(device)
        history = []
        
        for epoch in range(epochs):

            # Training
            self.model.train()
            training_loss = 0
            for inputs in trainloader:
                inputs = [i.to(device) for i in inputs]
                x = inputs[0]
                self.optimizer.zero_grad()
                y = self.model(inputs)
                loss = self.loss_fn(self.model, x, y)
                loss.backward()
                self.optimizer.step()
                training_loss += loss.item()
            
            # Validation
            self.model.eval()
            valid_loss = 0
            for inputs in validloader:
                inputs = [i.to(device) for i in inputs]
                x = inputs[0]
                y = self.model(inputs)
                loss = self.loss_fn(self.model, x, y)
                valid_loss += loss.item()
            
            training_loss /= len(trainloader)
            valid_loss /= len(validloader)

            if min_val_loss > valid_loss:
                min_val_loss = valid_loss
                if self._save_best_model:
                    os.makedirs(f"{self._checkpoint_dir}/VAE_best/", exist_ok=True)
                    torch.save(self.model.state_dict(), f"{self._checkpoint_dir}/VAE_best/model_weights.pt")

            if self._save_model_every_epoch:
                os.makedirs(f"{self._checkpoint_dir}/VAE_epoch_{{epoch:03}}/", exist_ok=True)
                torch.save(self.model.state_dict(), f"{self._checkpoint_dir}/VAE_epoch_{{epoch:03}}/model_weights.pt")
            
            if verbose:
                print("Epoch: {} \tTrainLoss: {} \tValidLoss: {}".format(epoch + 1, training_loss, valid_loss))
            
            history.append([epoch, training_loss, valid_loss])
        return history

    def _get_train_and_val_data(self, dataset: np.array, e_cond: np.array, angle_cond: np.array, geo_cond: np.array,
                                noise: np.array, train_indexes: np.array, validation_indexes: np.array) \
            -> Tuple[DataLoader, DataLoader]:
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
        train_data = [
            train_dataset.astype(np.float32),
            train_e_cond.astype(np.float32).reshape(-1, 1),
            train_angle_cond.astype(np.float32).reshape(-1, 1),
            train_geo_cond,
            train_noise.astype(np.float32)
            ]
        val_data = [
            val_dataset.astype(np.float32),
            val_e_cond.astype(np.float32).reshape(-1, 1),
            val_angle_cond.astype(np.float32).reshape(-1, 1),
            val_geo_cond,
            val_noise.astype(np.float32)]

        trainset = TensorDataset(*[torch.from_numpy(i) for i in train_data])
        validset = TensorDataset(*[torch.from_numpy(i) for i in val_data])

        trainloader = DataLoader(trainset, batch_size=self._batch_size)
        validloader = DataLoader(validset, batch_size=self._batch_size)
        return trainloader, validloader

    def _k_fold_training(self, dataset: np.array, e_cond: np.array, angle_cond: np.array, geo_cond: np.array,
                         noise: np.array, device, verbose: bool = True) -> List[List[float]]:
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
            trainloader, validloader = self._get_train_and_val_data(dataset, e_cond, angle_cond, geo_cond, noise,
                                                                train_indexes, validation_indexes)

            self._build_model()

            history = self._fit(self._epochs, trainloader, validloader, device, verbose)
            histories.append(history)

            if self._save_best_model:
                self.model.save_weights(f"{self._checkpoint_dir}/VAE_fold_{i + 1}/model_weights")
                print(f"Best model from fold {i + 1} was saved.")

            # Remove all unnecessary data from previous fold.
            del self.model
            del train_data
            del val_data
            tf.keras.backend.clear_session()
            gc.collect()

        return histories

    def _single_training(self, dataset: np.array, e_cond: np.array, angle_cond: np.array, geo_cond: np.array,
                         noise: np.ndarray, device, verbose: bool = True) -> List[List[float]]:
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

        trainloader, validloader = self._get_train_and_val_data(dataset, e_cond, angle_cond, geo_cond, noise, train_indexes,
                                                            validation_indexes)

        history = self._fit(self._epochs, trainloader, validloader, device, verbose)

        return [history]

    def train(self, dataset: np.array, e_cond: np.array, angle_cond: np.array, geo_cond: np.array,
              device, verbose: bool = True) -> List[List[float]]:
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

        noise = np.random.normal(0, 1, size=(dataset.shape[0], self.latent_dim))

        if self._number_of_k_fold_splits > 1:
            return self._k_fold_training(dataset, e_cond, angle_cond, geo_cond, noise, device, verbose)
        else:
            return self._single_training(dataset, e_cond, angle_cond, geo_cond, noise, device, verbose)
