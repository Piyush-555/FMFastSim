import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

from core.constants import BATCH_SIZE_PER_REPLICA, N_CELLS_R, N_CELLS_PHI, N_CELLS_Z


class _Sampling(layers.Layer):
    """ Custom layer to do the reparameterization trick: sample random latent vectors z from the latent Gaussian
    distribution.

    The sampled vector z is given by sampled_z = mean + std * epsilon
    """

    def __call__(self, inputs, **kwargs):
        z_mean, z_log_var, epsilon = inputs
        z_sigma = K.exp(0.5 * z_log_var)
        return z_mean + z_sigma * epsilon


# KL divergence computation
class _KLDivergenceLayer(layers.Layer):

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

"""
Miscellaneous
"""
class DummyArgParse:
    def __init__(self):
        pass

    def __setarr__(self, key, value):
        self.key = value

    def __getarr__(self, key):
        return self.key
