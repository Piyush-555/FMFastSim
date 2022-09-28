"""
** generate **
generate showers using a saved VAE model 
"""
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.python.data import Dataset

from core.constants import GLOBAL_CHECKPOINT_DIR, GEN_DIR, BATCH_SIZE_PER_REPLICA, MAX_GPU_MEMORY_ALLOCATION, GPU_IDS, INIT_DIR, MODEL_TYPE
from utils.gpu_limiter import GPULimiter
from utils.preprocess import get_condition_arrays, load_showers


def parse_args():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--model-type", type=str, default=MODEL_TYPE)
    argument_parser.add_argument("--geometry", type=str, default="")
    argument_parser.add_argument("--energy", type=int, default="")
    argument_parser.add_argument("--angle", type=int, default="")
    argument_parser.add_argument("--events", type=int, default=10000)
    argument_parser.add_argument("--epoch", type=int, default=None)
    argument_parser.add_argument("--study-name", type=str, default="default_study_name")
    argument_parser.add_argument("--max-gpu-memory-allocation", type=int, default=MAX_GPU_MEMORY_ALLOCATION)
    argument_parser.add_argument("--gpu-ids", type=str, default=GPU_IDS)
    args = argument_parser.parse_args()
    return args


def generate(generator, energy, angle, geometry, num_events, latent_dim=None, e_layer_g4=None):
    # 3. Prepare data. Get condition values. Sample from the prior (normal distribution) in d dimension (d=latent_dim,
    # latent space dimension). Gather them into tuples. Wrap data in Dataset objects. The batch size must now be set
    # on the Dataset objects. Disable AutoShard.
    if latent_dim:
        e_cond, angle_cond, geo_cond = get_condition_arrays(geometry, energy, angle, num_events)
        z_r = np.random.normal(loc=0, scale=1, size=(num_events, latent_dim))
        data = ((z_r, e_cond, angle_cond, geo_cond),)
    else:
        e_layer_g4 = e_layer_g4.reshape(e_layer_g4.shape[0], -1)
        data = ((e_layer_g4,),)

    data = Dataset.from_tensor_slices(data)

    batch_size = BATCH_SIZE_PER_REPLICA

    data = data.batch(batch_size)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    data = data.with_options(options)

    # 4. Generate showers using the VAE model.
    generated_events = generator.predict(data) * (energy * 1000)
    return generated_events


# main function
def main():
    # 0. Parse arguments.
    args = parse_args()
    model_type = args.model_type
    energy = args.energy
    angle = args.angle
    geometry = args.geometry
    events = args.events
    epoch = args.epoch
    study_name = args.study_name
    max_gpu_memory_allocation = args.max_gpu_memory_allocation
    gpu_ids = args.gpu_ids

    # 1. Set GPU memory limits.
    GPULimiter(_gpu_ids=gpu_ids, _max_gpu_memory_allocation=max_gpu_memory_allocation)()

    # 2. Load a saved model.

    # Create a handler and build model.
    # This import must be local because otherwise it is impossible to call GPULimiter.
    from core.models import ResolveModel
    model_handler = ResolveModel(model_type)()

    # Load the saved weights
    weights_dir = f"Epoch_{epoch:03}" if epoch is not None else "Best"
    model_handler.model.load_weights(f"{GLOBAL_CHECKPOINT_DIR}/{study_name}/{weights_dir}/model_weights").expect_partial()

    # The generator is defined as the decoder part only
    generator = model_handler.get_decoder()

    latent_dim = getattr(model_handler, 'latent_dim', None)
    e_layer_g4 = None
    if not latent_dim:
        e_layer_g4 = load_showers(INIT_DIR, geometry, energy, angle)

    generated_events = generate(generator, energy, angle, geometry, events, latent_dim, e_layer_g4)

    # 5. Save the generated showers.
    np.save(f"{GEN_DIR}/Geo_{geometry}_E_{energy}_Angle_{angle}.npy", generated_events)


if __name__ == "__main__":
    exit(main())
