"""
TODO: Make changes wrt to PyTorch
** generate **
generate showers using a saved VAE model 
"""
import argparse

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from core.constants import GLOBAL_CHECKPOINT_DIR, GEN_DIR, BATCH_SIZE_PER_REPLICA, MAX_GPU_MEMORY_ALLOCATION, GPU_IDS
from utils.gpu_limiter import GPULimiter
from utils.preprocess import get_condition_arrays


def parse_args():
    argument_parser = argparse.ArgumentParser()
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


# main function
def main():
    # 0. Parse arguments.
    args = parse_args()
    energy = args.energy
    angle = args.angle
    geometry = args.geometry
    events = args.events
    epoch = args.epoch
    study_name = args.study_name
    max_gpu_memory_allocation = args.max_gpu_memory_allocation
    gpu_ids = args.gpu_ids

    # 1. Set GPU memory limits.
    device = GPULimiter(_gpu_ids=gpu_ids, _max_gpu_memory_allocation=max_gpu_memory_allocation)()

    # 2. Load a saved model.

    # Create a handler and build model.
    # This import must be local because otherwise it is impossible to call GPULimiter.
    from core.model import VAEHandler
    vae = VAEHandler()

    # Load the saved weights
    weights_dir = f"VAE_epoch_{epoch:03}" if epoch is not None else "VAE_best"
    state_dict = torch.load(f"{GLOBAL_CHECKPOINT_DIR}/{study_name}/{weights_dir}/model_weights.pt")
    vae.model.load_state_dict(state_dict)

    # The generator is defined as the decoder part only
    generator = vae.model.decoder
    generator.eval()
    generator.to(device)

    # 3. Prepare data. Get condition values. Sample from the prior (normal distribution) in d dimension (d=latent_dim,
    # latent space dimension). Gather them into tuples. Wrap data in Dataset objects. The batch size must now be set
    # on the Dataset objects. Disable AutoShard.
    e_cond, angle_cond, geo_cond = get_condition_arrays(geometry, energy, angle, events)
    e_cond = e_cond.reshape(-1, 1)
    angle_cond = angle_cond.reshape(-1, 1)

    z_r = np.random.normal(loc=0, scale=1, size=(events, vae.latent_dim))
 
    data = []
    for d in [z_r, e_cond, angle_cond, geo_cond]:
        data.append(torch.from_numpy(d.astype(np.float32)))
    
    dataset = TensorDataset(*data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE_PER_REPLICA)

    # 4. Generate showers using the VAE model.
    generated_events = []
    for inputs in dataloader:
        inputs = [i.to(device) for i in inputs]
        y = generator(inputs)
        # import pdb;pdb.set_trace()
        generated_events.append(y.detach().cpu().numpy())

    generated_events = np.concatenate(generated_events, axis=0)

    # 5. Save the generated showers.
    np.save(f"{GEN_DIR}/VAE_Generated_Geo_{geometry}_E_{energy}_Angle_{angle}.npy", generated_events)


if __name__ == "__main__":
    exit(main())
