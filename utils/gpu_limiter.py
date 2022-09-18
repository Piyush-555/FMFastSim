import os
from dataclasses import dataclass

import torch


@dataclass
class GPULimiter:
    """
    Class responsible to set the limits of possible GPU usage by PyTorch. There is no option as of now to setup
    virtual device in PyTorch. So, keeping 1 virtual device per 1 physical device.

    Attributes:
        _gpu_ids: A string representing visible devices for the process. Identifiers of physical GPUs should
            be separated by commas (no spaces).
        _max_gpu_memory_allocation: An integer specifying limit of allocated memory per logical device.

    """
    _gpu_ids: str
    _max_gpu_memory_allocation: int

    def __call__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{self._gpu_ids}"
        gpus = torch.cuda.device_count()
        if gpus:
            # Restrict PyTorch to only allocate the required fraction for each GPU
            try:
                for gpu_id in range(gpus):
                    total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)  # In GB
                    fraction = self._max_gpu_memory_allocation / total_memory
                    torch.cuda.set_per_process_memory_fraction(fraction, gpu_id)
                return torch.device('cuda:{}'.format(gpu_id))  # TODO
            except RuntimeError as e:
                # Devices must be set before GPUs have been initialized
                print(e)
