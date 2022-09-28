from utils.optimizer import OptimizerType

"""
Experiment constants.
"""
# Geometries to train on
GEOMETRIES = ["SiW",]  # ["SiW", "SciPb"]
# Number of calorimeter layers (z-axis segmentation).
N_CELLS_Z = 45
# Segmentation in the r,phi direction.
N_CELLS_R = 18
N_CELLS_PHI = 50
# Cell size in the r and z directions 
SIZE_R = 2.325
SIZE_Z = 3.4

# Minimum and maximum primary particle energy to consider for training in GeV units.
MIN_ENERGY = 64
MAX_ENERGY = 256
# Minimum and maximum primary particle angle to consider for training in degrees units.
MIN_ANGLE = 70
MAX_ANGLE = 70

"""
Directories.
"""
# Directory to load the full simulation dataset.
INIT_DIR = "../eos/geant4/fastSim/Par04_public/HDF5_Zenodo/"
# Directory to save VAE checkpoints
GLOBAL_CHECKPOINT_DIR = "./checkpoint"
# Directory to save model after conversion to a format that can be used in C++.
CONV_DIR = "./conversion"
# Directory to save validation plots.
VALID_DIR = "./validation"
# Directory to save VAE generated showers.
GEN_DIR = "./generation"

"""
Model default parameters.
"""
MODEL_TYPE = 'TransformerVAE'
BATCH_SIZE_PER_REPLICA = 256
INLCUDE_PHYSICS_LOSS = False
# Total number of readout cells (represents the number of nodes in the input/output layers of the model).
ORIGINAL_DIM = N_CELLS_Z * N_CELLS_R * N_CELLS_PHI
EPOCHS = 500
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.10
NUMBER_OF_K_FOLD_SPLITS = 1
OPTIMIZER_TYPE = OptimizerType.ADAM
EARLY_STOP = False
SAVE_BEST_MODEL = True
SAVE_MODEL_EVERY_EPOCH = False
PATIENCE = 10
MIN_DELTA = 0.01
# GPU identifiers separated by comma, no spaces.
GPU_IDS = "0"
# Maximum allowed memory on one of the GPUs (in GB)
MAX_GPU_MEMORY_ALLOCATION = 32
# Buffer size used while shuffling the dataset.
BUFFER_SIZE = 1000

"""
VAE params
"""
INTERMEDIATE_DIMS = [100, 50, 20, 14]
LATENT_DIM = 10  # Also applicable to TransformerVAE
ACTIVATION = "leaky_relu"
OUT_ACTIVATION = "sigmoid"
KERNEL_INITIALIZER = "RandomNormal"
BIAS_INITIALIZER = "Zeros"


"""
Transformer parameters.
"""
NUM_LAYERS = 6
NUM_HEADS = NUM_LAYERS * [8]
PROJECTION_DIM = 256
FF_DIMS = NUM_LAYERS * [[256, 256]]
MASKING_PERCENT = 0.25
MASK_AFTER_EMBEDDING = False
PATCH_R = 3
PATCH_P = 5
PATCH_Z = 5

"""
Optimizer parameters.
"""
N_TRIALS = 50
# Maximum size of a hidden layer
MAX_HIDDEN_LAYER_DIM = 2000

"""
Validator parameter.
"""
FULL_SIM_HISTOGRAM_COLOR = "blue"
ML_SIM_HISTOGRAM_COLOR = "red"
FULL_SIM_GAUSSIAN_COLOR = "green"
ML_SIM_GAUSSIAN_COLOR = "orange"
HISTOGRAM_TYPE = "step"

"""
W&B parameters.
"""
# Change this to your entity name.
WANDB_ENTITY = "piyush_555"
PLOT_FREQ = 50
PLOT_CONFIG = [
    [70, 128, 'SiW']
]  # List of [angle, energy, geometry]
