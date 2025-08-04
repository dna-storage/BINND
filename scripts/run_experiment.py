import os
from utils.paths import ROOT_DIR

# --- Define Valid Network Names ---
# These are names of the different CNN architecture classes defined in ROOT_DIR/src/networks/cnn.py
# While there are sevetal CNN architecture classes defined in cnn.py, only a few (e.g. "BINND") are relevant for BINND
VALID_NETWORK_NAMES = (
    "BINND",
    "BINNDLite"
)

# --- Define Valid Encoder Names ---
# These are names of the encoder functions defined in ROOT_DIR/src/dataloader/dataset.py
# While there are several encoder functions defined in dataset.py, only a few (e.g. "4xn_v2") are compatible with CNN architecture of BINND
VALID_ENCODER_NAMES = (
    "4xn_v2"
)

print(f"Root directory is: {ROOT_DIR}")

data_dir = os.path.join(ROOT_DIR, "data", "demo")
checkpoint_dir = os.path.join(ROOT_DIR, "experiments", "checkpoints", "demo")
log_dir = os.path.join(ROOT_DIR, "experiments", "logs", "demo")

# Check if data directory exists
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

# Create directories if they don't exist
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
print(f"Ensured checkpoint directory exists: {checkpoint_dir}")
print(f"Ensured log directory exists: {log_dir}")

network_name = "BINNDLite"
if network_name not in VALID_NETWORK_NAMES:
    raise ValueError(
        f"Invalid network name: {network_name}. Valid options are: {VALID_NETWORK_NAMES}")

encoder_name = "4xn_v2"
if encoder_name not in VALID_ENCODER_NAMES:
    raise ValueError(
        f"Invalid encoder name: {encoder_name}. Valid options are: {VALID_ENCODER_NAMES}")

is_train = True  # Set to True for training, False for inference
is_test = True  # Set to True for testing (inference)

max_seq_length = 20
batch_size = 512
learning_rate = 0.0004
num_epochs = 2
patience = 2
log_interval = 500

# --- Build the Command Arguments List ---
cmd_args = [
    "python", f"{ROOT_DIR}/main.py",
    "--data_dir", data_dir,
    "--checkpoint_dir", checkpoint_dir,
    "--log_dir", log_dir,
    "--network_name", network_name,
    "--encoder_name", encoder_name,
    "--max_seq_length", str(max_seq_length),
    "--batch_size", str(batch_size),
    "--log_interval", str(log_interval),
]

# Add training-specific arguments if is_train is True
if is_train:
    cmd_args.extend([
        "--learning_rate", str(learning_rate),
        "--num_epochs", str(num_epochs),
        "--patience", str(patience),
        "--is_train"  # Flag, no value needed
    ])

# Add testing-specific argument if is_test is True
if is_test:
    cmd_args.append("--is_test")  # Flag, no value needed

cmd = " ".join(cmd_args)

print(f"Running command:\n{cmd}")

os.system(cmd)

