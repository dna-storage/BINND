import torch
import numpy as np
import os
from networks.cnn import BINND, BINNDLite
from utils.paths import ROOT_DIR
MODEL_PATH = f"{ROOT_DIR}/inference_demo/BINND.pt" # Path to the pre-trained model file. Choose between BINND.pt or BINNDLite.pt
MAX_SEQUENCE_LENGTH = 20 # The current implementation only supports sequences of length 20


# --- Device Setup ---
def get_device() -> torch.device:
    """Determines and returns the appropriate device (CUDA if available, else CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

# --- Data Encoding ---
def encode_sequences(seq1: str, seq2: str, max_seq_length: int) -> np.ndarray:
    """
    Encodes two DNA sequences (seq1 and seq2) into a numerical matrix for model input.
    Both sequences are expected to be provided from 5' to 3' end.

    Args:
        seq1 (str): The first DNA sequence (e.g., target sequence).
        seq2 (str): The second DNA sequence (e.g., guide sequence).
        max_seq_length (int): The expected maximum length of the sequences.
                              This ensures a consistent input size for the model.

    Returns:
        np.ndarray: A NumPy array representing the encoded sequences,
                    with shape (1, 4, max_seq_length * 2).
                    The first dimension is for batching, second for channels (A, C, G, T),
                    and the last for the combined sequence length.
    """
    if not (len(seq1) == len(seq2) and len(seq1) <= max_seq_length):
        raise ValueError(
            f"Sequences must have equal length and not exceed max_seq_length ({max_seq_length}). "
            f"Got seq1 length: {len(seq1)}, seq2 length: {len(seq2)}"
        )

    NUCLEOTIDE_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    # Initialize a zero matrix for the encoding
    matrix = np.zeros((len(NUCLEOTIDE_MAP), max_seq_length * 2), dtype=np.float32)

    # Reverse seq2 to simulate binding orientation (e.g., 5'-3' with 3'-5')
    seq2_rev = seq2[::-1]

    for i in range(len(seq1)):
        index = i * 2
        # One-hot encode seq1 nucleotide
        matrix[NUCLEOTIDE_MAP[seq1[i]]][index] = 1.0
        # One-hot encode seq2 (reversed) nucleotide
        matrix[NUCLEOTIDE_MAP[seq2_rev[i]]][index + 1] = 1.0

    return np.array([matrix]) # Add batch dimension for consistency

# --- Model Inference Function ---
def predict_binding(
    model: torch.nn.Module,
    seq1: str,
    seq2: str,
    max_seq_length: int,
    device: torch.device
) -> tuple[str, float]:
    """
    Performs inference on two DNA sequences to predict binding.

    Args:
        model (torch.nn.Module): The loaded PyTorch model for prediction.
        seq1 (str): The first DNA sequence.
        seq2 (str): The second DNA sequence.
        max_seq_length (int): The maximum expected sequence length for encoding.
        device (torch.device): The device (CPU or CUDA) to run the inference on.

    Returns:
        tuple[str, float]: A tuple containing the predicted label ("Bound" or "Unbound")
                           and the raw probability output from the model.
    """
    # Encode the input sequences
    encoded_input_np = encode_sequences(seq1, seq2, max_seq_length)

    # Convert to PyTorch tensor and add batch dimension
    # Unsqueeze(0) for batch dimension since the model expects (Batch_Size, Channels, Height, Width)
    encoded_input_tensor = torch.tensor(
        encoded_input_np, dtype=torch.float32
    ).unsqueeze(0)
    encoded_input_tensor = encoded_input_tensor.to(device)

    # Perform inference with no gradient calculation
    with torch.no_grad():
        _, output = model(encoded_input_tensor)
    
    output = output.item()  # Get the scalar value from the tensor

    # Determine prediction based on a threshold
    prediction_label = "Bound" if output >= 0.5 else "Unbound"

    return prediction_label, output

if __name__ == "__main__":
    device = get_device()

    # Determine which model class to load based on MODEL_PATH
    filename = os.path.basename(MODEL_PATH)

    # Load the model
    try:
        if filename == "BINND.pt":
            model = BINND().to(device)
        elif filename == "BINNDLite.pt":
            model = BINNDLite().to(device)
        else:
            raise ValueError(f"Unsupported model file: {filename}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()  # Set the model to evaluation mode
        print(f"Model loaded successfully from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}. Please ensure the path is correct.")
        exit() # Exit if model cannot be loaded
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        exit()
    
    
    # Example sequences for inference
    example_seq1 = "AGCGATACGCCTTAACGTCT" # This should be 20 nucleotides long. 5' to 3' end
    example_seq2 = "AATGGCGAAGGGGATCGTTC" # This should also be 20 nucleotides long. 5' to 3' end
    
    # Perform prediction
    predicted_label, probability = predict_binding(
        model, example_seq1, example_seq2, MAX_SEQUENCE_LENGTH, device
    )

    # Display results
    print("\n--- Inference Results ---")
    print(f"Sequence 1: {example_seq1}")
    print(f"Sequence 2: {example_seq2}")
    print(f"Prediction: {predicted_label}, Probability: {probability:.4f}")

