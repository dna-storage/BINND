import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class HTPDataset(Dataset):
    def __init__(self, data_path, max_seq_length, encode_fn_name=None):
        self.max_seq_length = max_seq_length
        self.data_df = pd.read_csv(data_path)
        if encode_fn_name == 'nxn':
            self.encode_fn = self._encode_nxn
        elif encode_fn_name == '4xn_v1':
            self.encode_fn = self._encode_4xn_v1
        elif encode_fn_name == '4xn_v2':
            self.encode_fn = self._encode_4xn_v2
        else:
            raise ValueError("Invalid encode_fn_version. Supported versions: 'nxn', '4xn_v1', '4xn_v2'")


    def _encode_nxn(self, seq1, seq2):
        """Encodes two DNA sequences into a max_seq_length x max_seq_length matrix."""
        seq2 = seq2[::-1]  # Reverse second sequence
        pair_scores = {("G", "C"): 1.0, ("C", "G"): 1.0, ("A", "T"): 0.5, ("T", "A"): 0.5}

        # Ensure sequences fit within max_seq_length
        seq1 = seq1[:self.max_seq_length].ljust(self.max_seq_length, "N")
        seq2 = seq2[:self.max_seq_length].ljust(self.max_seq_length, "N")

        # Create matrix
        matrix = np.zeros((self.max_seq_length, self.max_seq_length), dtype=np.float32)
        for i in range(self.max_seq_length):
            for j in range(self.max_seq_length):
                matrix[i, j] = pair_scores.get((seq1[i], seq2[j]), 0.0)
        return np.array([matrix])
    

    def _encode_4xn_v1(self, seq1, seq2):
        '''
        Encode seq1 followed by seq2 in a 4 x max_len matrix.
        '''
        def _encode(seq, max_len):
            nucl_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
            mat = np.zeros((4, max_len), dtype=np.float32)

            for i, nucl in enumerate(seq):
                mat[nucl_dict[nucl]][i] = 1.0
            return mat
        
        enc1 = _encode(seq1, self.max_seq_length)
        enc2 = _encode(seq2, self.max_seq_length)
        return np.array([enc1, enc2])


    def _encode_4xn_v2(self, seq1, seq2):
        '''
        Encode seq1 and seq2 in such a way that bases that will align in case of a perfect binding are togther.
        We assume that both sequences are provided from 5' to 3' end.
        '''
        nucl_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

        matrix = np.zeros((4, self.max_seq_length * 2), dtype=np.float32)

        # reverse seq2
        seq2 = seq2[::-1]

        for i in range(len(seq1)):
            index = i * 2
            matrix[nucl_dict[seq1[i]]][index] = 1.0
            matrix[nucl_dict[seq2[i]]][index + 1] = 1.0

        return np.array([matrix])


    def __len__(self):
        return len(self.data_df)


    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        seq1, seq2 = row["Seq1"], row["Seq2"]
        matrix = self.encode_fn(seq1, seq2)
        matrix = torch.tensor(matrix, dtype=torch.float32)
        label = torch.tensor(row["Label"], dtype=torch.float32)
        return {"matrix": matrix, "label": label}
#end of HTPDataset----------------------------------------------------------------------------------------------------------------------------


def _test_encode_dna_matrix():
    # Create dummy DNA sequences and labels
    data = pd.DataFrame({
        "Seq1": ["ATTC", "GCTA", "CGTA"],
        "Seq2": ["AGCT", "ATCG", "TACG"],
        "Label": [1, 0, 1]
    })

    # Save to CSV (optional, only needed if testing file-based loading)
    csv_path = "test_data.csv"
    data.to_csv(csv_path, index=False)

    # Initialize dataset
    dataset = HTPDataset(data_path=csv_path, max_seq_length=4)

    # Test encoding on the first data point
    sample = dataset.__getitem__(0)  # Get first sample
    matrix = sample["matrix"]
    label = sample["label"]

    # Print results
    print("Encoded Matrix:")
    print(matrix)
    print("\nExpected Label:", label)

    # Assertions for correctness (values should match expected outputs)
    assert matrix.shape == (4, 4), "Matrix shape incorrect!"
    assert isinstance(label, torch.Tensor), "Label is not a torch tensor!"

    print("\nTest Passed!")

if __name__ == '__main__':
    _test_encode_dna_matrix()