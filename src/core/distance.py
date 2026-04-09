import Levenshtein as ld


def levenshtein_distance(string1: str, string2: str) -> int:
    """Generates the edit distance between two strings using the Levenshtein distance.
    Args:
        string1 (str): The first string.
        string2 (str): The second string.
    Returns:
        int: The Levenshtein distance between the two strings.
    """
    return ld.distance(string1, string2)


def hamming_distance(seq_1: str, seq_2: str) -> int:
    # Last checked: Apr 5, 2023. Commented
    """Calcualte hamming distance between the given two sequences

    Args:
        seq_1 (str): sequence 1
        seq_2 (str): sequence 2

    Returns:
        int: hamming distance between seq_1 and seq_2
    """
    return sum(c1 != c2 for c1, c2 in zip(seq_1, seq_2))

def get_kmers(sequence : str, k : int =3) -> set:
    """Breaks a sequence into overlapping k-mers.
    
    Args:        
        sequence (str): The input DNA sequence.
        k (int): The length of each k-mer. Default is 3.
    Returns:
        set: A set of unique k-mers from the input sequence.
    """
    return set([sequence[i:i+k] for i in range(len(sequence) - k + 1)])

def jaccard_similarity(seq1 : str, seq2 : str, k : int =3) -> float:
    """Calculates the Jaccard Similarity between two DNA sequences.
    Args:
        seq1 (str): The first DNA sequence.
        seq2 (str): The second DNA sequence.
        k (int): The length of k-mers to use for the calculation. Default is 3.
    Returns:
        float: The Jaccard Similarity between the two sequences, ranging from 0 to 1.
    """
    set1 = get_kmers(seq1, k)
    set2 = get_kmers(seq2, k)
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0
