import pandas as pd
import resource

def calculate_mean_std(csv_path, col_name):
    # Load the CSV file
    data = pd.read_csv(csv_path)
    
    # Extract the column with predicted probabilities
    predictions = data[col_name]
    
    # Calculate mean and standard deviation
    mean = predictions.mean()
    std_dev = predictions.std()
    
    # Return the results
    return mean, std_dev

def get_max_memory_usage_mb():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is in kilobytes on Linux, bytes on macOS
    return usage.ru_maxrss / 1024  # returns MB on Linux

if __name__ == "__main__":
    mean, std = calculate_mean_std("/gpfs_backup/tuck_data/gbrihad/DNABindML/experiments/logs/CNNBinaryClassifierV3Original_0.0_encoder4xn_v2_htp_oct_2024_cond2/test_perf_comp_sequences/test_log.csv",
                        "Probability")
    
    print(f"Mean: {mean}, Standard Deviation: {std}")