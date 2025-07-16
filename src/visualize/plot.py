import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_prob_density_func(csv_path, col_name, out_path):
    # Load the CSV file
    data = pd.read_csv(csv_path)  # Replace with your file path

    # Assuming the predicted probabilities are in a column called 'predicted_probability'
    predictions = data[col_name]

    # Create a histogram
    plt.figure(figsize=(8, 6))
    plt.hist(predictions, bins=5, color='blue', edgecolor='black', alpha=0.7)

    # Add titles and labels
    plt.title('Histogram of Predicted Probabilities', fontsize=16)
    plt.xlabel('Predicted Probability', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)

    # Show the plot
    plt.grid(True)
    plt.savefig(out_path)

def plot_prediction_probability_distribution(file_path, out_path):

    # Load CSV
    df = pd.read_csv(file_path)

    df['Label'] = df['Label'].map({0.0: 'Unbound', 1.0: 'Bound'})

    # Plot
    sns.histplot(data=df, x='Probability', hue='Label', bins=30, kde=True, stat="percent", common_norm=False)

    plt.title('BINND Predicted Probability Distribution')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Percent sequences')
    # plt.legend(title='Label')
    plt.savefig(out_path)
   
if __name__ == "__main__":
    # Example usage
    # plot_prob_density_func('/gpfs_backup/tuck_data/gbrihad/DNABindML/experiments/logs/BINND_0.0_encoder4xn_v2_htp_oct_2024_cond2/test_perf_comp_sequences/test_log.csv', 
    #                        'Probability', 
    #                        'prob_density_func.png')
    #========================================================================================================
    plot_prediction_probability_distribution("/gpfs_backup/tuck_data/gbrihad/DNABindML/experiments/logs/BINND_0.0_encoder4xn_v2_htp_oct_2024_cond2/test_random_1mill_E/test_log.csv",
                                             "random_1mill_E.png")