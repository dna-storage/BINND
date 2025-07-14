import pandas as pd
import matplotlib.pyplot as plt

def plot_logs(train_log_path, val_log_path, out_dir):
    # Load training and validation logs
    train_df = pd.read_csv(train_log_path)
    val_df = pd.read_csv(val_log_path)

    y_min = min(train_df['loss'].min(), val_df['loss'].min())
    y_max = max(train_df['loss'].max(), val_df['loss'].max())

    # Training Loss and Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(train_df.index, train_df['loss'], label='Train Loss', marker='o', linestyle='-')
    plt.xlabel('Batch Index')
    plt.ylabel('Loss')
    plt.title('Training Loss (Batch-wise)')
    # plt.legend()
    plt.grid()
    plt.savefig(f"{out_dir}/train_loss.png", bbox_inches='tight', dpi=300)
  
    plt.figure(figsize=(10, 5))
    plt.plot(train_df.index, train_df['accuracy'], label='Train Accuracy', marker='o', linestyle='-')
    plt.xlabel('Batch Index')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy (Batch-wise)')
    # plt.legend()
    plt.grid()
    plt.savefig(f"{out_dir}/train_accuracy.png")

    # Validation Loss and Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(val_df.index, val_df['loss'], label='Validation Loss', marker='o', linestyle='-')
    plt.xlabel('Batch Index')
    plt.ylabel('Loss')
    plt.ylim(y_min, y_max)
    plt.title('Validation Loss (Batch-wise)')
    # plt.legend()
    plt.grid()
    plt.savefig(f"{out_dir}/val_loss.png", bbox_inches='tight', dpi=300)

    plt.figure(figsize=(10, 5))
    plt.plot(val_df.index, val_df['accuracy'], label='Validation Accuracy', marker='o', linestyle='-')
    plt.xlabel('Batch Index')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy (Batch-wise)')
    # plt.legend()
    plt.grid()
    plt.savefig(f"{out_dir}/val_accuracy.png", bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    train_log_path = "/gpfs_backup/tuck_data/gbrihad/DNABindML/experiments/logs/CNNBinaryClassifierV3Original_0.0_encoder4xn_v2_htp_oct_2024_cond2/train_log.csv"
    val_log_path = "/gpfs_backup/tuck_data/gbrihad/DNABindML/experiments/logs/CNNBinaryClassifierV3Original_0.0_encoder4xn_v2_htp_oct_2024_cond2/val_log.csv"
    out_dir = "/gpfs_backup/tuck_data/gbrihad/DNABindML/experiments/logs/CNNBinaryClassifierV3Original_0.0_encoder4xn_v2_htp_oct_2024_cond2"

    plot_logs(train_log_path, val_log_path, out_dir)