import pandas as pd
import matplotlib.pyplot as plt

def plot_logs(train_log_path, val_log_path, log_interval, out_dir):
    # Load training and validation logs
    train_df = pd.read_csv(train_log_path)
    val_df = pd.read_csv(val_log_path)

    y_min = min(train_df['loss'].min(), val_df['loss'].min())
    y_max = max(train_df['loss'].max(), val_df['loss'].max())

    # Training Loss and Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(train_df.index, train_df['loss'], label='Train Loss', marker='o', linestyle='-')
    plt.xlabel('Training step')
    plt.ylabel('Loss')
    plt.title(f'Training Loss per Log Interval (Average over {log_interval} Batches)')
    # plt.legend()
    plt.grid()
    plt.savefig(f"{out_dir}/train_loss.png", bbox_inches='tight', dpi=300)
  
    plt.figure(figsize=(10, 5))
    plt.plot(train_df.index, train_df['accuracy'], label='Train Accuracy', marker='o', linestyle='-')
    plt.xlabel('Training step')
    plt.ylabel('Accuracy')
    plt.title(f'Training Accuracy per Log Interval (Average over {log_interval} Batches)')
    # plt.legend()
    plt.grid()
    plt.savefig(f"{out_dir}/train_accuracy.png", bbox_inches='tight', dpi=300)

    # Validation Loss and Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(val_df.index, val_df['loss'], label='Validation Loss', marker='o', linestyle='-')
    plt.xlabel('Validation step')
    plt.ylabel('Loss')
    plt.ylim(y_min, y_max)
    plt.title(f'Validation Loss per Log Interval (Average over {log_interval} Batches)')
    # plt.legend()
    plt.grid()
    plt.savefig(f"{out_dir}/val_loss.png", bbox_inches='tight', dpi=300)

    plt.figure(figsize=(10, 5))
    plt.plot(val_df.index, val_df['accuracy'], label='Validation Accuracy', marker='o', linestyle='-')
    plt.xlabel('Validation step')
    plt.ylabel('Accuracy')
    plt.title(f'Validation Accuracy per Log Interval (Average over {log_interval} Batches)')
    # plt.legend()
    plt.grid()
    plt.savefig(f"{out_dir}/val_accuracy.png", bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    train_log_path = "/gpfs_backup/tuck_data/gbrihad/DNABindML/experiments/logs/demo/train_log.csv"
    val_log_path = "/gpfs_backup/tuck_data/gbrihad/DNABindML/experiments/logs/demo/val_log.csv"
    out_dir = "/gpfs_backup/tuck_data/gbrihad/DNABindML/experiments/logs/demo"

    plot_logs(train_log_path, val_log_path, 500, out_dir)