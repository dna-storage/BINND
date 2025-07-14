import json
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns
import pandas as pd
import os
import numpy as np
import json

def compute_eval_metrics(labels, prediction_probabilities, threshold=0.5, save_path=None):
    """
    Compute various evaluation metrics for a binary classification problem and optionally save results as JSON.
    
    Parameters:
    - labels: Ground truth binary labels (0 or 1).
    - prediction_probabilities: Predicted probabilities for the positive class.
    - threshold: Decision threshold to convert probabilities into class labels.
    - save_path: Path to save the JSON file (default: None, meaning no saving).
    
    Returns:
    - A dictionary containing accuracy, AUC, precision, recall, F1-score, and confusion matrix.
    """
    
    # Convert probabilities to binary predictions
    predictions = (prediction_probabilities >= threshold).astype(int)
    
    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    auc = roc_auc_score(labels, prediction_probabilities)
    f1 = f1_score(labels, predictions, zero_division=0)
    unique_classes = np.unique(labels)
    cm = confusion_matrix(labels, predictions, labels=unique_classes)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
        if 0 in unique_classes:
            idx_0 = np.where(unique_classes == 0)[0][0]
            tn = cm[idx_0, idx_0]
            fp = cm[idx_0, 1 - idx_0] if cm.shape[1] > 1 else 0
        if 1 in unique_classes:
            idx_1 = np.where(unique_classes == 1)[0][0]
            tp = cm[idx_1, idx_1]
            fn = cm[idx_1, 1 - idx_1] if cm.shape[1] > 1 else 0

    # Positive class metrics (label=1)
    precision_pos = precision_score(labels, predictions, pos_label=1, zero_division=0)
    recall_pos = recall_score(labels, predictions, pos_label=1, zero_division=0)

    # Negative class metrics (label=0)
    precision_neg = precision_score(labels, predictions, pos_label=0, zero_division=0)
    recall_neg = recall_score(labels, predictions, pos_label=0, zero_division=0)

    results = {
        "accuracy": accuracy,
        "AUC": auc,
        "F1-score": f1,
        "precision_positive": precision_pos,
        "recall_positive": recall_pos,
        "precision_negative": precision_neg,
        "recall_negative": recall_neg,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn
    }

    results = {
        key: (int(value) if isinstance(value, np.integer) else float(value) if isinstance(value, np.floating) else value)
        for key, value in results.items()
    }

    # Save results to a JSON file if a path is provided
    if save_path:
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                try:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = [data]
                except json.JSONDecodeError:
                    data = []
        else:
            data = []
        # Append new results to the existing data
        data.append(results)
        with open(save_path, "w") as json_file:
            json.dump(data, json_file, indent=4)
    
    return results
    
def plot_roc_curve(labels, prediction_probabilities, save_path=None):
    """
    Plots the ROC curve for binary classification.

    Parameters:
    - labels: List or numpy array of true labels (0 or 1).
    - prediction_probabilities: List or numpy array of predicted probabilities.
    - save_path: (Optional) Path to save the plot as an image.
    """
    fpr, tpr, _ = roc_curve(labels, prediction_probabilities)
    roc_auc = auc(fpr, tpr)

    #save roc curve to csv
    roc_data = pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr})
    roc_data.to_csv(save_path.replace('.png', '.csv'), index=False)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_confusion_matrix(metric_json_path, out_path):
    with open(metric_json_path, "r") as f:
        data = json.load(f)

    data = data[-1]  # Get the last entry in the JSON file
    # Extract confusion matrix values
    tn = data["true_negative"]
    fp = data["false_positive"]
    fn = data["false_negative"]
    tp = data["true_positive"]

    # Create confusion matrix
    cm = np.array([[tn, fp], [fn, tp]])

    # Plot confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    # Save plot
    plt.savefig(out_path, dpi=300)





if __name__ == "__main__":
    # plot_confusion_matrix("/gpfs_backup/tuck_data/gbrihad//DNABindML/experiments/logs/cnn_20x20_downsampled_26mil/test_metrics.json",
    #                       "/gpfs_backup/tuck_data/gbrihad//DNABindML/experiments/logs/cnn_20x20_downsampled_26mil/confusion_matrix.png")
    
    df = pd.read_csv("/gpfs_backup/tuck_data/gbrihad/DNABindML/experiments/logs/cnn_20x20_downsampled_26mil/test_log.csv")

    plot_roc_curve(df['Label'], df['Probability'], "/gpfs_backup/tuck_data/gbrihad/DNABindML/experiments/logs/cnn_20x20_downsampled_26mil/roc_curve.png")