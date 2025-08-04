import pandas as pd
from sklearn.model_selection import train_test_split
import os


def stratified_split_dataset(csv_file_path, label_column_name, out_dir_path, test_size=0.1, val_size=0.1):

    df = pd.read_csv(csv_file_path)
    temp_test_size = round(test_size + val_size, 2)

    train_df, test_val_df = train_test_split(
        df, test_size=temp_test_size, stratify=df[label_column_name], random_state=42)

    temp_val_size = round(val_size/(test_size + val_size), 2)
    test_df, val_df = train_test_split(
        test_val_df, test_size=temp_val_size, stratify=test_val_df[label_column_name], random_state=42)

    print('train_df shape: ', train_df.shape)
    print('test_df shape: ', test_df.shape)
    print('val_df shape: ', val_df.shape)

    train_df.to_csv(os.path.join(out_dir_path, 'train.csv'), mode='w', index=False)
    val_df.to_csv(os.path.join(out_dir_path, 'val.csv'), mode='w', index=False)
    test_df.to_csv(os.path.join(out_dir_path, 'test.csv'), mode='w', index=False)


def split_predictions_based_on_beadprimer(test_data_path, prediction_data_path, out_dir):
    test_df = pd.read_csv(test_data_path)
    prediction_df = pd.read_csv(prediction_data_path)

    prediction_df["exp_index"] = test_df["exp_index"]

    grouped = prediction_df.groupby('exp_index')

    # Save each group as a separate CSV
    for exp_idx, group in grouped:
        filename = f'{exp_idx}.csv'
        group.to_csv(os.path.join(out_dir, filename), index=False)
        print(f"Saved: {filename}") 


if __name__ == '__main__':
    stratified_split_dataset("/gpfs_backup/tuck_data/gbrihad/DNABindML/data/demo/data.csv",
                             "Label",
                             "/gpfs_backup/tuck_data/gbrihad/DNABindML/data/demo",
                             test_size=0.1,
                             val_size=0.1)
    #=============================================================================================================
    # split_predictions_based_on_beadprimer("/gpfs_backup/tuck_data/gbrihad/htp/data/ml_datasets/htp_oct_2024_cond2/balanced/merged/test.csv",
    #                                       "/gpfs_backup/tuck_data/gbrihad/DNABindML/experiments/logs/BINND_0.0_encoder4xn_v2_htp_oct_2024_cond2/test_log.csv",
    #                                       "/gpfs_backup/tuck_data/gbrihad/DNABindML/outputs/prediction_beadprimer_split")