import pandas as pd
import numpy as np
import os
import yaml

def load_config(config_path="config_test.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def process_metrics(config):
    # Load paths and file suffixes from config
    csv_root_dir = config['paths']['csv_root_dir']
    excel_root_dir = config['paths']['excel_root_dir']
    output_dir = config['paths']['output_dir']
    metrics_output_excel = os.path.join(output_dir, config['paths']['metrics_output_filename'])

    csv_suffix = config['file_suffixes']['csv_suffix']
    excel_suffix = config['file_suffixes']['excel_suffix']

    frame_number_column = config['columns']['frame_number_column']
    calculated_time_column = config['columns']['calculated_time_column']
    matched_time_column = config['columns']['matched_time_column']
    type_column = config['columns']['type_column']
    time_column_options = config['columns']['time_column_options']
    match_threshold = config['matching']['match_threshold']
    
    # Step 2: Get all CSV and Excel files in the respective directories
    csv_files = [f for f in os.listdir(csv_root_dir) if f.endswith(csv_suffix)]
    excel_files = [f for f in os.listdir(excel_root_dir) if f.endswith(excel_suffix)]

    # Step 3: Initialize a list to store all the metrics for each subject
    all_metrics = []

    # Step 4: Loop through each CSV file and find the corresponding Excel file
    for csv_file in csv_files:
        subject_name = csv_file.replace(csv_suffix, '')
        excel_file = f"{subject_name}{excel_suffix}"
        
        if excel_file in excel_files:
            print(f"Processing subject: {subject_name}")
            coder2_data = pd.read_excel(os.path.join(excel_root_dir, excel_file))

            coder2_time_column = next((col for col in time_column_options if col in coder2_data.columns), None)
            if coder2_time_column is None:
                print(f"Time column not found in {excel_file} for subject {subject_name}. Skipping.")
                continue

            coder2_times = coder2_data[coder2_time_column].to_numpy()
            csv_data = pd.read_csv(os.path.join(csv_root_dir, csv_file))

            # Calculate 'Calculated_Time(s)' from 'Frame Number' at 30 fps
            if frame_number_column in csv_data.columns:
                csv_data[calculated_time_column] = csv_data[frame_number_column] / 30
            else:
                print(f"'{frame_number_column}' column not found in {csv_file} for subject {subject_name}. Skipping.")
                continue

            coder1_times = csv_data[calculated_time_column].to_numpy()
            matched_times = np.zeros(len(coder1_times))
            coder2_times_used = np.zeros(len(coder2_times), dtype=bool)

            diff_matrix = np.abs(coder1_times[:, np.newaxis] - coder2_times)
            closest_indices = np.argmin(diff_matrix, axis=1)
            matched_times = coder2_times[closest_indices]
            coder2_times_used[closest_indices] = True

            csv_data[matched_time_column] = matched_times
            csv_data[type_column] = csv_data.apply(
                lambda row: 'TP' if abs(row[calculated_time_column] - row[matched_time_column]) <= match_threshold else 'FP', axis=1
            )

            unmatched_coder2_times = coder2_times[~coder2_times_used]
            fn_data = pd.DataFrame({
                calculated_time_column: [np.nan] * len(unmatched_coder2_times),
                matched_time_column: unmatched_coder2_times,
                type_column: ['FN'] * len(unmatched_coder2_times)
            })

            combined_data = pd.concat([csv_data[[calculated_time_column, matched_time_column, type_column]], fn_data], ignore_index=True)

            tp_count = len(combined_data[combined_data[type_column] == 'TP'])
            fp_count = len(combined_data[combined_data[type_column] == 'FP'])
            fn_count = len(combined_data[combined_data[type_column] == 'FN'])

            precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
            recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
            accuracy = tp_count / (tp_count + fp_count + fn_count) if (tp_count + fp_count + fn_count) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            all_metrics.append({
                "Subject": subject_name,
                "True Positives (TP)": tp_count,
                "False Positives (FP)": fp_count,
                "False Negatives (FN)": fn_count,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1_score
            })

            print(f"Metrics computed for {subject_name}")

        else:
            print(f"Matching Excel file not found for {subject_name}")

    metrics_df = pd.DataFrame(all_metrics)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with pd.ExcelWriter(metrics_output_excel, engine='xlsxwriter') as writer:
        metrics_df.to_excel(writer, sheet_name='Metrics', index=False)

    print(f"All metrics saved successfully at {metrics_output_excel}")

# Load config and run the process
config = load_config("config_test.yaml")
process_metrics(config)
