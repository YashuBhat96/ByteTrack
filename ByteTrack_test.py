import pandas as pd
import numpy as np
import os
import yaml
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI environments
import matplotlib.pyplot as plt
from openpyxl import Workbook


def load_config(config_path="config_test.yml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


# Safe Outlier Detection
def identify_outliers(x, y):
    """
    Identify outliers using Cook's Distance.
    """
    if len(x) == 0 or len(y) == 0:
        return np.array([])  # No outliers if data is empty

    X = sm.add_constant(x)  # Add constant for intercept
    model = sm.OLS(y, X).fit()
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]

    # Identify points where Cook's Distance > 4/N as potential outliers
    threshold = 4 / len(x)
    outliers = np.where(cooks_d > threshold)[0]
    return outliers

# Enhanced Plot Saving
def save_plot(x, y, x_label, y_label, title, filename, output_dir, subject_names, exclude_outliers=False):
    """
    Save scatter plot with optional outlier removal, equal axis scaling, and both reference line (y = x) and trendline.
    """
    if exclude_outliers:
        outliers = identify_outliers(x, y)
        x = np.delete(x, outliers)
        y = np.delete(y, outliers)
        subject_names = [name for i, name in enumerate(subject_names) if i not in outliers]
        title += " (Without Outliers)"
        filename = filename.replace(".png", "_without_outliers.png")

    # Compute metrics
    r2 = r2_score(y, x)
    rmse = np.sqrt(mean_squared_error(y, x))
    percentage_rmse = (rmse / np.mean(y)) * 100 if np.mean(y) > 0 else np.nan

    # Determine axis limits for equal scaling
    min_val = min(min(x), min(y))
    max_val = max(max(x), max(y))
    buffer = (max_val - min_val) * 0.05  # Add a 5% buffer for aesthetics
    axis_limits = (min_val - buffer, max_val + buffer)

    # Compute trendline (linear regression)
    coeffs = np.polyfit(x, y, 1)  # Linear fit (degree 1)
    trendline_x = np.linspace(min(x), max(x), 100)  # Trendline x-values
    trendline_y = np.polyval(coeffs, trendline_x)   # Trendline y-values

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.7, label="Data Points")
    plt.plot(trendline_x, trendline_y, 'b-', lw=2, label=f"Trendline: y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}")
    plt.plot(axis_limits, axis_limits, 'r--', lw=2, label="y = x Reference")  # Reference line

    # Set axis limits and equal scaling
    plt.xlim(axis_limits)
    plt.ylim(axis_limits)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # Display metrics
    plt.text(0.05, 0.95, f'R²: {r2:.2f}\nRMSE: {rmse:.2f}\n%RMSE: {percentage_rmse:.2f}%',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle="round", alpha=0.1))

    # Remove gridlines
    plt.grid(False)

    plt.legend()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def process_metrics(config):
    """
    Compute metrics and save them to an Excel file.
    """
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
    combine_analysis = config['analysis']['combine']  # Combine peak:bite and peak:sips if true

    csv_files = [f for f in os.listdir(csv_root_dir) if f.endswith(csv_suffix)]
    excel_files = [f for f in os.listdir(excel_root_dir) if f.endswith(excel_suffix)]
    all_metrics = []

    for csv_file in csv_files:
        subject_name = csv_file.replace(csv_suffix, '')
        excel_file = f"{subject_name}{excel_suffix}"

        if excel_file in excel_files:
            print(f"Processing subject: {subject_name}")
            coder2_data = pd.read_excel(os.path.join(excel_root_dir, excel_file))

            # Dynamically find the correct time column
            coder2_time_column = next((col for col in time_column_options if col in coder2_data.columns), None)
            if coder2_time_column is None:
                print(f"No valid time column found in {excel_file} for subject {subject_name}. Skipping.")
                continue

            csv_data = pd.read_csv(os.path.join(csv_root_dir, csv_file))

            if frame_number_column in csv_data.columns:
                csv_data[calculated_time_column] = csv_data[frame_number_column] / 30
            else:
                print(f"'{frame_number_column}' column not found in {csv_file} for subject {subject_name}. Skipping.")
                continue

            coder1_times = csv_data[calculated_time_column].to_numpy()

            # Combine analysis if specified
            if combine_analysis:
                combined_analysis_data = coder2_data[coder2_data["Behavior"].isin(["peak:bite", "peak:sips"])]
                analysis_label = "Combined (peak:bite + peak:sips)"
            else:
                combined_analysis_data = coder2_data[coder2_data["Behavior"] == "peak:bite"]
                analysis_label = "peak:bite"

            if combined_analysis_data.empty:
                print(f"No data for {analysis_label} in {excel_file}. Skipping.")
                continue

            # Extract times for the current analysis
            coder2_times = combined_analysis_data[coder2_time_column].to_numpy()

            # Match coder1_times (from CSV) to coder2_times (from Excel)
            diff_matrix = np.abs(coder1_times[:, np.newaxis] - coder2_times)
            closest_indices = np.argmin(diff_matrix, axis=1)
            matched_times = coder2_times[closest_indices]

            csv_data[matched_time_column] = matched_times
            csv_data[type_column] = csv_data.apply(
                lambda row: 'TP' if abs(row[calculated_time_column] - row[matched_time_column]) <= match_threshold else 'FP', axis=1
            )

            fn_data = pd.DataFrame({
                calculated_time_column: [np.nan] * len(coder2_times[~np.isin(coder2_times, matched_times)]),
                matched_time_column: coder2_times[~np.isin(coder2_times, matched_times)],
                type_column: ['FN'] * np.sum(~np.isin(coder2_times, matched_times))
            })

            combined_data = pd.concat([csv_data[[calculated_time_column, matched_time_column, type_column]], fn_data], ignore_index=True)

            tp_count = len(combined_data[combined_data[type_column] == 'TP'])
            fp_count = len(combined_data[combined_data[type_column] == 'FP'])
            fn_count = len(combined_data[combined_data[type_column] == 'FN'])

            # Core metrics
            precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
            recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0



                        # Manual and modeled metrics
            manual_bite_count = len(coder2_times)
            modeled_bite_count = tp_count + fp_count

            # Manual and modeled meal durations
            manual_meal_duration = (coder2_times[-1] - coder2_times[0]) / 60 if len(coder2_times) > 1 else 0
            modeled_meal_duration = (coder1_times[-1] - coder1_times[0]) / 60 if len(coder1_times) > 1 else 0

            # Manual bite rate
            manual_bite_rate = manual_bite_count / manual_meal_duration if manual_meal_duration > 0 else 0

            # TP-based metrics
            tp_based_bite_count = tp_count
            tp_based_bite_count_error = ((tp_based_bite_count - manual_bite_count) / manual_bite_count) * 100 if manual_bite_count > 0 else np.nan
            tp_based_bite_count_rmse = np.sqrt(mean_squared_error([manual_bite_count], [tp_based_bite_count]))
            tp_based_bite_count_rmse_percentage = (tp_based_bite_count_rmse / manual_bite_count) * 100 if manual_bite_count > 0 else np.nan

            tp_based_meal_duration = (combined_data[calculated_time_column].max() - combined_data[calculated_time_column].min()) / 60 if tp_count > 1 else 0
            tp_based_meal_duration_error = ((tp_based_meal_duration - manual_meal_duration) / manual_meal_duration) * 100 if manual_meal_duration > 0 else np.nan
            tp_based_meal_duration_rmse = np.sqrt(mean_squared_error([manual_meal_duration], [tp_based_meal_duration]))
            tp_based_meal_duration_rmse_percentage = (tp_based_meal_duration_rmse / manual_meal_duration) * 100 if manual_meal_duration > 0 else np.nan

            tp_based_bite_rate = tp_based_bite_count / tp_based_meal_duration if tp_based_meal_duration > 0 else 0
            tp_based_bite_rate_error = ((tp_based_bite_rate - manual_bite_rate) / manual_bite_rate) * 100 if manual_bite_rate > 0 else np.nan
            tp_based_bite_rate_rmse = np.sqrt(mean_squared_error([manual_bite_rate], [tp_based_bite_rate]))
            tp_based_bite_rate_rmse_percentage = (tp_based_bite_rate_rmse / manual_bite_rate) * 100 if manual_bite_rate > 0 else np.nan

            # Append metrics
            # Append metrics
            all_metrics.append({
                "Subject": subject_name,
                "Analysis Type": analysis_label,
                "True Positives (TP)": tp_count,
                "False Positives (FP)": fp_count,
                "False Negatives (FN)": fn_count,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1_score,
                "Manual Bite Count": manual_bite_count,
                "Modeled Bite Count": modeled_bite_count,  # Add this line
                "TP-Based Bite Count": tp_based_bite_count,
                "TP-Based Bite Count Error%": tp_based_bite_count_error,
                "TP-Based Bite Count RMSE": tp_based_bite_count_rmse,
                "TP-Based Bite Count RMSE%": tp_based_bite_count_rmse_percentage,
                "Modeled Meal Duration (min)": modeled_meal_duration,  # Add this line if needed for other plots
                "Manual Meal Duration (min)": manual_meal_duration,
                "TP-Based Meal Duration (min)": tp_based_meal_duration,
                "TP-Based Meal Duration Error%": tp_based_meal_duration_error,
                "TP-Based Meal Duration RMSE": tp_based_meal_duration_rmse,
                "TP-Based Meal Duration RMSE%": tp_based_meal_duration_rmse_percentage,
                "Manual Bite Rate (bites/min)": manual_bite_rate,
                "Modeled Bite Rate (bites/min)": modeled_bite_count / modeled_meal_duration if modeled_meal_duration > 0 else 0,
                "TP-Based Bite Rate (bites/min)": tp_based_bite_rate,
                "TP-Based Bite Rate Error%": tp_based_bite_rate_error,
                "TP-Based Bite Rate RMSE": tp_based_bite_rate_rmse,
                "TP-Based Bite Rate RMSE%": tp_based_bite_rate_rmse_percentage,
            })

    # Convert to DataFrame and save to Excel
    metrics_df = pd.DataFrame(all_metrics)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with pd.ExcelWriter(metrics_output_excel, engine='xlsxwriter') as writer:
        metrics_df.to_excel(writer, sheet_name='Metrics', index=False)

    print("Metrics saved successfully.")

    # Generate Plots
    generate_plots(metrics_df, output_dir)


def generate_plots(metrics_df, output_dir):
    """
    Generate required plots from the metrics DataFrame.
    """
    plots = [
        # Modeled vs Manual
        ("Manual Bite Count", "Modeled Bite Count", "Manual Bite Count", "Modeled Bite Count", "manual_vs_modeled_bite_count.png"),
        ("Manual Meal Duration (min)", "Modeled Meal Duration (min)", "Manual Meal Duration (min)", "Modeled Meal Duration (min)", "manual_vs_modeled_meal_duration.png"),
        ("Manual Bite Rate (bites/min)", "Modeled Bite Rate (bites/min)", "Manual Bite Rate (bites/min)", "Modeled Bite Rate (bites/min)", "manual_vs_modeled_bite_rate.png"),

        # TP-Based vs Manual
        ("Manual Bite Count", "TP-Based Bite Count", "Manual Bite Count", "TP-Based Bite Count", "manual_vs_tp_bite_count.png"),
        ("Manual Meal Duration (min)", "TP-Based Meal Duration (min)", "Manual Meal Duration (min)", "TP-Based Meal Duration (min)", "manual_vs_tp_meal_duration.png"),
        ("Manual Bite Rate (bites/min)", "TP-Based Bite Rate (bites/min)", "Manual Bite Rate (bites/min)", "TP-Based Bite Rate (bites/min)", "manual_vs_tp_bite_rate.png"),
    ]

    for true_col, modeled_col, x_label, y_label, filename in plots:
        save_plot(
            metrics_df[true_col].values,
            metrics_df[modeled_col].values,
            x_label, y_label,
            f"{y_label} vs. {x_label}",
            filename,
            output_dir,
            metrics_df["Subject"].values
        )

        save_plot(
            metrics_df[true_col].values,
            metrics_df[modeled_col].values,
            x_label, y_label,
            f"{y_label} vs. {x_label}",
            filename,
            output_dir,
            metrics_df["Subject"].values,
            exclude_outliers=True
        )

    print("Plots saved successfully.")


# Load config and run the process
config = load_config("config_test.yml")
process_metrics(config)





