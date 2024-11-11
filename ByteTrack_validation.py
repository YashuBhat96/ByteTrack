import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import os
import yaml

# Load configuration paths from YAML file
with open("config_validation.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Assign paths from the config file
total_intake_path = config["total_intake_path"]
manual_folder = config["manual_data_folder"]
modeled_folder = config["modeled_data_folder"]

# Define the logistic function (LODE model)
def logistic_function(t, L, k, t0):
    return L / (1 + np.exp(-k * (t - t0)))

# Load total intake data from the specified Excel file
total_intake_df = pd.read_excel(total_intake_path)

# Function to load and process data for each subject
def load_subject_data(subject_id):
    # Check if the subject_id exists in total_intake_df
    if subject_id not in total_intake_df['Subject'].values:
        print(f"Skipping {subject_id}: No matching entry in total intake data.")
        return None, None, None, None

    # Get total intake for the subject
    total_intake = total_intake_df.loc[total_intake_df['Subject'] == subject_id, 'total_g'].values[0]

    # Check if manual data file exists
    manual_file = os.path.join(manual_folder, f"{subject_id}.xlsx")
    if not os.path.exists(manual_file):
        print(f"Skipping {subject_id}: Manual data file not found.")
        return None, None, None, None

    # Load manual data and calculate cumulative intake
    manual_data = pd.read_excel(manual_file)
    time_col = 'Time_point_s' if 'Time_point_s' in manual_data.columns else 'Time_Relative_sf'
    total_manual_bites = len(manual_data)
    intake_per_manual_bite = total_intake / total_manual_bites
    manual_data['Cumulative Intake (g)'] = (manual_data.index + 1) * intake_per_manual_bite

    # Check if modeled data file exists
    modeled_file = os.path.join(modeled_folder, f"{subject_id}_fixed_results.csv")
    if not os.path.exists(modeled_file):
        print(f"Skipping {subject_id}: Modeled data file not found.")
        return None, None, None, None

    # Load modeled data and calculate cumulative intake
    modeled_data = pd.read_csv(modeled_file)
    total_modeled_bites = modeled_data['Bite Count'].max()
    intake_per_modeled_bite = total_intake / total_modeled_bites
    modeled_data['Cumulative Intake (g)'] = modeled_data['Bite Count'] * intake_per_modeled_bite

    return manual_data, modeled_data, total_intake, time_col

# Fit the LODE model using timestamps and cumulative intake
def fit_lode_model(time, intake):
    L = intake.iloc[-1]
    t0 = time.iloc[np.argmax(intake)]
    k = 1 / (time.iloc[-1] - time.iloc[0])
    initial_guess = [L, k, t0]

    try:
        params, _ = curve_fit(logistic_function, time, intake, p0=initial_guess, maxfev=5000)
    except RuntimeError as e:
        print(f"Error fitting LODE model for the given data: {e}")
        return None
    return params

# Plotting function for cumulative intake comparison
def plot_intake_comparison(ax, manual_time, manual_pred, modeled_time, modeled_pred, subject_id, rmse, total_intake):
    ax.plot(manual_time, manual_pred, label='Manual LODE', linestyle='--')
    ax.plot(modeled_time, modeled_pred, label='Modeled LODE', linestyle='-')
    ax.axhline(y=total_intake, color='red', linestyle=':', label='Actual Intake')
    ax.set_title(f'Subject: {subject_id} (RMSE: {rmse:.2f})')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cumulative Intake (g)')
    ax.legend()
    ax.grid(True)

# Main loop to process each subject
subject_ids = total_intake_df['Subject'].unique()
final_manual_errors = []
final_modeled_errors = []

for subject_id in subject_ids:
    manual_data, modeled_data, total_intake, time_col = load_subject_data(subject_id)

    if manual_data is None or modeled_data is None:
        continue  # Skip subjects with missing data or files

    # Fit the LODE model for both manual and modeled data
    manual_params = fit_lode_model(manual_data[time_col], manual_data['Cumulative Intake (g)'])
    modeled_params = fit_lode_model(modeled_data['Calculated Time (s)'], modeled_data['Cumulative Intake (g)'])

    if manual_params is None or modeled_params is None:
        continue

    # Generate LODE predictions
    manual_time = np.linspace(manual_data[time_col].min(), manual_data[time_col].max(), 100)
    manual_lode_pred = logistic_function(manual_time, *manual_params)
    
    modeled_time = np.linspace(modeled_data['Calculated Time (s)'].min(), modeled_data['Calculated Time (s)'].max(), 100)
    modeled_lode_pred = logistic_function(modeled_time, *modeled_params)

    # Calculate RMSE and final intake errors
    rmse = np.sqrt(mean_squared_error(manual_lode_pred, modeled_lode_pred))
    final_manual_intake = manual_lode_pred[-1]
    final_modeled_intake = modeled_lode_pred[-1]
    manual_error = final_manual_intake - total_intake
    modeled_error = final_modeled_intake - total_intake
    final_manual_errors.append(manual_error)
    final_modeled_errors.append(modeled_error)

    # Plot
    fig, ax = plt.subplots()
    plot_intake_comparison(ax, manual_time, manual_lode_pred, modeled_time, modeled_lode_pred, subject_id, rmse, total_intake)
    plt.show()

    # Print final intake errors for each subject
    print(f"Subject: {subject_id}")
    print(f"Final Manual Intake vs Actual: {final_manual_intake:.2f} g (Error: {manual_error:.2f} g)")
    print(f"Final Modeled Intake vs Actual: {final_modeled_intake:.2f} g (Error: {modeled_error:.2f} g)")

# Summary statistics for final intake errors across all subjects
average_manual_error = np.mean(final_manual_errors)
average_modeled_error = np.mean(final_modeled_errors)
print(f"Average Final Manual Intake Error vs Actual: {average_manual_error:.2f} g")
print(f"Average Final Modeled Intake Error vs Actual: {average_modeled_error:.2f} g")
