import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend to save plots without GUI
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import os
import yaml

# Load configuration paths from YAML file
with open("config_validation.yml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Define root directory
root_dir = os.path.dirname(os.path.abspath(__file__))

# Assign paths from the config file
total_intake_path = config["total_intake_path"]
manual_folder = config["manual_data_folder"]
modeled_folder = config["modeled_data_folder"]
output_folder = config["output_folder"]

# Create the output folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

# Define the logistic function (LODE model)
def logistic_function(t, L, k, t0):
    return L / (1 + np.exp(-k * (t - t0)))

# Load total intake data from the specified Excel file
total_intake_df = pd.read_excel(total_intake_path)
subject_ids = total_intake_df['Subject'].unique()

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
    modeled_file = os.path.join(modeled_folder, f"{subject_id}_results.csv")
    if not os.path.exists(modeled_file):
        print(f"Skipping {subject_id}: Modeled data file not found.")
        return None, None, None, None

    # Load modeled data and calculate cumulative intake
    modeled_data = pd.read_csv(modeled_file)
    
    # Calculate "Calculated Time (s)" based on "Frame Number" if not present
    if 'Calculated Time (s)' not in modeled_data.columns:
        if 'Frame Number' in modeled_data.columns:
            modeled_data['Calculated Time (s)'] = modeled_data['Frame Number'] / 30
        else:
            print(f"Skipping {subject_id}: 'Frame Number' column missing, cannot calculate 'Calculated Time (s)'.")
            return None, None, None, None

    # Calculate cumulative intake for modeled data
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

# Define global limits for all plots
all_manual_times = []
all_modeled_times = []
all_manual_intakes = []
all_modeled_intakes = []

# Main loop to process each subject
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

    # Append times and intakes for global limit calculation
    all_manual_times.extend(manual_time)
    all_modeled_times.extend(modeled_time)
    all_manual_intakes.extend(manual_lode_pred)
    all_modeled_intakes.extend(modeled_lode_pred)

# Updated plot function with RMSE and final intake errors, consistent x and y limits
def plot_intake_comparison(ax, manual_time, manual_pred, modeled_time, modeled_pred, subject_id, total_intake, rmse_manual_vs_modeled, percent_rmse_manual_vs_modeled):
    # Calculate final cumulative intake values for manual and modeled predictions
    final_manual_intake = manual_pred[-1]
    final_modeled_intake = modeled_pred[-1]
    
    # Calculate errors of final cumulative intake vs actual intake
    manual_final_error_vs_actual = final_manual_intake - total_intake
    modeled_final_error_vs_actual = final_modeled_intake - total_intake

    # Plot the LODE model predictions
    ax.plot(manual_time, manual_pred, label=f'Manual LODE (Final Error vs Actual: {manual_final_error_vs_actual:.2f} g)', linestyle='--')
    ax.plot(modeled_time, modeled_pred, label=f'Modeled LODE (Final Error vs Actual: {modeled_final_error_vs_actual:.2f} g)', linestyle='-')
    
    # Plot actual intake line
    ax.axhline(y=total_intake, color='red', linestyle=':', label='Actual Intake')
    
    # Set consistent axis limits
    ax.set_xlim(0, 1850)
    ax.set_ylim(0, 1200)
    
    # Add title and labels, including RMSE information
    ax.set_title(f'Subject: {subject_id} | RMSE (Manual vs Modeled): {rmse_manual_vs_modeled:.2f}, %RMSE: {percent_rmse_manual_vs_modeled:.2f}%')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cumulative Intake (g)')
    
    # Add legend and grid
    ax.legend()
    ax.grid(True)

# Initialize lists for storing intake errors
final_manual_errors = []
final_modeled_errors = []

# Main loop to process each subject
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

    # Calculate RMSE and %RMSE between manual and modeled predictions
    rmse_manual_vs_modeled = np.sqrt(mean_squared_error(manual_lode_pred, modeled_lode_pred))
    percent_rmse_manual_vs_modeled = (rmse_manual_vs_modeled / total_intake) * 100
    
    # Calculate final cumulative intake errors
    final_manual_intake = manual_lode_pred[-1]
    final_modeled_intake = modeled_lode_pred[-1]
    manual_error = final_manual_intake - total_intake
    modeled_error = final_modeled_intake - total_intake

    # Append errors to lists for summary statistics
    final_manual_errors.append(manual_error)
    final_modeled_errors.append(modeled_error)

    # Plot and save to file
    fig, ax = plt.subplots()
    plot_intake_comparison(ax, manual_time, manual_lode_pred, modeled_time, modeled_lode_pred, subject_id, total_intake, rmse_manual_vs_modeled, percent_rmse_manual_vs_modeled)

    # Save plot as a PNG image in the output folder
    plot_path = os.path.join(output_folder, f"{subject_id}_intake_comparison.png")
    fig.savefig(plot_path)
    plt.close(fig)  # Close the plot to free up memory

    # Print confirmation of saved plot
    print(f"Plot saved for subject {subject_id} at {plot_path}")

    # Print final intake errors for each subject
    print(f"Subject: {subject_id}")
    print(f"Final Manual Intake vs Actual: {final_manual_intake:.2f} g (Error: {manual_error:.2f} g)")
    print(f"Final Modeled Intake vs Actual: {final_modeled_intake:.2f} g (Error: {modeled_error:.2f} g)")

# Summary statistics for final intake errors across all subjects
average_manual_error = np.mean(final_manual_errors)
average_modeled_error = np.mean(final_modeled_errors)
print(f"Average Final Manual Intake Error vs Actual: {average_manual_error:.2f} g")
print(f"Average Final Modeled Intake Error vs Actual: {average_modeled_error:.2f} g")

