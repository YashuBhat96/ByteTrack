import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import os
import logging
import yaml

# Set matplotlib to non-interactive mode
import matplotlib
matplotlib.use("Agg")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Get the root directory and config file path
root_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(root_dir, 'config_ICC.yml')

# Print paths to confirm
print("Root Directory:", root_dir)
print("Config Path:", config_path)

# Load configuration from config_ICC.yaml
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)


# Assign paths and parameters from the configuration file
csv_directory = config['csv_directory']
excel_directory = config['excel_directory']
output_directory = config['output_directory']
plot_title = config['plot_title']
plot_x_label = config['plot_x_label']
plot_y_label = config['plot_y_label']
min_timestamps = config['min_timestamps']
fps = config['fps']

# Ensure output directory exists
os.makedirs(output_directory, exist_ok=True)

# Initialize list to store ICC results and set up the consolidated Bland-Altman plot
icc_results = []
plt.figure(figsize=(10, 8))
plt.title(plot_title)
plt.xlabel(plot_x_label)
plt.ylabel(plot_y_label)

# List all files in the CSV and Excel directories
csv_files = [f for f in os.listdir(csv_directory) if f.endswith('_results.csv')]
excel_files = [f for f in os.listdir(excel_directory) if f.endswith('.xlsx')]

# Process each subject based on matching files in the two directories
for excel_file in excel_files:
    subject_name = excel_file.replace('.xlsx', '')
    csv_file = f"{subject_name}_results.csv"

    # Ensure a matching CSV file exists
    if csv_file in csv_files:
        try:
            # Load coder1 timestamps from the Excel file
            coder1_path = os.path.join(excel_directory, excel_file)
            coder1_data = pd.read_excel(coder1_path)
            coder1_times = coder1_data['Time_point_s'].dropna().to_numpy() if 'Time_point_s' in coder1_data.columns else []

            # Load coder2 timestamps from the CSV file, converting Frame Number to seconds
            coder2_path = os.path.join(csv_directory, csv_file)
            coder2_data = pd.read_csv(coder2_path)
            coder2_times = (coder2_data['Frame Number'] / fps).dropna().to_numpy() if 'Frame Number' in coder2_data.columns else []

            # Align the two sets of timestamps by trimming to the shortest list
            min_length = min(len(coder1_times), len(coder2_times))
            if min_length < min_timestamps:
                logging.warning(f"Skipping {subject_name}: fewer than {min_timestamps} matched timestamps.")
                continue

            coder1_times, coder2_times = coder1_times[:min_length], coder2_times[:min_length]

            # Prepare long-format data for ICC calculation
            bite_labels = [f"bite_{i+1}" for i in range(min_length)]
            long_data = pd.DataFrame({
                'subject': bite_labels * 2,
                'rater': ['Rater1'] * min_length + ['Rater2'] * min_length,
                'timestamp': list(coder1_times) + list(coder2_times)
            })

            # Calculate ICC(3,1) for absolute agreement between the two fixed raters
            icc_result = pg.intraclass_corr(data=long_data, targets='subject', raters='rater', ratings='timestamp')
            icc_3_1 = icc_result[(icc_result['Type'] == 'ICC3') & (icc_result['Description'] == 'Single fixed raters')]
            
            if not icc_3_1.empty:
                icc_value = icc_3_1['ICC'].values[0]
                icc_results.append({'Subject': subject_name, 'ICC(3,1)': icc_value})
                logging.info(f"ICC(3,1) for {subject_name}: {icc_value}")
            else:
                logging.warning(f"No valid ICC(3,1) entry for {subject_name}.")

            # Calculate differences and averages for Bland-Altman
            differences = coder1_times - coder2_times
            averages = (coder1_times + coder2_times) / 2

            # Plot each subject as a line on the consolidated Bland-Altman plot
            plt.plot(averages, differences, label=subject_name, linestyle='-', marker=None, alpha=0.5)

        except Exception as e:
            logging.error(f"An error occurred while processing {subject_name}: {e}")

# Add reference lines to the Bland-Altman plot and save it
plt.axhline(0, color='gray', linestyle='--')
plt.axhline(differences.mean() + 1.96 * differences.std(), color='red', linestyle='--', label='95% Limits of Agreement')
plt.axhline(differences.mean() - 1.96 * differences.std(), color='red', linestyle='--')

# Add a legend and save the consolidated plot
plt.legend(loc='best', fontsize='small')
plot_path = os.path.join(output_directory, 'consolidated_bland_altman_plot.png')
plt.savefig(plot_path)
plt.close()

# Save all ICC results to an Excel file
icc_df = pd.DataFrame(icc_results)
icc_df.to_excel(os.path.join(output_directory, 'icc_results.xlsx'), index=False)

logging.info("Consolidated Bland-Altman plot and ICC calculations have been saved.")
