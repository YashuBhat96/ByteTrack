import pandas as pd
import numpy as np
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

# Load configuration
with open("config_ICC.yml", "r") as file:
    config = yaml.safe_load(file)

# Assign paths and parameters from the configuration file
csv_directory = config["csv_directory"]
excel_directory = config["excel_directory"]
output_directory = config["output_directory"]
min_timestamps = config["min_timestamps"]
fps = config["fps"]
behavior_filter = config["behavior_filter"]

# Ensure output directory exists
os.makedirs(output_directory, exist_ok=True)

# Initialize list for ICC results
icc_results = []

# Bland-Altman consolidated plot setup
plt.figure(figsize=(10, 8))
plt.title("Consolidated Bland-Altman Plot for All Subjects")
plt.xlabel("Average of Timestamps (minutes)")
plt.ylabel("Difference in Timestamps (minutes)")

# Process each manual Excel file
for excel_file in os.listdir(excel_directory):
    if not excel_file.endswith(".xlsx"):
        continue

    subject_name = excel_file.replace(".xlsx", "")
    csv_file = f"{subject_name}_results.csv"
    csv_path = os.path.join(csv_directory, csv_file)

    # Check if the corresponding CSV file exists
    if not os.path.exists(csv_path):
        logging.warning(f"Missing CSV file for subject {subject_name}. Skipping.")
        continue

    try:
        # Load manual timestamps
        manual_path = os.path.join(excel_directory, excel_file)
        manual_data = pd.read_excel(manual_path)
        manual_data = manual_data[manual_data["Behavior"] == behavior_filter]  # Filter rows

        # Dynamically select the correct column for timestamps
        if "Time_point_s" in manual_data.columns:
            manual_time_column = "Time_point_s"
        elif "Time_Relative_sf" in manual_data.columns:
            manual_time_column = "Time_Relative_sf"
        else:
            logging.error(f"Missing both 'Time_point_s' and 'Time_Relative_sf' in {excel_file}. Skipping.")
            continue

        manual_timestamps = manual_data[manual_time_column].dropna().to_numpy() / 60  # Convert to minutes

        # Load modeled timestamps
        modeled_data = pd.read_csv(csv_path)
        modeled_timestamps = modeled_data["Frame Number"].dropna().to_numpy() / (fps * 60)  # Convert to minutes

        # Perform exact matching
        min_length = min(len(manual_timestamps), len(modeled_timestamps))
        if min_length < min_timestamps:
            logging.warning(f"Not enough exact matches for subject {subject_name}. Skipping.")
            continue

        manual_timestamps = manual_timestamps[:min_length]
        modeled_timestamps = modeled_timestamps[:min_length]

        # Prepare long-format data for ICC calculation
        bite_labels = [f"bite_{i+1}" for i in range(min_length)]
        long_data = pd.DataFrame({
            "subject": bite_labels * 2,
            "rater": ["Manual"] * min_length + ["Modeled"] * min_length,
            "timestamp": list(manual_timestamps) + list(modeled_timestamps)
        })

        # Calculate ICC(3,1)
        icc_result = pg.intraclass_corr(data=long_data, targets="subject", raters="rater", ratings="timestamp")
        icc_3_1 = icc_result[(icc_result["Type"] == "ICC3") & (icc_result["Description"] == "Single fixed raters")]

        if not icc_3_1.empty:
            icc_value = icc_3_1["ICC"].values[0]
            icc_results.append({"Subject": subject_name, "ICC(3,1)": icc_value})
            logging.info(f"Subject: {subject_name}, ICC(3,1): {icc_value}")
        else:
            logging.warning(f"No valid ICC(3,1) entry for {subject_name}.")

        # Calculate Bland-Altman values
        differences = manual_timestamps - modeled_timestamps
        averages = (manual_timestamps + modeled_timestamps) / 2

        # Plot Bland-Altman for this subject
        plt.scatter(averages, differences, alpha=0.6, label=subject_name)

    except Exception as e:
        logging.error(f"An error occurred while processing {subject_name}: {e}")

# Add reference lines to the Bland-Altman plot
plt.axhline(0, color="gray", linestyle="--", label="Mean Difference")
plt.axhline(differences.mean() + 1.96 * differences.std(), color="red", linestyle="--", label="95% Limits of Agreement")
plt.axhline(differences.mean() - 1.96 * differences.std(), color="red", linestyle="--")

# Finalize and save the Bland-Altman plot
plt.legend(loc="best", fontsize="small")
bland_altman_plot_path = os.path.join(output_directory, "bland_altman_plot.png")
plt.savefig(bland_altman_plot_path)
plt.close()
logging.info(f"Bland-Altman plot saved to {bland_altman_plot_path}")

# Save ICC results to an Excel file
icc_df = pd.DataFrame(icc_results)
icc_output_path = os.path.join(output_directory, "icc_results.xlsx")
icc_df.to_excel(icc_output_path, index=False)
logging.info(f"ICC results saved to {icc_output_path}")
