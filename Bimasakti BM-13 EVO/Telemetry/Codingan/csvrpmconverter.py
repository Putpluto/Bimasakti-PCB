import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import os
import glob


# DONT USE THIS PROGRAM!

# Load the data
input_folder = r"D:\telemetry\withDegreeOutput"
output_folder = r"D:\telemetry\withDegreeOutput\withRPM"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
print("Looking for CSV files in:", input_folder)
csv_files = glob.glob(os.path.join(input_folder, "*.csv")) + glob.glob(os.path.join(input_folder, "*.CSV"))
print("CSV files found:", csv_files)
if not csv_files:
    print("No CSV files found in the specified folder.")
for file_path in csv_files:
    print(f"\nProcessing file: {file_path}")
    try:
        data = pd.read_csv(file_path, delimiter=',', decimal='.')
        print(data.head())
        # Ensure proper column names
        expected_col_list = [
        'Timestamp', 'Suspension1', 'Suspension2', 'Suspension3', 'Suspension4',
        'Steering', 'RPM_FrontRight', 'RPM_FrontLeft', 'RPM_RearRight', 'RPM_RearLeft',
        'Theta', 'Derajat'
]

        if set(data.columns) != set(expected_col_list):
            print(f"Skipping file {file_path} because columns do not match the expected set.")
            continue

        data.columns = expected_col_list

        # Zero out all outliers above 5000 in RPM columns
        data.loc[data['RPM_FrontRight'] > 5000, 'RPM_FrontRight'] = 0
        data.loc[data['RPM_FrontLeft'] > 5000, 'RPM_FrontLeft'] = 0

        # Scale 'Derajat'
        data['Derajat'] = data['Derajat'] * 20 

        #Median Filter
        N = 17  # Adjust as needed
        data['CMA_RPM_FrontRight'] = medfilt(data['RPM_FrontRight'], kernel_size=N)
        data['CMA_RPM_FrontLeft'] = medfilt(data['RPM_FrontLeft'], kernel_size=N)
        original_file_name = os.path.basename(file_path)
        base_name, ext = os.path.splitext(original_file_name)
        new_file_name = base_name + "withDegree" + ext

        # Build the new file path in the output folder
        new_file_path = os.path.join(output_folder, new_file_name)
        data.to_csv("output_with_cma.csv", index=False)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# # Plot the data
# plt.figure(figsize=(12, 6))
# plt.plot(data.index, data['RPM_FrontRight'], label='RPM_FrontRight (Raw)', color='orange', alpha=0)
# plt.plot(data.index, data['CMA_RPM_FrontRight'], label='CMA RPM_FrontRight', color='red',)
# plt.plot(data.index, data['RPM_FrontLeft'], label='RPM_FrontLeft (Raw)', color='blue', alpha=0)
# plt.plot(data.index, data['CMA_RPM_FrontLeft'], label='CMA RPM_FrontLeft', color='purple',)
# plt.plot(data.index, data['Derajat'], label='Derajat', color='green', alpha=0)

# plt.xlabel('Index')
# plt.ylabel('RPM')
# plt.title('RPM Data with Centered Moving Average')
# plt.legend()
# plt.show()
