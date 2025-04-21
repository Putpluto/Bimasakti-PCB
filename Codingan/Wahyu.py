import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt, savgol_filter

# Load the data
file_name = "0302 copy.csv"  # Use your actual file name
data = pd.read_csv(file_name, delimiter=',', decimal='.')
        # Calculate the Theta column using the provided formula and round to 3 decimals:
        # Theta = 13.02 + 0.01 * Steering
if 'Steering' in data.columns:
    # Convert 'Steering' to numeric
    data['Steering'] = pd.to_numeric(data['Steering'], errors='coerce').fillna(0)
    

# Only keep rows from index 15700 to 20500
# if len(data) > 25500:
#     data = data.iloc[25500:29500]  # +1 to include index 20500
#     data.reset_index(drop=True, inplace=True)  # Reset index to start from 0
# else:
#     print("Warning: Data has fewer rows than the starting index 15700")

# Print column names to verify what we're working with
print(f"Original columns: {data.columns.tolist()}")

# Ensure proper column names (modify as needed based on your actual file)
expected_columns = ['Timestamp', 'EngineTemp', 'BatteryVoltage', 'VGear', 'RPM', 
                   'Suspension1', 'Suspension2', 'Suspension3', 'Suspension4',
                   'Steering', 'RPM_FrontRight', 'RPM_FrontLeft', 'RPM_RearRight', 'RPM_RearLeft',
                   'AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ', 'Flag']

# Handle column name assignment safely
if len(data.columns) == len(expected_columns):
    data.columns = expected_columns
else:
    print(f"Warning: Expected {len(expected_columns)} columns, got {len(data.columns)}")
    # Only assign as many column names as we have columns
    num_cols = min(len(expected_columns), len(data.columns))
    data.columns = expected_columns[:num_cols]

# Convert numeric columns to numeric type BEFORE comparison
numeric_columns = data.columns.drop('Timestamp') if 'Timestamp' in data.columns else data.columns

for col in numeric_columns:
    if col in data.columns:
        # Print sample values before conversion for debugging
        print(f"Sample values in {col}: {data[col].iloc[:3].tolist()}")
        data[col] = pd.to_numeric(data[col], errors='coerce')

data['Theta'] = (13.02 + 0.01 * data['Steering']).round(3)

# Zero the Theta values by subtracting the first Theta value (create Derajat) and round to 3 decimals:
theta_zero = data['Theta'].iloc[0]
data['Derajat'] = (data['Theta'] - theta_zero).round(3)
data['Derajat'] = data['Derajat']+10
# Now we can safely zero out RPM values over 1370
# if 'RPM_FrontRight' in data.columns:
#     data.loc[data['RPM_FrontRight'] > 1370, 'RPM_FrontRight'] = 0
# if 'RPM_FrontLeft' in data.columns:
#     data.loc[data['RPM_FrontLeft'] > 1370, 'RPM_FrontLeft'] = 0
# if 'RPM_RearRight' in data.columns:
#     data.loc[data['RPM_RearRight'] > 1370, 'RPM_RearRight'] = 0
# if 'RPM_RearLeft' in data.columns:
#     data.loc[data['RPM_RearLeft'] > 1370, 'RPM_RearLeft'] = 0

# Replace NaN values with zeros
data.fillna(0, inplace=True)

def print_avg_metrics(data, start_idx, end_idx):
    if start_idx < 0 or end_idx >= len(data):
        print("Error: Index out of range.")
        return
    subset = data.iloc[start_idx:end_idx + 1]  # Extract the relevant portion

    # Compute non-zero averages for speed
    avg_kmph_fr = subset['KMPH_FrontRight'][subset['KMPH_FrontRight'] > 0].mean()
    avg_kmph_fl = subset['KMPH_FrontLeft'][subset['KMPH_FrontLeft'] > 0].mean()

    # Compute non-zero averages for RPM
    avg_rpm_fr = subset['CMA_RPM_FrontRight'][subset['CMA_RPM_FrontRight'] > 0].mean()
    avg_rpm_fl = subset['CMA_RPM_FrontLeft'][subset['CMA_RPM_FrontLeft'] > 0].mean()
    avg_rpm_br = subset['CMA_RPM_RearRight'][subset['CMA_RPM_RearRight'] > 0].mean()
    avg_rpm_bl = subset['CMA_RPM_RearLeft'][subset['CMA_RPM_RearLeft'] > 0].mean()

    print(f"Average KMPH_FrontRight (non-zero) from index {start_idx} to {end_idx}: {avg_kmph_fr:.2f}")
    print(f"Average KMPH_FrontLeft (non-zero) from index {start_idx} to {end_idx}: {avg_kmph_fl:.2f}")
    print(f"Average CMA_RPM_FrontRight (non-zero) from index {start_idx} to {end_idx}: {avg_rpm_fr:.2f}")
    print(f"Average CMA_RPM_FrontLeft (non-zero) from index {start_idx} to {end_idx}: {avg_rpm_fl:.2f}")
    print(f"Average CMA_RPM_RearRight (non-zero) from index {start_idx} to {end_idx}: {avg_rpm_br:.2f}")
    print(f"Average CMA_RPM_RearLeft (non-zero) from index {start_idx} to {end_idx}: {avg_rpm_bl:.2f}")
    print("\n")

# Apply median filter
N = min(17, len(data))  # Ensure N is not larger than the dataset
if N % 2 == 0:  # Ensure N is odd
    N -= 1
if N > 1:
    data['CMA_RPM_FrontRight'] = medfilt(data['RPM_FrontRight'], kernel_size=N)
    data['CMA_RPM_FrontLeft'] = medfilt(data['RPM_FrontLeft'], kernel_size=N)
    data['CMA_RPM_RearRight'] = medfilt(data['RPM_RearRight'], kernel_size=N)
    data['CMA_RPM_RearLeft'] = medfilt(data['RPM_RearLeft'], kernel_size=N)
else:
    # If dataset is too small, just copy the original values
    data['CMA_RPM_FrontRight'] = data['RPM_FrontRight'].copy()
    data['CMA_RPM_FrontLeft'] = data['RPM_FrontLeft'].copy()
    data['CMA_RPM_RearRight'] = data['RPM_RearRight'].copy()
    data['CMA_RPM_RearLeft'] = data['RPM_RearLeft'].copy()

# Apply Savitzky-Golay filter with dynamic window size
window_length = min(11, len(data))
if window_length % 2 == 0:  # Ensure window_length is odd
    window_length -= 1
    
# Make sure polyorder is less than window_length
polyorder = min(2, window_length - 1)

if window_length > 2:  # Need at least 3 points for polyorder=2
    data['CMA_RPM_FrontRight'] = savgol_filter(data['CMA_RPM_FrontRight'], window_length=window_length, polyorder=polyorder)
    data['CMA_RPM_FrontLeft'] = savgol_filter(data['CMA_RPM_FrontLeft'], window_length=window_length, polyorder=polyorder)
    data['CMA_RPM_RearRight'] = savgol_filter(data['CMA_RPM_RearRight'], window_length=window_length, polyorder=polyorder)
    data['CMA_RPM_RearLeft'] = savgol_filter(data['CMA_RPM_RearLeft'], window_length=window_length, polyorder=polyorder)

data['KMPH_FrontRight'] = data['CMA_RPM_FrontRight'] * 0.0766
data['KMPH_FrontLeft'] = data['CMA_RPM_FrontLeft'] * 0.0766
print("FILE NAME : "+f'{file_name}')
bins = range(1, 1370, 100)

# FrontRight
counts_fr, edges_fr = np.histogram(data['CMA_RPM_FrontRight'], bins=bins)
print("RPM_FrontRight counts in 100-RPM increments:")
for i in range(len(counts_fr)):
    print(f"{edges_fr[i]}–{edges_fr[i+1]} : {counts_fr[i]}")
bin_centers_fr = (edges_fr[:-1] + edges_fr[1:]) / 2
mean_rpm_fr = np.sum(counts_fr * bin_centers_fr) / np.sum(counts_fr)


# FrontLeft
counts_fl, edges_fl = np.histogram(data['CMA_RPM_FrontLeft'], bins=bins)
print("\nRPM_FrontLeft counts in 100-RPM increments:")
for i in range(len(counts_fl)):
    print(f"{edges_fl[i]}–{edges_fl[i+1]} : {counts_fl[i]}")
bin_centers_fl = (edges_fl[:-1] + edges_fl[1:]) / 2
mean_rpm_fl = np.sum(counts_fl * bin_centers_fl) / np.sum(counts_fl)


# RearRight
counts_rr, edges_rr = np.histogram(data['CMA_RPM_RearRight'], bins=bins)
print("\nRPM_RearRight counts in 100-RPM increments:")
for i in range(len(counts_rr)):
    print(f"{edges_rr[i]}–{edges_rr[i+1]} : {counts_rr[i]}")
bin_centers_rr = (edges_rr[:-1] + edges_rr[1:]) / 2
mean_rpm_rr = np.sum(counts_rr * bin_centers_rr) / np.sum(counts_rr)


# RearLeft
counts_rl, edges_rl = np.histogram(data['CMA_RPM_RearLeft'], bins=bins)
print("\nRPM_RearLeft counts in 100-RPM increments:")
for i in range(len(counts_rl)):
    print(f"{edges_rl[i]}–{edges_rl[i+1]} : {counts_rl[i]}")
bin_centers_rl = (edges_rl[:-1] + edges_rl[1:]) / 2
mean_rpm_rl = np.sum(counts_rl * bin_centers_rl) / np.sum(counts_rl)
print(f"Average RPM_FrontRight (whole dataset excluding zero): {mean_rpm_fr}")
print(f"Average RPM_FrontLeft (whole dataset excluding zero): {mean_rpm_fl}")
print(f"Average RPM_RearRight (whole dataset excluding zero): {mean_rpm_rr}")
print(f"Average RPM_RearLeft (whole dataset excluding zero): {mean_rpm_rl}")

print("\n")
# check_direction(data['Derajat'])
print_avg_metrics(data, 1, len(data)-1)
# print_avg_metrics(data, 2660, 4000)
# print_avg_metrics(data, 4266, 5628)
# print_avg_metrics(data, 5795, 7100)


fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Plot histogram for CMA_RPM_FrontRight
axs[0, 0].bar(edges_fr[:-1], counts_fr,color='red', width=100, alpha=1, label='CMA_RPM_FrontRight')
axs[0, 0].set_xlabel('RPM Range')
axs[0, 0].set_ylabel('Count')
axs[0, 0].set_title(f'{file_name} Histogram of CMA_RPM_FrontRight in 100-RPM increments starting from 1', fontsize=10)
axs[0, 0].legend()

# Plot histogram for CMA_RPM_FrontLeft
axs[0, 1].bar(edges_fl[:-1], counts_fl,color='blue', width=100, alpha=1, label='CMA_RPM_FrontLeft')
axs[0, 1].set_xlabel('RPM Range')
axs[0, 1].set_ylabel('Count')
axs[0, 1].set_title(f'{file_name} Histogram of CMA_RPM_FrontLeft in 100-RPM increments starting from 1', fontsize=10)
axs[0, 1].legend()

# Plot histogram for CMA_RPM_RearRight
axs[1, 0].bar(edges_rr[:-1], counts_rr,color='green', width=100, alpha=1, label='CMA_RPM_RearRight')
axs[1, 0].set_xlabel('RPM Range')
axs[1, 0].set_ylabel('Count')
axs[1, 0].set_title(f'{file_name} Histogram of CMA_RPM_RearRight in 100-RPM increments starting from 1', fontsize=10)
axs[1, 0].legend()

# Plot histogram for CMA_RPM_RearLeft
axs[1, 1].bar(edges_rl[:-1], counts_rl,color='purple', width=100, alpha=1, label='CMA_RPM_RearLeft')
axs[1, 1].set_xlabel('RPM Range')
axs[1, 1].set_ylabel('Count')
axs[1, 1].set_title(f'{file_name} Histogram of CMA_RPM_RearLeft in 100-RPM increments starting from 1', fontsize=10)
axs[1, 1].legend()

plt.tight_layout()
plt.show()

# Plot the data
plt.figure(figsize=(12, 6))
# plt.plot(data.index, data['RPM_FrontRight'], label='RPM_FrontRight (Raw)', color='darkorange', alpha=0)
plt.plot(data.index, data['CMA_RPM_FrontRight'], label='CMA RPM_FrontRight', color='gold', alpha=0.9)

# plt.plot(data.index, data['RPM_FrontLeft'], label='RPM_FrontLeft (Raw)', color='deepskyblue', alpha=0)
plt.plot(data.index, data['CMA_RPM_FrontLeft'], label='CMA RPM_FrontLeft', color='green', alpha=0.9)

# plt.plot(data.index, data['RPM_RearRight'], label='RPM_RearRight (Raw)', color='turquoise', alpha=0.2)
plt.plot(data.index, data['CMA_RPM_RearRight'], label='CMA RPM_RearRight', color='blue', alpha=0.9)

# plt.plot(data.index, data['RPM_RearLeft'], label='RPM_RearLeft (Raw)', color='violet', alpha=0)
plt.plot(data.index, data['CMA_RPM_RearLeft'], label='CMA RPM_RearLeft', color='red', alpha=0.9)
plt.plot(data.index, data['Derajat'], label='Derajat CW is increment', color='black', alpha=1)
plt.plot(data.index, data['KMPH_FrontRight'], linestyle='--', label='KMPH_FrontRight', color='orangered', alpha=0.9)
plt.plot(data.index, data['KMPH_FrontLeft'], linestyle='--', label='KMPH_FrontLeft', color='gold', alpha=0.9)
plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
# plt.axhline(y=100, color='yellow', linestyle='--', alpha=0.5)
plt.xlabel('Index')
plt.ylabel('RPM')
plt.title(f'{file_name} RPM Data with Centered Moving Average')
plt.legend()
plt.show()


def show_row(data, index):
    """Display all values in a specific row by index
    
    Args:
        data: DataFrame containing the data
        index: The row index to display
    """
    if index < 0 or index >= len(data):
        print(f"Error: Index {index} is out of range. Valid range is 0 to {len(data)-1}")
        return
        
    row = data.iloc[index]
    
    print(f"\n--- DATA AT INDEX {index} ---")
    print("-" * 50)
    
    # Get the maximum column name length for nice alignment
    max_col_width = max(len(str(col)) for col in data.columns) + 2
    
    # Print each value with column name
    for col in data.columns:
        value = row[col]
        # Format numerical values nicely
        if isinstance(value, (int, float)):
            if value == int(value):
                value_str = f"{int(value)}"
            else:
                value_str = f"{value:.3f}"
        else:
            value_str = str(value)
            
        print(f"{col.ljust(max_col_width)}: {value_str}")
    
    print("-" * 50)

# Example usage - add this at the end of your script:
while True:
    try:
        index_input = input("\nEnter row index to view (or 'q' to quit): ")
        if index_input.lower() == 'q':
            break
        show_row(data, int(index_input))
    except ValueError:
        print("Please enter a valid integer or 'q' to quit")
