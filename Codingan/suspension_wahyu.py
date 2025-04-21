import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt, savgol_filter

S1 = 36500  # (START)SESUAIKAN DENGAN INDEX SUSPENSI YANG MAU DIJADIKAN ACUAN NOL
S2 = 39000 # (END)
file_name = "0302 copy.csv"
data = pd.read_csv(file_name, delimiter=',', decimal='.')

# Ensure proper column names
expected_columns = ['Timestamp', 'EngineTemp', 'BatteryVoltage', 'VGear', 'RPM',
                    'Suspension1', 'Suspension2', 'Suspension3', 'Suspension4',
                    'Steering', 'RPM_FrontRight', 'RPM_FrontLeft', 'RPM_RearRight', 'RPM_RearLeft',
                    'AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ', 'Flag']

# Assign column names safely
# if len(data) > 25500:
#     if len(data) >= 29500:
#         data = data.iloc[25500:29500]  # Slicing from index 25500 to 29500
#     else:
#         print(f"Warning: Dataset has only {len(data)} rows, skipping slicing.")
#     data.reset_index(drop=True, inplace=True)  # Reset index to start from 0
# elif len(data) > 0:
#     print("Warning: Dataset has fewer than 25500 rows. Skipping slicing operation.")
# else:
#     print("Error: Dataset is empty. Please check the input file.")
#     exit(1)

# Convert 'Steering' to numeric and create 'Derajat'
if 'Steering' in data.columns:
    data['Steering'] = pd.to_numeric(data['Steering'], errors='coerce').fillna(0)
    data['Steering'] = data['Steering'].astype(float)
    data['Theta'] = (13.02 + 0.01 * data['Steering']).round(3)
    theta_zero = data['Theta'].iloc[0]
    data['Derajat'] = (data['Theta'] - theta_zero).round(3)
    data['Derajat'] = data['Derajat']/10
else:
    data['Derajat'] = 0  # Create 'Derajat' column if 'Steering' is missing

# Convert suspension columns to numeric
for col in ['Suspension1', 'Suspension2', 'Suspension3', 'Suspension4', 'Derajat']:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

# Apply calculations to suspension columns
for col in ['Suspension1', 'Suspension2', 'Suspension3', 'Suspension4']:
    if col in data.columns:
        data[col] = ((data[col] - 17785) / -227.97) + 7
        data[col] = data[col] - data.iloc[S1:S2][col].mean()

N = 17
for col in ['Suspension1', 'Suspension2', 'Suspension3', 'Suspension4']:
    if col in data.columns:
        data[col] = medfilt(data[col], kernel_size=N)


# Plotting
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Suspension1'], label='Suspension1', color='red', alpha=0.8)
plt.plot(data.index, data['Suspension2'], label='Suspension2', color='blue', alpha=0.8)
plt.plot(data.index, data['Suspension3'], label='Suspension3', color='green', alpha=0.8)
plt.plot(data.index, data['Suspension4'], label='Suspension4', color='purple', alpha=0.8)
plt.plot(data.index, data['Derajat'], linewidth=2, label='Derajat CW is increment', color='gold', alpha=1)
plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
plt.xlabel('Index')
plt.ylabel('mm')
plt.title(f'{file_name} mm Data')
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
    print(data["Derajat"].iloc[index])
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
