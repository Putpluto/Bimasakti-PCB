import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_name = "0302 copy.csv"  # Use your actual file name
data = pd.read_csv(file_name, delimiter=',', decimal='.')

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
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['RPM_FrontRight'], label='RPM_FrontRight', color='gold', alpha=0.8)
plt.plot(data.index, data['RPM_FrontLeft'], label='RPM_FrontLeft', color='green', alpha=0.8)
plt.plot(data.index, data['RPM_RearRight'], label='RPM_RearRight', color='blue', alpha=0.8)
plt.plot(data.index, data['RPM_RearLeft'], label='RPM_RearLeft', color='red', alpha=0.8)
plt.xlabel('Index')
plt.ylabel('RPM')
plt.title(f'{file_name} Raw RPM Data')
plt.legend()
plt.show()