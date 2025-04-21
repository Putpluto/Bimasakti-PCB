import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt, savgol_filter


# Load the data
file_name = "log_179withDegree.csv"
data = pd.read_csv(file_name, delimiter=',', decimal='.')

# Ensure proper column names
data.columns = ['Timestamp', 'Suspension1', 'Suspension2', 'Suspension3', 'Suspension4',
                'Steering', 'RPM_FrontRight', 'RPM_FrontLeft', 'RPM_RearRight', 'RPM_RearLeft',
                'Theta', 'Derajat']

# Zero out all outliers above 5000 in RPM columns
data.loc[data['RPM_FrontRight'] > 1370, 'RPM_FrontRight'] = 0
data.loc[data['RPM_FrontLeft'] > 1370, 'RPM_FrontLeft'] = 0
data.loc[data['RPM_RearRight'] > 1370, 'RPM_RearRight'] = 0
data.loc[data['RPM_RearLeft'] > 1370, 'RPM_RearLeft'] = 0
# Scale 'Derajat'
data['Derajat'] = data['Derajat'] * 10

# DONT USE THIS
def flatten_sudden_jumps(series, threshold=500, window=10):
    smoothed_series = series.copy()
    for i in range(1, len(series) - 1):
        if abs(series[i] - series[i - 1]) > threshold: 
            start = max(0, i - window)
            end = min(len(series), i + window + 1)
            smoothed_series[start:end] = np.median(series[start:end]) 
    return smoothed_series

def check_direction(deg_series):
    total = deg_series.sum()
    if total < 0:
        print("CCW")
    elif total > 0:
        print("CW")
    else:
        print("No net rotation")

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


#Median Filter
# Apply the median filter first
N = 17  # Kernel size
data['CMA_RPM_FrontRight'] = medfilt(data['RPM_FrontRight'], kernel_size=N)
data['CMA_RPM_FrontLeft'] = medfilt(data['RPM_FrontLeft'], kernel_size=N)
data['CMA_RPM_RearRight'] = medfilt(data['RPM_RearRight'], kernel_size=N)
data['CMA_RPM_RearLeft'] = medfilt(data['RPM_RearLeft'], kernel_size=N)
# Apply the custom spike-smoothing filter
# data['CMA_RPM_FrontRight'] = flatten_sudden_jumps(data['CMA_RPM_FrontRight'])
# data['CMA_RPM_FrontLeft'] = flatten_sudden_jumps(data['CMA_RPM_FrontLeft'])

# Apply Savitzky-Golay filter for final smoothing
data['CMA_RPM_FrontRight'] = savgol_filter(data['CMA_RPM_FrontRight'], window_length=11, polyorder=2)
data['CMA_RPM_FrontLeft'] = savgol_filter(data['CMA_RPM_FrontLeft'], window_length=11, polyorder=2)
data['CMA_RPM_RearRight'] = savgol_filter(data['CMA_RPM_RearRight'], window_length=11, polyorder=2)
data['CMA_RPM_RearLeft'] = savgol_filter(data['CMA_RPM_RearLeft'], window_length=11, polyorder=2)

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
print_avg_metrics(data, 2660, 4000)
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
plt.plot(data.index, data['RPM_FrontRight'], label='RPM_FrontRight (Raw)', color='darkorange', alpha=0)
plt.plot(data.index, data['CMA_RPM_FrontRight'], label='CMA RPM_FrontRight', color='gold', alpha=0.9)

plt.plot(data.index, data['RPM_FrontLeft'], label='RPM_FrontLeft (Raw)', color='deepskyblue', alpha=0)
plt.plot(data.index, data['CMA_RPM_FrontLeft'], label='CMA RPM_FrontLeft', color='green', alpha=0.9)

plt.plot(data.index, data['RPM_RearRight'], label='RPM_RearRight (Raw)', color='turquoise', alpha=0.2)
plt.plot(data.index, data['CMA_RPM_RearRight'], label='CMA RPM_RearRight', color='blue', alpha=0.9)

plt.plot(data.index, data['RPM_RearLeft'], label='RPM_RearLeft (Raw)', color='violet', alpha=0)
plt.plot(data.index, data['CMA_RPM_RearLeft'], label='CMA RPM_RearLeft', color='red', alpha=0.9)

plt.plot(data.index, data['Derajat'], label='Derajat CW is increment', color='black', alpha=1)

plt.plot(data.index, data['KMPH_FrontRight'], linestyle='--', label='KMPH_FrontRight', color='orangered', alpha=0.9)
plt.plot(data.index, data['KMPH_FrontLeft'], linestyle='--', label='KMPH_FrontLeft', color='gold', alpha=0.9)
plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
plt.xlabel('Index')
plt.ylabel('RPM')
plt.title(f'{file_name} RPM Data with Centered Moving Average')
plt.legend()
plt.show()

from imgui_bundle import imgui, immapp, implot
import pandas as pd
import numpy as np
from scipy.signal import medfilt, savgol_filter
from dataclasses import dataclass

@dataclass
class Lineplot:
    x_data: np.ndarray
    y_data1: np.ndarray
    y_data2: np.ndarray
    y_data3: np.ndarray
    rect: implot.Rect
    flags: int

    def __init__(self, file_name: str = "log_179withDegree.csv"):
        # Load and preprocess the data
        data = pd.read_csv(file_name, delimiter=',', decimal='.')
        data.columns = ['Timestamp', 'Suspension1', 'Suspension2', 'Suspension3', 'Suspension4',
                        'Steering', 'RPM_FrontRight', 'RPM_FrontLeft', 'RPM_RearRight', 'RPM_RearLeft',
                        'Theta', 'Derajat']

        # Zero out all outliers above 1370 in RPM columns
        data.loc[data['RPM_FrontRight'] > 1370, 'RPM_FrontRight'] = 0
        data.loc[data['RPM_FrontLeft'] > 1370, 'RPM_FrontLeft'] = 0
        data.loc[data['RPM_RearRight'] > 1370, 'RPM_RearRight'] = 0
        data.loc[data['RPM_RearLeft'] > 1370, 'RPM_RearLeft'] = 0

        # Scale 'Derajat'
        data['Derajat'] = data['Derajat'] * 10

        # Apply median filter
        N = 17  # Kernel size
        data['CMA_RPM_FrontRight'] = medfilt(data['RPM_FrontRight'], kernel_size=N)
        data['CMA_RPM_FrontLeft'] = medfilt(data['RPM_FrontLeft'], kernel_size=N)
        data['CMA_RPM_RearRight'] = medfilt(data['RPM_RearRight'], kernel_size=N)
        data['CMA_RPM_RearLeft'] = medfilt(data['RPM_RearLeft'], kernel_size=N)

        # Apply Savitzky-Golay filter
        data['CMA_RPM_FrontRight'] = savgol_filter(data['CMA_RPM_FrontRight'], window_length=11, polyorder=2)
        data['CMA_RPM_FrontLeft'] = savgol_filter(data['CMA_RPM_FrontLeft'], window_length=11, polyorder=2)
        data['CMA_RPM_RearRight'] = savgol_filter(data['CMA_RPM_RearRight'], window_length=11, polyorder=2)
        data['CMA_RPM_RearLeft'] = savgol_filter(data['CMA_RPM_RearLeft'], window_length=11, polyorder=2)

        # Convert RPM to KMPH
        data['KMPH_FrontRight'] = data['CMA_RPM_FrontRight'] * 0.0766
        data['KMPH_FrontLeft'] = data['CMA_RPM_FrontLeft'] * 0.0766

        # Store the processed data
        self.x_data = data.index.to_numpy()
        self.y_data1 = data['CMA_RPM_FrontRight'].to_numpy()
        self.y_data2 = data['CMA_RPM_FrontLeft'].to_numpy()
        self.y_data3 = data['Derajat'].to_numpy()

        y_min = min(self.y_data1.min(), self.y_data2.min(), self.y_data3.min())
        y_max = max(self.y_data1.max(), self.y_data2.max(), self.y_data3.max())
        self.rect = implot.Rect(0, len(data), y_min - 10, y_max + 10)  # Add margin
        # Adjust as needed
        self.flags = implot.DragToolFlags_.none.value

    def print_avg_metrics(self, start_idx: int, end_idx: int) -> None:
        if start_idx < 0 or end_idx >= len(self.x_data):
            print("Error: Index out of range.")
            return

        # Extract the relevant portion of the data
        subset = self.y_data1[start_idx:end_idx + 1]

        # Compute non-zero averages for RPM
        avg_rpm_fr = np.mean(subset[subset > 0])
        avg_rpm_fl = np.mean(self.y_data2[start_idx:end_idx + 1][self.y_data2[start_idx:end_idx + 1] > 0])
        avg_rpm_br = np.mean(self.y_data3[start_idx:end_idx + 1][self.y_data3[start_idx:end_idx + 1] > 0])

        print(f"Average CMA_RPM_FrontRight (non-zero) from index {start_idx} to {end_idx}: {avg_rpm_fr:.2f}")
        print(f"Average CMA_RPM_FrontLeft (non-zero) from index {start_idx} to {end_idx}: {avg_rpm_fl:.2f}")
        print(f"Average Derajat (non-zero) from index {start_idx} to {end_idx}: {avg_rpm_br:.2f}")
        print("\n")

    def plot(self) -> None:
        implot.begin_plot("RPM Data")
        implot.setup_axes_limits(self.rect.x.min, self.rect.x.max, self.rect.y.min, self.rect.y.max, implot.Cond_.once)
        implot.plot_line("CMA_RPM_FrontRight", self.x_data, self.y_data1)
        implot.plot_line("CMA_RPM_FrontLeft", self.x_data, self.y_data2)
        implot.plot_line("Derajat", self.x_data, self.y_data3)
        implot.end_plot()


def gui():
    implot.create_context() 
    
    lineplot = Lineplot(file_name="log_179withDegree.csv") # Print average metrics for a specific range

    # Start a new ImGui window
    imgui.begin("RPM Data Visualization")
    lineplot.plot()  # Visualize the data
    imgui.end()



immapp.run(
    gui,  # The Gui function to run
    window_title="RPM Data Analysis",  # the window title
    window_size_auto=True,  # Auto size the application window given its widgets
    window_size=(1000, 800),
    with_implot=True# Uncomment the next line to restore window position and size from previous run
    # window_restore_previous_geometry=True
)