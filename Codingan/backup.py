import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt, savgol_filter

#GENERAL PLOTTER 2 : DONT USE THIS

# Load the data
file_name = "log_126withDegree.csv"
data = pd.read_csv(file_name, delimiter=',', decimal='.')

# Ensure proper column names
data.columns = ['Timestamp', 'Suspension1', 'Suspension2', 'Suspension3', 'Suspension4',
                'Steering', 'RPM_FrontRight', 'RPM_FrontLeft', 'RPM_RearRight', 'RPM_RearLeft',
                'Theta', 'Derajat']

# Zero out all outliers above 5000 in RPM columns
data.loc[data['RPM_FrontRight'] > 5000, 'RPM_FrontRight'] = 0
data.loc[data['RPM_FrontLeft'] > 5000, 'RPM_FrontLeft'] = 0

# Scale 'Derajat'
data['Derajat'] = data['Derajat'] * 20 

#Median Filter
N = 41  # Adjust as needed
data['CMA_RPM_FrontRight'] = medfilt(data['RPM_FrontRight'], kernel_size=N)
data['CMA_RPM_FrontLeft'] = medfilt(data['RPM_FrontLeft'], kernel_size=N)
data['KMPH_FrontRight'] = data['CMA_RPM_FrontRight'] * 0.0766
data['KMPH_FrontLeft'] = data['CMA_RPM_FrontLeft'] * 0.0766


bins = range(1, 5101, 100)
counts_fr, edges_fr = np.histogram(data['RPM_FrontRight'], bins=bins)
print("RPM_FrontRight counts in 100-RPM increments:")
for i in range(len(counts_fr)):
    print(f"{edges_fr[i]}–{edges_fr[i+1]} : {counts_fr[i]}")

counts_fl, edges_fl = np.histogram(data['RPM_FrontLeft'], bins=bins)
print("\nRPM_FrontLeft counts in 100-RPM increments:")
for i in range(len(counts_fl)):
    print(f"{edges_fl[i]}–{edges_fl[i+1]} : {counts_fl[i]}")
bin_centers_fr = (edges_fr[:-1] + edges_fr[1:]) / 2
mean_rpm_fr = np.sum(counts_fr * bin_centers_fr) / np.sum(counts_fr)
print(f"Average RPM_FrontRight (excluding 0 bins): {mean_rpm_fr}")

bin_centers_fl = (edges_fl[:-1] + edges_fl[1:]) / 2
mean_rpm_fl = np.sum(counts_fl * bin_centers_fl) / np.sum(counts_fl)
print(f"Average RPM_FrontLeft (excluding 0 bins): {mean_rpm_fl}")

plt.figure()
plt.bar(edges_fr[:-1], counts_fr, width=100, alpha=0.5, label='RPM_FrontRight')
plt.bar(edges_fl[:-1], counts_fl, width=100, alpha=0.5, label='RPM_FrontLeft')
plt.xlabel('RPM Range')
plt.ylabel('Count')
plt.title('Histogram of RPM in 100-RPM increments starting from 1')
plt.legend()
plt.show()

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['RPM_FrontRight'], label='RPM_FrontRight (Raw)', color='orange', alpha=0.2)
plt.plot(data.index, data['CMA_RPM_FrontRight'], label='CMA RPM_FrontRight', color='red',alpha=1)
plt.plot(data.index, data['RPM_FrontLeft'], label='RPM_FrontLeft (Raw)', color='blue', alpha=0)
plt.plot(data.index, data['CMA_RPM_FrontLeft'], label='CMA RPM_FrontLeft', color='purple',alpha=0)
plt.plot(data.index, data['Derajat'], label='Derajat', color='green', alpha=0)
plt.plot(data.index, data['KMPH_FrontRight'], label='KMPH_FrontRigh', color='green',alpha=0)
plt.plot(data.index, data['KMPH_FrontLeft'], label='KMPH_FrontLeft', color='yellow',alpha=0)

plt.xlabel('Index')
plt.ylabel('RPM')
plt.title('RPM Data with Centered Moving Average')
plt.legend()
plt.show()
