import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

#GENERAL PLOTTER 1 : DATA MENTAH (RAW)

# Load the data
file_name = "log_126withDegree.csv"
data = pd.read_csv(file_name, delimiter=',', decimal='.')

# Ensure proper column names
data.columns = ['Timestamp' ,'Suspension1' ,'Suspension2' ,'Suspension3' ,'Suspension4','Steering' ,'RPM_FrontRight' ,'RPM_FrontLeft','RPM_RearRight','RPM_RearLeft','Theta','Derajat']
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['RPM_FrontRight'], label='Original Vertical Acceleration (ay)',color='orange', alpha=0.5)
plt.plot(data.index, data['RPM_FrontLeft'], label='Filtered Vertical Acceleration (ay)',color='blue', alpha=0.5)
plt.show()
