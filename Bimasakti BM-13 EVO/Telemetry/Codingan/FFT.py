import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq
# Load the data
file_name = "log_126withDegree.csv"
data = pd.read_csv(file_name, delimiter=',', decimal='.')

# Ensure proper column names
data.columns = ['Timestamp', 'Suspension1', 'Suspension2', 'Suspension3', 'Suspension4',
                'Steering', 'RPM_FrontRight', 'RPM_FrontLeft', 'RPM_RearRight', 'RPM_RearLeft',
                'Theta', 'Derajat']


# Perform FFT
n = len(data['RPM_FrontRight'])  # Number of data points
fs = 50  # Sampling frequency (Hz)
fft_values = fft(data['RPM_FrontRight'])
fft_frequencies = fftfreq(n, d=1/fs)

# Plot the FFT
plt.figure(figsize=(12, 6))
plt.plot(fft_frequencies[:n//2], np.abs(fft_values[:n//2]))  # Positive frequencies
plt.title('Frequency Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

