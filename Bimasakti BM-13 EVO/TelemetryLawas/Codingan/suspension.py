import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt, savgol_filter
S1 = 4200 # (START)SESUAIKAN DENGAN INDEX SUSPENSI YANG MAU DIJADIKAN ACUAN NOL
S2 = 4500 # (END)
file_name = "log_179withDegree.csv"
data = pd.read_csv(file_name, delimiter=',', decimal='.')

# Ensure proper column names
data.columns = ['Timestamp', 'Suspension1', 'Suspension2', 'Suspension3', 'Suspension4',
                'Steering', 'RPM_FrontRight', 'RPM_FrontLeft', 'RPM_RearRight', 'RPM_RearLeft',
                'Theta', 'Derajat']

for col in ['Suspension1', 'Suspension2', 'Suspension3', 'Suspension4']:
    # data[col] = (data[col] - 26139)/-334.73
    data[col] = ((data[col] - 17785)/-227.97)+7
    data[col] = data[col] - data[col].iloc[S1:S2].mean()  # SESUAIKAN DENGAN INDEX YANG MAU DIJADIKAN ACUAN NOL


N=17
data['Suspension1'] = medfilt(data['Suspension1'], kernel_size=N)
data['Suspension2'] = medfilt(data['Suspension2'], kernel_size=N)
data['Suspension3'] = medfilt(data['Suspension3'], kernel_size=N)
data['Suspension4'] = medfilt(data['Suspension4'], kernel_size=N)


data['Derajat'] = data['Derajat'] / 20
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Suspension1'], label='Suspension1', color='red', alpha=0.8)
plt.plot(data.index, data['Suspension2'], label='Suspension2', color='blue', alpha=0.8)
plt.plot(data.index, data['Suspension3'], label='Suspension3', color='green', alpha=0.8)
plt.plot(data.index, data['Suspension4'], label='Suspension4', color='purple', alpha=0.8)
plt.plot(data.index, data['Derajat'],linewidth=2, label='Derajat CW is increment', color='gold', alpha=1)
plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
plt.xlabel('Index')
plt.ylabel('mm')
plt.title(f'{file_name} mm Data')
plt.legend()
plt.show()