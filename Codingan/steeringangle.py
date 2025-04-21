import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read CSV
file_name = "cal 1 post proc.csv"
data = pd.read_csv(file_name, delimiter=';', decimal=',')

# Rename columns
data.columns = ['adc', 'derajat', 'adjinverse']

# Convert to numeric
data['adjinverse'] = pd.to_numeric(data['adjinverse'], errors='coerce')
data['derajat'] = pd.to_numeric(data['derajat'], errors='coerce')

# Drop NaN rows if necessary
data = data.dropna()

# Perform linear regression
slope, intercept = np.polyfit(data['adjinverse'], data['derajat'], 1)
regression_line = slope * data['adc'] + intercept

# Plot
plt.figure(figsize=(12, 6))
plt.plot(data['adc'], data['derajat'], label='adc-derajat', color='orange', alpha=1)
plt.plot(data['adc'], regression_line, label='Linear Regression', color='blue', linestyle='--')

# Add regression equation to plot
equation = f'y = {slope:.2f}x + {intercept:.2f}'
plt.text(0.05, 0.95, equation, transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

plt.xlabel('Time (s)')
plt.ylabel('Vertical Acceleration (m/s²)')
plt.title('Vertical Acceleration with Linear Regression')
plt.legend()
plt.grid()
plt.show()
