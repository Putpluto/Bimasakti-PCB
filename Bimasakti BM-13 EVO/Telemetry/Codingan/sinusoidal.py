import numpy as np
import pandas as pd

def append_sinusoidal_column(csv_file, column_name, min_val, max_val, frequency=1.0, phase=0.0):
    """
    Appends a sinusoidal column to a CSV file.
    
    Parameters:
    csv_file (str): Path to the CSV file.
    column_name (str): Name of the new column.
    min_val (float): Minimum value of the sine wave.
    max_val (float): Maximum value of the sine wave.
    frequency (float, optional): Frequency of the sine wave (default is 1.0).
    phase (float, optional): Phase shift in radians (default is 0.0).
    """
    # Load CSV file
    df = pd.read_csv(csv_file)
    
    # Generate sinusoidal values
    num_rows = len(df)
    x = np.linspace(0, 2 * np.pi * frequency, num_rows) + phase
    amplitude = (max_val - min_val) / 2
    midpoint = (max_val + min_val) / 2
    df[column_name] = midpoint + amplitude * np.sin(x)
    
    # Save the updated CSV file
    df.to_csv(csv_file, index=False)
    print(f"Column '{column_name}' appended to {csv_file}")


append_sinusoidal_column("log_126withDegreecopy.csv", "RPM", 0, 13500, frequency=100, phase=np.pi/2)