import os
import glob
import pandas as pd

# Set the absolute path to your folder containing the CSV files
input_folder = r"D:\telemetry\zakka"  # Update this path accordingly

# Set the absolute path to the output folder (new folder)
output_folder = r"D:\telemetry\withDegreeOutput"  # Update this path accordingly

# Create the output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print("Looking for CSV files in:", input_folder)

# Use glob to get all CSV files in the folder (both lowercase and uppercase extensions)
csv_files = glob.glob(os.path.join(input_folder, "*.csv")) + glob.glob(os.path.join(input_folder, "*.CSV"))
print("CSV files found:", csv_files)

if not csv_files:
    print("No CSV files found in the specified folder.")

for file_path in csv_files:
    print(f"\nProcessing file: {file_path}")
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Print first few rows to verify content
        print("Initial DataFrame head:")
        print(df.head())
        
        # Check that the 'Steering' column exists
        if 'Steering' not in df.columns:
            print(f"Skipping {file_path}: 'Steering' column not found.")
            continue

        # Calculate the Theta column using the provided formula and round to 3 decimals:
        # Theta = 13.02 + 0.01 * Steering
        df['Theta'] = (13.02 + 0.01 * df['Steering']).round(3)
        
        # Zero the Theta values by subtracting the first Theta value (create Derajat) and round to 3 decimals:
        theta_zero = df['Theta'].iloc[0]
        df['Derajat'] = (df['Theta'] - theta_zero).round(3)
        
        # Create a new filename with "withDegree" appended before the file extension.
        original_file_name = os.path.basename(file_path)
        base_name, ext = os.path.splitext(original_file_name)
        new_file_name = base_name + "withDegree" + ext
        
        # Build the new file path in the output folder
        new_file_path = os.path.join(output_folder, new_file_name)
        
        # Write the updated DataFrame to the new CSV file (without the index column)
        df.to_csv(new_file_path, index=False)
        print(f"Processed and saved: {new_file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
