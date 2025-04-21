import csv

# Define the header
header = ['accelerometerX', 'accelerometerY', 'accelerometerZ',
          'gyroX', 'gyroY', 'gyroZ', 'date', 'time']

current_date = None
current_rows = []

# Read the input CSV file
with open('DATA.csv', 'r') as infile:
    reader = csv.reader(infile)
    for row in reader:
        # Skip header rows
        if row == header:
            continue
        # Skip rows that don't have exactly 8 columns
        if len(row) != 8:
            continue
        date = row[6]
        # Check if this is the first row
        if current_date is None:
            current_date = date
            current_rows.append(row)
        else:
            if date == current_date:
                current_rows.append(row)
            else:
                # Write the current group to a file
                filename = f"{current_date}.csv"
                with open(filename, 'w', newline='') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(header)
                    writer.writerows(current_rows)
                # Start new group
                current_date = date
                current_rows = [row]

# Write any remaining rows after processing all lines
if current_rows:
    filename = f"{current_date}.csv"
    with open(filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(current_rows)
