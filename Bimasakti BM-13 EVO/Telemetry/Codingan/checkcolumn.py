import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
# Load dataset
import chardet

# Detect the file's encoding
with open('susregression.csv', 'rb') as file:
    result = chardet.detect(file.read())
    encoding = result['encoding']

# Read the file with the detected encoding
data = pd.read_csv('susregression.csv', sep=';', encoding=encoding, on_bad_lines='skip')
print(data.columns)
import chardet

with open('susregression.csv', 'rb') as f:
    result = chardet.detect(f.read())
    print(result)
