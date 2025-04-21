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

# Define target and features
y = data['SUS1MM']
X = data[['Suspension1']]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=2529)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
# After fitting the model
slope = model.coef_[0]  # Coefficient for 'adjinverse'
intercept = model.intercept_  # Intercept of the regression line

# Display the linear regression equation
print(f"The linear regression equation is: y = {intercept:.2f} + {slope:.2f} * adjinverse")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2%}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('ADC')
plt.ylabel('Degree')
plt.title('Actual vs Predicted')
plt.legend()
plt.grid(True)
plt.show()
