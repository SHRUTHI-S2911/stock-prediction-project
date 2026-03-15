# Import libraries
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')   # Fix graph display issue
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("stock_data.csv")

# Display first few rows
print(data.head())

# Convert days into numeric values
data['Day'] = np.arange(len(data))

# Features and target
X = data[['Day']]
y = data['Close']

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Predict next day price
next_day = [[len(data)]]
predicted_price = model.predict(next_day)

print("Predicted Next Day Stock Price:", predicted_price[0])

# Plot graph
plt.figure(figsize=(8,5))

# Actual prices
plt.scatter(X, y, label="Actual Price")

# Prediction line
plt.plot(X, model.predict(X), label="Prediction Line")

plt.title("Stock Price Prediction")
plt.xlabel("Days")
plt.ylabel("Stock Price")
plt.legend()

plt.show()