import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("shipping_data.csv")
data['weight_class'] = ['light' if w < 2.96 else 'heavy' for w in data['weight (kg)']]
# Delete rows with missing data
data.dropna(inplace=True)

# Define target variable
target = 'weight_class'

# Define features
features = ['price ($)', 'length (m)', 'width (m)', 'height (m)']

# Separate the data by class
class_heavy_data = data[data[target] == 'heavy']
class_light_data = data[data[target] == 'light']

# Calculate covariance matrices
covariance_heavy = np.cov(class_heavy_data[features].T)
covariance_light = np.cov(class_light_data[features].T)

# Create DataFrame for covariance matrices
covariance_heavy_df = pd.DataFrame(covariance_heavy, columns=features, index=features)
covariance_light_df = pd.DataFrame(covariance_light, columns=features, index=features)

# Plot covariance matrix for Class heavy
plt.figure(figsize=(8, 6))
sns.heatmap(covariance_heavy_df, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Covariance Matrix - Class heavy')
plt.show()

# Plot covariance matrix for Class light
plt.figure(figsize=(8, 6))
sns.heatmap(covariance_light_df, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Covariance Matrix - Class light')
plt.show()
