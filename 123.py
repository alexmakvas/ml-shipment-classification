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

# Calculate correlation coefficients for class heavy
correlation_heavy = class_heavy_data[features].corr()

# Calculate correlation coefficients for class light
correlation_light = class_light_data[features].corr()

# Plot the correlation matrices as heatmaps
plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1)
sns.heatmap(correlation_heavy, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix for Class heavy')

plt.subplot(1, 2, 2)
sns.heatmap(correlation_light, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix for Class light')

plt.tight_layout()
plt.show()
