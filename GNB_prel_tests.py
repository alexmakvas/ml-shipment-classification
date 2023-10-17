import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Load the dataset
data = pd.read_csv("shipping_data.csv")
data['weight_class'] = ['light' if w < 2.96 else 'heavy' for w in data['weight (kg)']]
# Delete rows with missing data
data.dropna(inplace=True)

target = 'weight_class'
# Create new columns


# Define features
features = ['price ($)', 'length (m)', 'width (m)', 'height (m)']

# Compute the correlation matrix
corr_matrix = data[features].corr()

# Plot the correlation matrix
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.show()
