import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
from scipy.spatial import distance
from scipy.stats import shapiro
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("shipping_data.csv")
data['weight_class'] = ['light' if w < 2.96 else 'heavy' for w in data['weight (kg)']]
# Delete rows with missing data
data.dropna(inplace=True)

# Define features and target
features = ['price ($)', 'length (m)', 'width (m)', 'height (m)', 'volume (m³)', 'surface area (m²)', 'length-to-width ratio']
features1 = ['price ($)', 'length (m)', 'width (m)', 'height (m)']

target = 'weight_class'
# Create new columns
data['volume (m³)'] = data['length (m)'] * data['width (m)'] * data['height (m)']
data['surface area (m²)'] = 2 * (data['length (m)'] * data['width (m)'] + data['length (m)'] * data['height (m)'] + data['width (m)'] * data['height (m)'])
data['length-to-width ratio'] = data['length (m)'] / data['width (m)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.3, random_state=42)

# Create a StandardScaler object
scaler = StandardScaler()

# Fit the scaler on the training data
scaler.fit(X_train)

# Transform the training and testing data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Assumption 1: Normality
for feature in features:
    for c in data[target].unique():
        feature_data = X_train_scaled[y_train == c, features.index(feature)]
        feature_name = feature
        
        # Test for normality using Shapiro-Wilk test
        stat, p_value = shapiro(feature_data)
        
        # Visualize the distribution using histograms
        plt.figure()
        plt.hist(feature_data, bins=20)
        plt.xlabel(feature_name)
        plt.ylabel("Frequency")
        plt.title(f"{feature_name} Distribution (Class: {c})")
        plt.show()
        
        # Print the normality test results
        print(f"Shapiro-Wilk Test ({feature_name}, Class: {c})")
        print(f"Statistic: {stat:.4f}, p-value: {p_value:.4f}")
        print()
        
# Perform LDA without feature engineering
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_scaled, y_train)
y_pred = lda.predict(X_test_scaled)

# Evaluate the accuracy and confusion matrix
accuracy = lda.score(X_test_scaled, y_test)
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])

print("Accuracy for LDA without feature engineering: {:.4f}".format(accuracy))
print("Confusion Matrix without feature engineering:")
print(confusion_matrix)