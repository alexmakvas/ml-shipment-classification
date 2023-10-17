import pandas as pd
from sklearn.naive_bayes import GaussianNB
from scipy.stats import normaltest

# Load the dataset
data = pd.read_csv("shipping_data.csv")
data['weight_class'] = ['light' if w < 2.96 else 'heavy' for w in data['weight (kg)']]
# Delete rows with missing data
data.dropna(inplace=True)

target = 'weight_class'
# Define features
features = ['price ($)', 'length (m)', 'width (m)', 'height (m)']

# Extract feature and target variables
X = data[features]
y = data[target]

# Instantiate and fit a Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X, y)

# Check feature independence assumption
# Calculate the correlation matrix
correlation_matrix = X.corr()
# Display the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

# Check normal distribution assumption within classes
# Iterate over each class
for class_label in data[target].unique():
    # Select data points for the current class
    class_data = X[y == class_label]
    # Perform a normality test on each feature within the class
    for feature in features:
        _, p_value = normaltest(class_data[feature])
        p_value = max(p_value, 1e-16)  # Set a minimum p-value for precision
        print(f"Class {class_label}, Feature {feature} - p-value: {p_value:.8f}")
