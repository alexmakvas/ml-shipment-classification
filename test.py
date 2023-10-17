import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.stats import stats
from scipy.stats import zscore
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import cross_val_score

# Load the dataset
data = pd.read_csv("shipping_data.csv")

# Delete rows with missing data
data.dropna(inplace=True)

data['weight_class'] = ['light' if w < 2.96 else 'heavy' for w in data['weight (kg)']]

# Create new columns
data['volume (m³)'] = data['length (m)'] * data['width (m)'] * data['height (m)']
data['surface area (m²)'] = 2 * (data['length (m)'] * data['width (m)'] + data['length (m)'] * data['height (m)'] + data['width (m)'] * data['height (m)'])
data['length-to-width ratio'] = data['length (m)'] / data['width (m)']
data['price-to-volume ratio'] = data['price ($)'] / data['volume (m³)']
data['height-to-width ratio'] = data['height (m)'] / data['width (m)']
data['height-to-length ratio'] = data['height (m)'] / data['length (m)']
data['price-to-length ratio'] = data['price ($)'] / data['length (m)']
data['price-to-width ratio'] = data['price ($)'] / data['width (m)']
data['price-to-height ratio'] = data['price ($)'] / data['height (m)']
data['price squared'] = data['price ($)'] * data['price ($)']
data['vertical area'] = data['length (m)'] * data['width (m)']
data['horizontal area 1'] = data['height (m)'] * data['width (m)']
data['horizontal area 2'] = data['height (m)'] * data['length (m)']
data['length squared'] = data['length (m)'] * data['length (m)']
data['width squared'] = data['width (m)'] * data['width (m)']
data['height squared'] = data['height (m)'] * data['height (m)']

features = ['width (m)', 'height (m)', 'length-to-width ratio']
#features for LDA and KNN:
#'price ($)', 'length (m)', 'width (m)', 'height (m)', 'volume (m³)', 'surface area (m²)', 'length-to-width ratio', 'price-to-volume ratio', 'height-to-width ratio', 'height-to-length ratio', 'price-to-length ratio', 'price squared', 'vertical area', 'length squared', 'width squared', 'height squared', 'price-to-width ratio', 'price-to-height ratio', 'horizontal area 1', 'horizontal area 2'

#features for GNB:
#'width (m)', 'height (m)', 'price-to-volume ratio', 'length (m)', 'length-to-width ratio'

#features for QDA: 
# 'width (m)', 'height (m)', 'length-to-width ratio'

#features for KNN:
#'width (m)', 'height (m)', 'length-to-width ratio'

X = data[features]
y = data['weight_class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Split the training set into train and validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.43, random_state=42)

#NORMALIZATION
# Create a StandardScaler object
scaler = StandardScaler()

# Fit the scaler on the training data
scaler.fit(X_train)

# ###OTULIERS

# # Calculate z-scores for 'price ($)' and 'volume' in the training set only
# z_scores_train = np.abs(zscore(X_train))



# # Set threshold for outliers
# threshold = 30

# # Create a boolean array indicating outliers in the training set
# outliers_train = (z_scores_train > threshold).any(axis=1)
# # Filter the training data to extract outliers
# outliers_data = X_train[outliers_train]
# # Filter the training data to remove outliers
# X_train = X_train[~outliers_train]
# y_train = y_train[~outliers_train]

# ###

# ## OVERSAMPLING

# # Apply random oversampling to the training set
# oversampler = RandomOverSampler(random_state=42)
# X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)
# X_train = X_train_resampled
# y_train = y_train_resampled

# ###


# Transform the training, validation, and testing data
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Train classifiers
nb_classif = GNB()
qda_classif = QDA()
knn_classif = KNN()
lda_classif = LDA()

nb_classif.fit(X_train, y_train)
qda_classif.fit(X_train, y_train)
knn_classif.fit(X_train, y_train)
lda_classif.fit(X_train, y_train)

# Predict on test set
nb_pred = nb_classif.predict(X_test)
qda_pred = qda_classif.predict(X_test)
knn_pred = knn_classif.predict(X_test)
lda_pred = lda_classif.predict(X_test)

# Compute accuracy for each classifier
nb_accuracy = accuracy_score(y_test, nb_pred)
qda_accuracy = accuracy_score(y_test, qda_pred)
knn_accuracy = accuracy_score(y_test, knn_pred)
lda_accuracy = accuracy_score(y_test, lda_pred)

print("Accuracy for Naive Bayes: {:.4f}".format(nb_accuracy))
print("Accuracy for QDA: {:.4f}".format(qda_accuracy))
print("Accuracy for KNN: {:.4f}".format(knn_accuracy))
print("Accuracy for LDA: {:.4f}".format(lda_accuracy))

# Evaluate and print confusion matrix for each classifier
nb_cm = confusion_matrix(y_test, nb_pred)
qda_cm = confusion_matrix(y_test, qda_pred)
knn_cm = confusion_matrix(y_test, knn_pred)
lda_cm = confusion_matrix(y_test, lda_pred)

print("Confusion Matrix for Naive Bayes:")
print(nb_cm)
print("Confusion Matrix for QDA:")
print(qda_cm)
print("Confusion Matrix for KNN:")
print(knn_cm)
print("Confusion Matrix for LDA:")
print(lda_cm)
# Perform cross-validation on the training data
cv_scores1 = cross_val_score(nb_classif, X_train, y_train, cv=5)
cv_scores2 = cross_val_score(qda_classif, X_train, y_train, cv=5)
cv_scores3 = cross_val_score(knn_classif, X_train, y_train, cv=5)
cv_scores4 = cross_val_score(lda_classif, X_train, y_train, cv=5)

# Calculate the mean accuracy across all cross-validation folds
cv_accuracy1 = np.mean(cv_scores1)
cv_accuracy2 = np.mean(cv_scores2)
cv_accuracy3 = np.mean(cv_scores3)
cv_accuracy4 = np.mean(cv_scores4)

print("GNB Cross-Validation Accuracy: {:.4f}".format(cv_accuracy1))
print("QDA Cross-Validation Accuracy: {:.4f}".format(cv_accuracy2))
print("KNN Cross-Validation Accuracy: {:.4f}".format(cv_accuracy3))
print("LDA Cross-Validation Accuracy: {:.4f}".format(cv_accuracy4))