import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

data = pd.read_csv(r'C:\Users\ADMIN\Desktop\iris flower classification\iris.csv')

# Step 2: Check the data structure
print(data.head())
print(data.info())

data['species'] = data['species'].astype('category').cat.codes

numeric_data = data.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()

plt.figure(figsize=(10, 8))  # Set the figure size
sns.heatmap(
    correlation_matrix,
    annot=True,                # Show correlation values
    fmt=".2f",                 # Format for correlation values
    cmap="coolwarm",           # Color palette
    linewidths=0.5,            # Add lines between cells
    square=True,               # Make cells square
    cbar_kws={"shrink": 0.8},  # Shrink the color bar
    annot_kws={"size": 10},    # Annotation font size
)

plt.title("Correlation Heatmap", fontsize=16, pad=20)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()  # Adjust layout to fit
plt.show()

X = data.drop(columns='species')
y = data['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Train the model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = clf.predict(X_test)
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Encode the species column (convert categorical to numerical)
data['species'] = data['species'].astype('category').cat.codes

# Separate features and target
X = data.drop(columns='species')
y = data['species']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Train a Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

feature_importance = clf.feature_importances_
print(f"Feature Importances: {feature_importance}")

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='d',
            xticklabels=['setosa', 'versicolor', 'virginica'],
            yticklabels=['setosa', 'versicolor', 'virginica'])
plt.xlabel('PREDICTED')
plt.ylabel('ACTUAL')
plt.title('Confusion Matrix')
plt.show()

import joblib

joblib.dump(clf, 'iris_classifier.pkl')

loaded_model = joblib.load('iris_classifier.pkl')
