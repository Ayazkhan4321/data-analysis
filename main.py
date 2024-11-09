# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve

# Step 1: Load Data
# Replace 'data.csv' with your dataset path
data = pd.read_csv('data.csv')

# Step 2: Data Preprocessing
# View data info and handle missing values
print(data.info())
data = data.dropna()  # Drop missing values, or use data.fillna() for imputation

# Convert categorical columns if necessary
# For example:
# data['gender'] = data['gender'].map({'M': 0, 'F': 1})

# Feature Engineering (e.g., creating interaction features)
data['avg_time_per_session'] = data['total_time'] / data['sessions']
data['interaction_score'] = data['quiz_attempts'] + data['forum_posts']

# Step 3: Exploratory Data Analysis (EDA)
# Plot the distribution of target variable
sns.countplot(x='target', data=data)  # Change 'target' to your target column name
plt.title('Distribution of Pass/Fail')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()

# Step 4: Model Building
# Split data into features (X) and target (y)
X = data.drop(columns=['target'])  # Replace 'target' with the actual target column name
y = data['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 5: Model Evaluation
# Predictions and accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Detailed Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ROC Curve and AUC Score
y_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Step 6: Feature Importance and Visualization
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("Feature Importances:")
print(feature_importances)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importances')
plt.show()

# Optional: Save the model for future use
import joblib
joblib.dump(model, 'student_performance_model.pkl')
