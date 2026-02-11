import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =========================
# Load and prepare dataset
# =========================

# Read the kidney disease dataset from CSV into a DataFrame
kidney_dataframe = pd.read_csv("kidney_disease.csv")

# Remove any accidental leading/trailing spaces in column names
kidney_dataframe.columns = kidney_dataframe.columns.str.strip()

# Name of the target column we want to predict
target_label_name = "classification"

# =================================
# Separate features and target data
# =================================

# All columns except the classification label are used as input features
feature_dataframe = kidney_dataframe.drop(columns=[target_label_name])

# The classification column ("ckd" or "notckd") is the output label
target_labels = kidney_dataframe[target_label_name]

# ============================
# Handle missing data properly
# ============================

# Identify numeric feature columns (e.g., age, blood pressure, etc.)
numeric_feature_columns = feature_dataframe.select_dtypes(include=["number"]).columns

# Replace missing numeric values with the column median to reduce outlier impact
feature_dataframe[numeric_feature_columns] = (
    feature_dataframe[numeric_feature_columns]
    .fillna(feature_dataframe[numeric_feature_columns].median())
)

# Identify categorical feature columns (e.g., normal/abnormal, yes/no)
categorical_feature_columns = feature_dataframe.select_dtypes(exclude=["number"]).columns

# Replace missing categorical values with the most frequent value (mode) in each column
for column_name in categorical_feature_columns:
    most_frequent_category = feature_dataframe[column_name].mode(dropna=True)[0]
    feature_dataframe[column_name] = feature_dataframe[column_name].fillna(
        most_frequent_category
    )

# ==========================================
# Convert categorical features into numerics
# ==========================================

# Apply one-hot encoding so categorical variables become usable by KNN
# drop_first=True avoids redundant dummy columns
encoded_feature_matrix = pd.get_dummies(feature_dataframe, drop_first=True)

# ========================
# Split data for training
# ========================

# Split data into training (70%) and testing (30%) sets
# random_state ensures reproducible results
X_train, X_test, y_train, y_test = train_test_split(
    encoded_feature_matrix,
    target_labels,
    test_size=0.30,
    random_state=42
)

# ==========================
# Train K-Nearest Neighbors
# ==========================

# Initialize KNN classifier using 5 nearest neighbors
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Train the model using the training data
knn_classifier.fit(X_train, y_train)

# =====================
# Generate predictions
# =====================

# Predict kidney disease labels for the test dataset
predicted_test_labels = knn_classifier.predict(X_test)

# ================================
# Evaluate model performance
# ================================

# Create confusion matrix comparing true vs predicted labels
confusion_matrix_result = confusion_matrix(y_test, predicted_test_labels)

# Calculate standard classification metrics
model_accuracy = accuracy_score(y_test, predicted_test_labels)
model_precision = precision_score(y_test, predicted_test_labels, pos_label="ckd")
model_recall = recall_score(y_test, predicted_test_labels, pos_label="ckd")
model_f1_score = f1_score(y_test, predicted_test_labels, pos_label="ckd")

# ==================
# Display results
# ==================

print("Confusion Matrix:")
print(confusion_matrix_result)
print()

print("Accuracy:", model_accuracy)
print("Precision:", model_precision)
print("Recall:", model_recall)
print("F1-score:", model_f1_score)

# ==================================================
# Written explanations of confusion matrix outcomes
# ==================================================

# True Positive (TP): The model predicts "ckd" and the patient truly has kidney disease.
# True Negative (TN): The model predicts "notckd" and the patient truly does not have kidney disease.
# False Positive (FP): The model predicts "ckd" when the patient is actually healthy, causing unnecessary concern or testing.
# False Negative (FN): The model predicts "notckd" when the patient actually has kidney disease, which is the most dangerous error.
# Accuracy alone can be misleading if one class is much more common than the other.
# In medical diagnosis, recall is especially important because it measures how well the model detects real kidney disease cases.
