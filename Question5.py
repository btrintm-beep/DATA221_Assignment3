import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ======================================
# Load and clean the kidney disease data
# ======================================

# Load the dataset from the CSV file into a pandas DataFrame
kidney_dataframe = pd.read_csv("kidney_disease.csv")

# Remove any extra spaces from column names to avoid key errors
kidney_dataframe.columns = kidney_dataframe.columns.str.strip()

# Name of the target column we want to predict
target_column_name = "classification"

# ======================================
# Separate input features and target
# ======================================

# All columns except the classification label are used as model inputs
feature_dataframe = kidney_dataframe.drop(columns=[target_column_name])

# The classification column ("ckd" or "notckd") is the output variable
target_labels = kidney_dataframe[target_column_name]

# ==============================
# Handle missing feature values
# ==============================

# Identify numeric feature columns
numeric_feature_columns = feature_dataframe.select_dtypes(include=["number"]).columns

# Replace missing numeric values with the median of each column
feature_dataframe[numeric_feature_columns] = (
    feature_dataframe[numeric_feature_columns]
    .fillna(feature_dataframe[numeric_feature_columns].median())
)

# Identify categorical feature columns
categorical_feature_columns = feature_dataframe.select_dtypes(exclude=["number"]).columns

# Replace missing categorical values with the most common value (mode)
for column_name in categorical_feature_columns:
    most_frequent_value = feature_dataframe[column_name].mode(dropna=True)[0]
    feature_dataframe[column_name] = feature_dataframe[column_name].fillna(
        most_frequent_value
    )

# =================================================
# Convert categorical variables into numeric format
# =================================================

# Apply one-hot encoding so KNN can calculate distances correctly
encoded_feature_matrix = pd.get_dummies(feature_dataframe, drop_first=True)

# ==========================
# Split data into train/test
# ==========================

# Use 70% of the data for training and 30% for testing
# random_state ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    encoded_feature_matrix,
    target_labels,
    test_size=0.30,
    random_state=42
)

# =====================================================
# Train and evaluate KNN models with different k values
# =====================================================

# List of k values to test
candidate_k_values = [1, 3, 5, 7, 9]

# Store accuracy results for each k
knn_accuracy_results = []

for neighbor_count in candidate_k_values:
    # Initialize KNN model with the current number of neighbors
    knn_classifier = KNeighborsClassifier(n_neighbors=neighbor_count)

    # Train the model on the training data
    knn_classifier.fit(X_train, y_train)

    # Predict labels for the test data
    predicted_test_labels = knn_classifier.predict(X_test)

    # Compute test accuracy for this k value
    test_set_accuracy = accuracy_score(y_test, predicted_test_labels)

    # Save the k value and corresponding accuracy
    knn_accuracy_results.append({
        "k": neighbor_count,
        "accuracy": test_set_accuracy
    })

# ======================================
# Display accuracy results in table form
# ======================================

# Convert results into a DataFrame for cleaner output
accuracy_results_table = pd.DataFrame(knn_accuracy_results)

print("KNN Test Accuracy Results:")
print(accuracy_results_table.to_string(index=False))

# ======================================
# Determine the best k based on accuracy
# ======================================

# Identify the row with the highest test accuracy
best_result_row = accuracy_results_table.loc[
    accuracy_results_table["accuracy"].idxmax()
]

# Extract the optimal k value and its accuracy
optimal_k = int(best_result_row["k"])
highest_test_accuracy = float(best_result_row["accuracy"])

print("\nBest k:", optimal_k)
print("Highest test accuracy:", highest_test_accuracy)

# ==================================================
# Written explanation of k selection and behavior
# ==================================================

# Changing k controls the balance between sensitivity and smoothness in the decision boundary.
# Very small k values (like k=1) can overfit by reacting too strongly to noise and outliers.
# Very large k values can underfit by averaging too much and missing real patterns in the data.
# We select k using test accuracy because it measures how well the model generalizes to unseen patients.
