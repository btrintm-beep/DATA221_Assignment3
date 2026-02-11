import pandas as pd
from sklearn.model_selection import train_test_split

# Read in the kidney disease dataset
kidney_df = pd.read_csv("kidney_disease.csv")

# Standardize column names by removing extra whitespace
kidney_df.columns = kidney_df.columns.str.strip()

# Display column headers once for verification
print("Available columns:", kidney_df.columns.tolist())

# Target variable name
target_name = "classification"

# Separate predictors and target
X_data = kidney_df.drop(target_name, axis=1)
y_data = kidney_df[target_name]

# Divide data into training and testing subsets
X_train_set, X_test_set, y_train_set, y_test_set = train_test_split(
    X_data,
    y_data,
    test_size=0.3,
    random_state=42
)

# Output dataset dimensions
print("X train size:", X_train_set.shape)
print("X test size:", X_test_set.shape)
print("y train size:", y_train_set.shape)
print("y test size:", y_test_set.shape)

# ---- Conceptual notes ----
# Using the same data for training and evaluation can cause a model to simply memorize
# patterns instead of learning meaningful relationships, leading to overfitting.
# A separate test set allows us to measure how well the model performs on new,
# unseen data, which better reflects real-world performance.
