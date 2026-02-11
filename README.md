# DATA221_Assignment3
# Assignment 3 — Introduction to Data Science

This repository contains my solutions for **Assignment 3** for Introduction to Data Science.
All questions are implemented in Python using pandas, matplotlib, and scikit-learn, with written explanations included directly in the code as comments.

---

## File Overview

### `Question1.py`
- Loads the **crime dataset**
- Computes summary statistics (mean, median, standard deviation, minimum, maximum)
- Includes written interpretation of the results

### `Question2.py`
- Creates a **histogram** and **box plot** for `ViolentCrimesPerPop`
- Properly labeled axes and titles
- Includes written interpretation explaining spread, skewness, and outliers

### `Question3.py`
- Loads the **kidney disease dataset**
- Separates feature matrix `X` and label vector `y`
- Performs a **70/30 train-test split** with a fixed random state
- Includes explanation of why splitting is required

### `Question4.py`
- Trains a **K-Nearest Neighbors (KNN)** classifier with `k = 5`
- Generates predictions on the test set
- Computes confusion matrix, accuracy, precision, recall, and F1-score
- Includes written explanation of TP, TN, FP, FN and why recall is important

### `Question5.py`
- Trains multiple KNN models with different values of `k` (1, 3, 5, 7, 9)
- Compares test accuracy for each `k`
- Identifies the best `k` based on highest test accuracy
- Includes explanation of overfitting vs underfitting and model selection

---

## Datasets Used
- `crime.csv` — used in **Question 1 and Question 2**
- `kidney_disease.csv` — used in **Question 3, 4, and 5**

> All dataset files must be in the **same directory** as the Python scripts.

---

## Requirements
- Python 3
- Required libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scikit-learn`

Install dependencies with:
```bash
pip install pandas numpy matplotlib scikit-learn
