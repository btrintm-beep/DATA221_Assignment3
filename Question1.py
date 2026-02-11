import pandas as pd

# Read the crime dataset (file must be in the same directory)
crime_df = pd.read_csv("crime.csv")

# Extract the violent crime rate column
violent_rates_series = crime_df["ViolentCrimesPerPop"]

# Eliminate any missing or undefined values
violent_rates_series = violent_rates_series.dropna()

# Compute descriptive statistics
avg_rate = violent_rates_series.mean()
midpoint_rate = violent_rates_series.median()
spread_rate = violent_rates_series.std()
lowest_rate = violent_rates_series.min()
highest_rate = violent_rates_series.max()

# Display the computed statistics
print("Violent Crime Rate Summary")
print("Average:", avg_rate)
print("Median:", midpoint_rate)
print("Standard Deviation:", spread_rate)
print("Lowest Value:", lowest_rate)
print("Highest Value:", highest_rate)

# Conceptual explanations
# Mean vs. Median:
# When the mean is noticeably higher than the median, the data is right-skewed,
# indicating that a few large values are increasing the average.
# If both values are similar, the distribution is closer to symmetric.

# Impact of outliers:
# The mean is sensitive to extreme values because every observation contributes
# to it. The median is more robust since it depends only on the center value
# once the data is ordered.
