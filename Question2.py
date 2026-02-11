import pandas as pd
import matplotlib.pyplot as plt

# Import the crime dataset (ensure the CSV is in the same folder)
crime_df = pd.read_csv("crime.csv")

# Pull violent crime values and filter out missing entries
violent_rate_values = crime_df["ViolentCrimesPerPop"].dropna()

# Histogram visualization
plt.figure()
plt.hist(violent_rate_values, bins=20)
plt.title("Violent Crime Rate Distribution")
plt.xlabel("Violent Crimes Per Population")
plt.ylabel("Count")
plt.show()

# Box-and-whisker plot
plt.figure()
plt.boxplot(violent_rate_values)
plt.title("Violent Crime Rate Box Plot")
plt.xlabel("Violent Crimes Per Population")
plt.ylabel("Observed Values")
plt.show()

# Interpretation notes
# The histogram indicates that most areas experience lower levels of violent crime,
# while a smaller number show much higher rates.
# The distribution is uneven and stretches further to the right, implying right skewness.
# In the box plot, the line inside the box marks the median crime rate.
# The median’s position closer to the lower end supports the right-skewed pattern.
# Data points outside the whiskers highlight extreme high-crime communities.
