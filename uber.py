import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


sales = pd.read_csv("uber.csv")

# Convert date column
sales["pickup_datetime"] = pd.to_datetime(sales["pickup_datetime"])
sales["hour"] = sales["pickup_datetime"].dt.hour


# Average fare by hour
avg_fare_hour = sales.groupby("hour")["fare_amount"].mean()

# Plot it
plt.plot(avg_fare_hour.index, avg_fare_hour.values)
plt.title("Avg Fare by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Fare ($)")
plt.grid(True)
#plt.show()




#LINEAR REGRESSION MODEL


# Extract day of week (0 = Monday, 6 = Sunday)
sales["dayofweek"] = sales["pickup_datetime"].dt.dayofweek

# Filter to weekends only (Saturday = 5, Sunday = 6)
weekend_sales = sales[sales["dayofweek"].isin([5, 6])]

outlier_threshold = 100
weekend_sales = weekend_sales[weekend_sales["fare_amount"] <= outlier_threshold]

# Group by hour and calculate average fare
avg_fare_hour_weekend = weekend_sales.groupby("hour")["fare_amount"].mean().reset_index()

# Linear Regression
X = avg_fare_hour_weekend[["hour"]]
y = avg_fare_hour_weekend["fare_amount"]

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Plot
plt.figure(figsize=(10, 5))
plt.scatter(X, y, color="blue", label="Avg Weekend Fare", s=60)
plt.plot(X, y_pred, color="red", linewidth=2, label="Regression Line")
plt.xlabel("Hour of Day")
plt.ylabel("Average Fare ($)")
plt.title("Linear Regression: Avg Weekend Fare vs Hour")
plt.grid(True)
plt.legend()
plt.show()






