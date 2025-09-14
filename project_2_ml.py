import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge


df = pd.read_csv("survey.csv", header=None)
df_array = df.values


# SAP Confidence prediction from Experience - Standardized with pipeline
# I used standardation and pipelines just as experiment. It doesn't really change the output

q1_data = df_array[:, 0]
q2_data = df_array[:, 1]

X1 = np.log1p(q1_data).reshape(-1, 1)
y1 = q2_data

X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.2, random_state=42)

model_1 = make_pipeline(
    StandardScaler(),
    LinearRegression())

model_1.fit(X1_train, y1_train)

y1_pred = model_1.predict(X1_test)
mse1 = mean_squared_error(y1_test, y1_pred)
r2_1 = r2_score(y1_test, y1_pred)
print(f"Model 1 performance -> MSE: {mse1:.4f}, R²: {r2_1:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(q1_data, q2_data, alpha=0.7, label='Actual Data')

X_plot = np.linspace(q1_data.min(), q1_data.max(), 100)
X_plot_trans = np.log1p(X_plot).reshape(-1, 1)
y_plot_pred = model_1.predict(X_plot_trans)

plt.plot(X_plot, y_plot_pred, color='red', linewidth=2, label='Log Linear Regression (scaled)')
plt.title('Model 1: Experience vs SAP Confidence (log linear + scaling)')
plt.xlabel('Days Worked')
plt.ylabel('SAP Confidence')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Self Rating prediction from SAP Confidence

X2 = df_array[:, 1].reshape(-1, 1)
y2 = df_array[:, 9]

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

model_2 = LinearRegression()
model_2.fit(X2_train, y2_train)

y2_pred = model_2.predict(X2_test)
mse2 = mean_squared_error(y2_test, y2_pred)
r2_2 = r2_score(y2_test, y2_pred)

print(f"Model 2 performance -> MSE: {mse2:.4f}, R²: {r2_2:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(X2, y2, alpha=0.7, color='green', label='Actual Data')

plt.plot(X2, model_2.predict(X2), color='red', linewidth=2, label='Linear Regression')
plt.title('Model 2: SAP Confidence vs Self Rating (linear)')
plt.xlabel('SAP Confidence')
plt.ylabel('Self Rating')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Production Unit Satisfaction prediction from 5S Scores

X3 = df_array[:, 10:15]
y3 = df_array[:, 15]

X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=42)

model_3 = Ridge(alpha=1.0)
model_3.fit(X3_train, y3_train)

y3_pred = model_3.predict(X3_test)
mse3 = mean_squared_error(y3_test, y3_pred)
r2_3 = r2_score(y3_test, y3_pred)

print(f"Model 3 performance -> MSE: {mse3:.4f}, R²: {r2_3:.4f}\n")

print("Model 3:")
feature_names = ['Sorting', 'Setting in order', 'Shining', 'Standardizing', 'Sustaining']
coefficients = model_3.coef_

for feature, coef in zip(feature_names, coefficients):
    print(f"{feature} coefficient: {coef:.4f}")



