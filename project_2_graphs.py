import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#sklearn

df = pd.read_csv("survey.csv", header=None)
df_array = df.values

# print(df_array)


'''

Quality Control Department

General Q's
Q01 -> How long have you been working for the company?
Q02 -> How confident are you in using SAP software?
Q03 -> How would you rate your job in terms of stress level?
Q04 -> How strongly do you feel connected to the company?
Q05 -> When was the last time you attended a training provided by the company?
Q06 -> Do you feel rewarded for the work you do?
Q07 -> How often do human related errors occur in your job, and leads to serious consequences?
Q08 -> How effectively is the whiteboard in your department used?
Q09 -> Overall, how satisfied are you with your working conditions?
Q10 -> How would you rate yourself from a manager’s perspective?

Production Hall Q's'
Q11 -> How would you evaluate the production unit, in terms of 5S - Sorting
Q12 -> How would you evaluate the production unit, in terms of 5S - Set in order
Q13 -> How would you evaluate the production unit, in terms of 5S - Shine
Q14 -> How would you evaluate the production unit, in terms of 5S - Standardize
Q15 -> How would you evaluate the production unit, in terms of 5S - Sustain
Q16 -> How would you rate the production unit, on a scale of 1 to 10?


'''

# Plot Diagrams

q2_data = df_array[:, 1]

# Q1 vs Q2 analysis

q1_data = df_array[:, 0]
correlation_q1_q2 = np.corrcoef(q1_data, q2_data)[0, 1]
print(f"Correlation between time spent (Q1) and SAP Confidence (Q2): {correlation_q1_q2:.4f}\n")

plt.figure(figsize=(8, 6))
plt.scatter(q1_data, q2_data, alpha=0.7)
plt.title('Experience vs SAP Confidence')
plt.xlabel('Days Worked')
plt.ylabel('SAP Confidence')
plt.grid(False)
plt.show()

# Q2 vs Q10 analysis

q10_data = df_array[:, 9]
correlation_q2_q10 = np.corrcoef(q2_data, q10_data)[0, 1]
print(f"Correlation between SAP Confidence (Q2) and Self Rating (Q10): {correlation_q2_q10:.4f}\n")

plt.figure(figsize=(8, 6))
plt.scatter(q2_data, q10_data, alpha=0.7, color='green')
plt.title('SAP Confidence vs Self Rating')
plt.xlabel('SAP Confidence')
plt.ylabel('Self Rating')
plt.grid(False)
plt.show()


# Histograms

bins = np.arange(0.5, 11.5, 1)

# Stress Level
plt.figure(figsize=(8, 6))
data_q3 = df_array[:, 2]
plt.hist(data_q3, bins=bins, color='orange', rwidth=0.8, edgecolor='black')
plt.title('Stress Level')
plt.xlabel('Rating')
plt.ylabel('Number of Employees')
plt.xticks(range(1, 11))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

avg_stress = np.mean(data_q3)
print(f"Average stress devel: {avg_stress:.2f}")

# Feeling Rewarded
plt.figure(figsize=(8, 6))
data_q6 = df_array[:, 5]
plt.hist(data_q6, bins=bins, color='purple', rwidth=0.8, edgecolor='black')
plt.title('Feeling Rewarded')
plt.xlabel('Rating')
plt.ylabel('Number of Employees')
plt.xticks(range(1, 11))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

avg_rewarded = np.mean(data_q6)
print(f"Averagerating of feeling rewarded: {avg_rewarded:.2f}")

# Feeling Connected
plt.figure(figsize=(8, 6))
data_q4 = df_array[:, 3]
plt.hist(data_q4, bins=bins, color='blue', rwidth=0.8, edgecolor='black')
plt.title('Feeling Connected')
plt.xlabel('Rating')
plt.ylabel('Number of Employees')
plt.xticks(range(1, 11))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

avg_connected = np.mean(data_q4)
print(f"Average rating of feeling connected: {avg_connected:.2f}")

# Whiteboard Effectiveness
plt.figure(figsize=(8, 6))
data_q8 = df_array[:, 7]
plt.hist(data_q8, bins=bins, color='green', rwidth=0.8, edgecolor='black')
plt.title('Whiteboard Effectiveness')
plt.xlabel('Rating')
plt.ylabel('Number of Employees')
plt.xticks(range(1, 11))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

avg_whiteboard = np.mean(data_q8)
print(f"Average rating for whiteboard effectiveness: {avg_whiteboard:.2f}")

# Working Conditions
plt.figure(figsize=(8, 6))
data_q9 = df_array[:, 8]
plt.hist(data_q9, bins=bins, color='red', rwidth=0.8, edgecolor='black')
plt.title('Working Conditions')
plt.xlabel('Rating')
plt.ylabel('Number of Employees')
plt.xticks(range(1, 11))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

avg_conditions = np.mean(data_q9)
print(f"Average satisfaction for working conditions: {avg_conditions:.2f}")
