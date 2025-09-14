import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#sklearn

df = pd.read_csv("survey.csv", header=None)
rows = []

for index,row in df.iterrows():
    rows.append(row.values)

df_array = np.array(rows)

print(df_array)

print(type(df_array))

df2 = pd.read_excel("survey.xlsx", header=None)
df_array2 = df2.values

print(df_array2)

print(type(df_array2))