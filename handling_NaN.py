import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


df = pd.read_csv('../Python/insurance.csv')
print(df.head(15))


#Count the NaN values
c_n = df.isnull().sum()
print(c_n)
#Output:
#bmi         5




#Replacing the NaN values with the mean()
df['bmi'].fillna(df['bmi'].mean(), inplace = True)
c_n = df.isnull().sum()
print(c_n)
#Output:
#bmi         0
