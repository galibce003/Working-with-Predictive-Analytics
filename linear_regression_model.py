import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression


df = pd.read_csv('../Python/insurance.csv')
print(df.head(15))



sex = df.iloc[:,1:2].values
smoker = df.iloc[:,4:5].values
region = df.iloc[:,4:5].values



le = LabelEncoder()
smoker[:,0] = le.fit_transform(smoker[:,0])
smoker = pd.DataFrame(smoker)
smoker.columns = ['smoker']
le_smoker_mapping = dict(zip(le.classes_,le.transform(le.classes_)))
print('sklearn label encoder results for smokers:')
print(le_smoker_mapping)
print(smoker[:10])


le = LabelEncoder()
sex[:,0] = le.fit_transform(sex[:,0])
sex = pd.DataFrame(sex)
sex.columns = ['sex']
le_sex_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Sklearn label encoder results for sex:") 
print(le_sex_mapping)
print(sex[:10])



region = df.iloc[:,5:6].values #ndarray
ohe = OneHotEncoder() 
region = ohe.fit_transform(region).toarray()
region = pd.DataFrame(region)
region.columns = ['northeast', 'northwest', 'southeast', 'southwest']
print("Sklearn one hot encoder results for region:") 
print(region[:10])



#Define the train and test data
X_num = df[['age', 'bmi', 'children']].copy()
X_final = pd.concat([X_num,region, sex, smoker], axis = 1)
Y_final = df[['charges']].copy()
X_train, X_test,Y_train, Y_test= train_test_split(X_final,Y_final, test_size= 0.33, random_state = 0)


#Normalize the data
n_scaler = MinMaxScaler()
X_train = n_scaler.fit_transform(X_train.astype(np.float))
X_test = n_scaler.transform(X_test.astype(np.float))
print(X_train)


#Standardize the data
s_scaler = StandardScaler()
X_train = s_scaler.fit_transform(X_train.astype(np.float))
X_test = s_scaler.transform(X_test.astype(np.float))
print(X_train)

lr = LinearRegression().fit(X_train, Y_train)
Y_train_p = lr.predict(X_train)
Y_test_p = lr.predict(X_test)
print(lr.coef_)
print(lr.intercept_)
print(lr.score(X_train, Y_train))
print(lr.score(X_test, Y_test))



      
