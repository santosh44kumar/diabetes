import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score


diabetes_df = pd.read_csv("diabetes.csv")

diabetes_df.head()

diabetes_df.describe()


plt.figure(figsize=(20,25),facecolor="white")
plotnumber=1

for column in diabetes_df:
  if plotnumber<=9:
    #print(column)
    ax = plt.subplot(3,3,plotnumber)
    sns.distplot(diabetes_df[column])
    plt.xlabel(column,fontsize=20)
  plotnumber+=1
plt.show()

#from pandas profiling and from above plots there are skewness and 0's in few columns
#replacing 0's in the columns with its column mean value

diabetes_df["Pregnancies"] = diabetes_df["Pregnancies"].replace(0,diabetes_df["Pregnancies"].mean())
diabetes_df["BloodPressure"] = diabetes_df["BloodPressure"].replace(0,diabetes_df["BloodPressure"].mean())
diabetes_df["SkinThickness"] = diabetes_df["SkinThickness"].replace(0,diabetes_df["SkinThickness"].mean())
diabetes_df["Insulin"] = diabetes_df["Insulin"].replace(0,diabetes_df["Insulin"].mean())
diabetes_df["BMI"] = diabetes_df["BMI"].replace(0,diabetes_df["BMI"].mean())

plt.figure(figsize=(20,25),facecolor="white")
plotnumber = 1

for column in diabetes_df:
  if plotnumber<=9:
    ax = plt.subplot(3,3,plotnumber)
    sns.distplot(diabetes_df[column])
    plt.xlabel(column,fontsize=20)
  plotnumber+=1
plt.show()

fix, ax = plt.subplots(figsize=(20,10))
sns.boxplot(data=diabetes_df,width=0.5,ax=ax, fliersize=3)
plt.show()

q = diabetes_df["Pregnancies"].quantile(0.98)
# we are removing the top 2% data from the Pregnancies column
data_cleaned = diabetes_df[diabetes_df["Pregnancies"]<q]
q = data_cleaned['BMI'].quantile(0.99)
# we are removing the top 1% data from the BMI column
data_cleaned  = data_cleaned[data_cleaned['BMI']<q]
q = data_cleaned['SkinThickness'].quantile(0.99)
# we are removing the top 1% data from the SkinThickness column
data_cleaned  = data_cleaned[data_cleaned['SkinThickness']<q]
q = data_cleaned['Insulin'].quantile(0.95)
# we are removing the top 5% data from the Insulin column
data_cleaned  = data_cleaned[data_cleaned['Insulin']<q]
q = data_cleaned['DiabetesPedigreeFunction'].quantile(0.99)
# we are removing the top 1% data from the DiabetesPedigreeFunction column
data_cleaned  = data_cleaned[data_cleaned['DiabetesPedigreeFunction']<q]
q = data_cleaned['Age'].quantile(0.99)
# we are removing the top 1% data from the Age column
data_cleaned  = data_cleaned[data_cleaned['Age']<q]

plt.figure(figsize=(20,25),facecolor="white")
plotnumber=1

for column in data_cleaned:
  if plotnumber<=9:
    ax = plt.subplot(3,3,plotnumber)
    sns.distplot(data_cleaned[column])
    plt.xlabel(column,fontsize=20)
  plotnumber+=1
plt.show()

x = diabetes_df.drop(columns=["Outcome"])
y = diabetes_df["Outcome"]

plt.figure(figsize=(25,20),facecolor="white")
plotnumber=1

for column in x:
  if plotnumber<=9:
    ax = plt.subplot(3,3,plotnumber)
    sns.stripplot(y,x[column])
  plotnumber+=1
plt.show()

print(x)

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
x_scaled = scalar.fit_transform(x)

x_scaled

#scalar.inverse_transform(x_scaled)

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["vif"] = [variance_inflation_factor(x_scaled,i) for i in range(x_scaled.shape[1])]
vif["Features"] = x.columns

vif

"""All the VIF values are less than 5 and are very low. That means no multicollinearity. Now, we can go ahead with fitting our data to the model. Before that, let's split our data in test and training set."""

x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.25,random_state=355)

log_reg = LogisticRegression()
log_reg.fit(x_train,y_train)


y_pred = log_reg.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)
accuracy

#Confusion matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

tp = confusion[0][0]
fp = confusion[0][1]
fn = confusion[1][0]
tn = confusion[1][1]

precision = tp/(tp+fp)
precision

recall = tp/(tp+fn)
recall

f1_score = 2*(precision * recall)/(precision + recall)
f1_score

import pickle
pickle.dump(log_reg, open('diabetes.pkl','wb'))