# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Finally execute the program and display the output.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SURYAMALARV
RegisterNumber:  212223230224
*/

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
dataset=pd.read_csv("/content/Placement_Data_Full_Class (1).csv")
dataset.head()
```
![image](https://github.com/user-attachments/assets/3f23a9db-cd8d-4cf6-87eb-51879c75965e)
```
dataset.tail()
```
![image](https://github.com/user-attachments/assets/13e2281b-9aa3-4dd6-872a-6dd79695569a)
```
dataset.info()
```
![image](https://github.com/user-attachments/assets/8e8f81cd-28cf-4d22-a7be-337770538797)
```
dataset=dataset.drop('sl_no',axis=1)
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes
```
![image](https://github.com/user-attachments/assets/d6f0aa6c-ca13-4e09-a7b4-09126f0c8d0a)
```
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset
```
![image](https://github.com/user-attachments/assets/6886629f-5115-4a49-abba-a4df2145fafb)
```
dataset.info()
```
![image](https://github.com/user-attachments/assets/3db11d33-1377-4d91-a874-2513358f7e60)
```
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=1)
clf=LogisticRegression()
clf.fit(x_train,y_train)
```
![image](https://github.com/user-attachments/assets/cce7b68b-146a-4e9b-ac81-c9dd3b7efbd3)
```
y_pred=clf.predict(x_test)
clf.score(x_test,y_test)
```
![image](https://github.com/user-attachments/assets/1312a7cb-9db6-4b75-818f-0e3c0f219efe)
```
from sklearn.metrics import  accuracy_score, confusion_matrix
cf=confusion_matrix(y_test, y_pred)
cf
```
![image](https://github.com/user-attachments/assets/44e3f888-6e7c-4145-aab9-c000e0d80dd9)
```
accuracy=accuracy_score(y_test, y_pred)
accuracy
```
![image](https://github.com/user-attachments/assets/b54d6d43-3d77-49de-a47e-ef6f98e23c56)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
