# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values. 
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.


## Program:
```c
## Developed by: harithashree.v
## RegisterNumber: 212222230046

import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])


```
## Output:
## Placement Data:

![Screenshot 2024-04-01 091040](https://github.com/haritha-venkat/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121285701/4237ecb8-f94a-47f7-9257-6a696bc1a2b7)

## Salary Data:

![Screenshot 2024-04-01 091125](https://github.com/haritha-venkat/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121285701/bef8bbf0-52bb-433c-bcff-20327941ac29)

## Checking the null() function:


![Screenshot 2024-04-01 091222](https://github.com/haritha-venkat/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121285701/d3763256-82a1-46d4-a90c-89d98cb69bd9)

## Data Duplicate:


![Screenshot 2024-04-01 091254](https://github.com/haritha-venkat/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121285701/5dd299b5-b6a2-4358-99e1-1efd7f2237ef)

## Print Data:

![Screenshot 2024-04-01 091442](https://github.com/haritha-venkat/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121285701/b2d22fd9-b2ed-4d8c-8630-5931f68d951f)


## Data-Status:

![Screenshot 2024-04-01 091531](https://github.com/haritha-venkat/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121285701/ceb1f67c-91a2-4b6a-8363-abf6f1e4fa19)

## Y_prediction array:
![Screenshot 2024-04-01 091554](https://github.com/haritha-venkat/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121285701/5fc4bd4f-2717-4158-a03e-ad06fb39cdea)

## Accuracy value:
![Screenshot 2024-04-01 091759](https://github.com/haritha-venkat/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121285701/92e7e2c5-ffb7-449d-a98a-bafbfaefe9a1)

## Confusion array:

![image](https://github.com/haritha-venkat/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121285701/3a82b67f-ec0d-459d-b7c7-1e85b41fba27)

## Classification Report:


![Screenshot 2024-04-01 091840](https://github.com/haritha-venkat/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121285701/9ea30f3d-b087-4300-bb06-2a3e8911c882)


## Prediction of LR:
![Screenshot 2024-04-01 091859](https://github.com/haritha-venkat/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121285701/d103166a-2bfb-42cd-a461-71a6bf2c6c07)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
