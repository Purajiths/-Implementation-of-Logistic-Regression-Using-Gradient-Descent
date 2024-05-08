# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages required.
2. Read the dataset.
3. Define X and Y array.
4. Define a function for sigmoid, loss, gradient and predict and perform operations. 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Purajith S
RegisterNumber: 212223040158
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("Placement_Data.csv")
dataset
dataset=dataset.drop("sl_no",axis=1)
dataset=dataset.drop("salary",axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes



dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes

dataset

X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values

Y

theta=np.random.randn(X.shape[1])
y=Y

def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1-y) * np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha*gradient
    return theta
    
theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)

def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred
    
y_pred = predict(theta,X)

accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:", accuracy)

print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```

## Output:
### Read the file and display
![WhatsApp Image 2024-05-09 at 01 00 12_f6cebc5a](https://github.com/Purajiths/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145548193/9f1e1ada-1de7-493f-863e-33b9e207f8eb)


### Categorizing columns
![WhatsApp Image 2024-05-09 at 01 00 21_27760166](https://github.com/Purajiths/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145548193/75a3cdd8-0267-411e-88be-c18d9b7a47bf)


### Labelling columns and displaying dataset
![WhatsApp Image 2024-05-09 at 01 00 27_efca6f69](https://github.com/Purajiths/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145548193/495316f1-83c8-49e3-adc6-9e2cdb767e26)


### Display dependent variable
![WhatsApp Image 2024-05-09 at 01 00 30_fb0bb823](https://github.com/Purajiths/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145548193/caf11785-082a-4cb7-be39-4be85e23a46f)

### Printing accuracy
![WhatsApp Image 2024-05-09 at 01 00 35_4dcc5403](https://github.com/Purajiths/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145548193/2c7f37f0-ad96-41b3-848a-5d22460fcffa)


### Printing Y
![WhatsApp Image 2024-05-09 at 01 00 38_2e9c5065](https://github.com/Purajiths/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145548193/0bc9fed1-93d0-4d5a-8a3d-32dee08ce22c)



### Printing y_prednew
![WhatsApp Image 2024-05-09 at 01 00 43_0fa26566](https://github.com/Purajiths/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145548193/65cd7039-4683-45b1-8503-bb6e9ea5b436)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
