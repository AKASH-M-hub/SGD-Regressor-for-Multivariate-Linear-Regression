# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load data, select features/targets, and split into train/test sets.

2. Scale features and targets using StandardScaler.

3. Train SGDRegressor with MultiOutputRegressor on training data.

4. Predict, inverse scale, and compute MSE.
 

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Akash M
RegisterNumber:  212224230013

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
dataset = fetch_california_housing()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['HousingPrice'] = dataset.target
print(df.head())
X = df.drop(columns=['AveOccup', 'HousingPrice'])  # Independent variables
X.info()

Y = df[['AveOccup', 'HousingPrice']]  # Dependent variables (Multi-output)
Y.info()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)
sgd = SGDRegressor(max_iter=1000, tol=1e-3)
multi_output_sgd = MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train, Y_train)
Y_pred = multi_output_sgd.predict(X_test)

# Inverse transform to get original scale
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)

# Evaluation
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)
print("\nPredictions:\n", Y_pred[:5])
*/
```

## Output:
![image](https://github.com/user-attachments/assets/91c6ea6c-69f2-4510-9fcc-2381c2204f53)
![image](https://github.com/user-attachments/assets/f50ce276-fcac-453d-972b-ed9f60bed7c9)

![image](https://github.com/user-attachments/assets/46492912-c73c-40e6-a35e-90c4a7d92368)
![image](https://github.com/user-attachments/assets/e9f523d5-f711-455a-bba2-b7f0dac8d191)




## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
