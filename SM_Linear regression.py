import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import math
import scipy.stats as stats
# import data
import numpy as np
import pandas as pd

# import data
data = pd.read_csv('training data.csv')
export_data = data.iloc[:, -7:]
export_data = export_data.values  # Convert DataFrame to numpy array
Y = data.iloc[:, 2].values  # Assuming the target variable is in the third column

# build matrix for the input data
x = export_data
x_1 = np.ones(x.shape[0])
x_1 = np.array(x_1)
x = np.array(x)
matrix = np.concatenate((x_1,x[:,0],x[:,1],x[:,2],x[:,3],x[:,4],x[:,5],x[:,6]),axis=0)
X = matrix.reshape(-688,688)
X = X.T
a = X.T.dot(X)
b = X.T.dot(Y)

coef = np.linalg.inv(a).dot(b)
print('The estimated parameter/coefficient is', coef)

def linear_regression(x1,x2,x3,x4,x5,x6,x7):
    return coef[0]+coef[1]*x1+coef[2]*x2+coef[3]*x3+coef[4]*x4+coef[5]*x5+coef[6]*x6+coef[7]*x7
x1 = np.linspace(57.7279,151.114,688)
x2 = np.linspace(84.5929,210.713,688)
x3 = np.linspace(16.2434,25.8594,688)
x4 = np.linspace(0.027339,0.308641,688)
x5 = np.linspace(-3.37401,1.95683,688)
x6 = np.linspace(-1.39577,2.85951,688)
x7 = np.linspace(0.008786,0.984069,688)
an = linear_regression(x1,x2,x3,x4,x5,x6,x7)

plt.plot(x1,an,label = 'linear regression model, x_1')
plt.scatter(X[:,3],Y,color = 'green')
plt.show()

plt.figure(2)
plt.plot(x2,an,label = 'linear regression model, x_2')
plt.scatter(X[:,3],Y,color = 'green')
plt.show()

plt.figure(3)
plt.plot(x3,an,label = 'linear regression model, x_3')
plt.scatter(X[:,3],Y,color = 'green')
plt.show()

plt.figure(4)
plt.plot(x4,an,label = 'linear regression model, x_4')
plt.scatter(X[:,3],Y,color = 'green')
plt.show()

plt.figure(5)
plt.plot(x5,an,label = 'linear regression model, x_5')
plt.scatter(X[:,3],Y,color = 'green')
plt.show()

plt.figure(6)
plt.plot(x6,an,label = 'linear regression model, x_6')
plt.scatter(X[:,3],Y,color = 'green')
plt.show()

plt.figure(7)
plt.plot(x7,an,label = 'linear regression model, x_7')
plt.scatter(X[:,3],Y,color = 'green')
plt.show()