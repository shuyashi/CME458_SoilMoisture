import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the CSV data for training
data = pd.read_csv('training data.csv')
export_data = data.iloc[:, 2:10]
ext_data = np.array(export_data)

# Load the CSV data for validation
data_v = pd.read_csv('validation data.csv')
valid_data = data_v.iloc[:, 2:10]
va_data = np.array(valid_data)

# Normalize the input data and validation data
scaler = StandardScaler()
X_normalized = scaler.fit_transform(ext_data)
Xv_normalized = scaler.fit_transform(va_data)
X = X_normalized[:, 1:]
y = X_normalized[:, 0]
X_v = Xv_normalized[:, 1:]
y_v = Xv_normalized[:, 0]
X_train, X_test = X, X_v
y_train, y_test = y, y_v


# logistic
# Define the logistic function
def logistic_function(X_train, a, b):
    z = np.dot(X_train, a) + b
    return 1 / (1 + np.exp(-z))


# Fit the logistic function to the data using curve_fit
params, _ = curve_fit(logistic_function, X_train, y_train)

# Extract the fitted parameters
a_fit, b_fit = params[:-1], params[-1]

# Generate points for plotting the fitted curve
X_fit = np.linspace(0, 12, 100)
y_fit = logistic_function(X_fit, a_fit, b_fit)

# Plot the original data and the fitted curve
plt.scatter(X_train, y_train, label='Original Data')
plt.plot(X_fit, y_fit, color='r', label='Fitted Curve')
plt.xlabel('True output')
plt.ylabel('Fitted output')
plt.legend()
plt.show()


# # exponential
# def exponential_function(X_train, a, b):
#     return a * np.exp(bx)
#
#
# # logarithm
# def logarithm_function(X_train, a, b):
#     return a + b * log(X_train)

# # polynomial y = a+b1*x1+b2*x2+b3*x3+b4*x4+b5*x5+b6*x6+b7*x7+c(1 to 7)*(x1^2 to x7^2)
# a_in = 0
# b1_in = 0
# b2_in = 0
# b3_in = 0
# b4_in = 0
# b5_in = 0
# b6_in = 0
# b7_in = 0
# c1_in = 0
# c2_in = 0
# c3_in = 0
# c4_in = 0
# c5_in = 0
# c6_in = 0
# c7_in = 0
# A_in = a_in * X[:, 0]
# B1_in = b1_in * X[:, 1]
# B2_in = b2_in * X[:, 2]
# B3_in = b3_in * X[:, 3]
# B4_in = b4_in * X[:, 4]
# B5_in = b5_in * X[:, 5]
# B6_in = b6_in * X[:, 6]
# B7_in = b7_in * X[:, 7]
# C1_in = c1_in * X[:, 1] ** 2
# c1in = c1_in * 2 * X[:, 1]
# C2_in = c2_in * X[:, 2] ** 2
# c2in = c2_in * 2 * X[:, 2]
# C3_in = c3_in * X[:, 3] ** 2
# c3in = c3_in * 2 * X[:, 3]
# C4_in = c4_in * X[:, 4] ** 2
# c4in = c4_in * 2 * X[:, 4]
# C5_in = c5_in * X[:, 5] ** 2
# c5in = c5_in * 2 * X[:, 5]
# C6_in = c6_in * X[:, 6] ** 2
# c6in = c6_in * 2 * X[:, 6]
# C7_in = c7_in * X[:, 7] ** 2
# c7in = c7_in * 2 * X[:, 7]
# F = A_in + B1_in + B2_in + B3_in + B4_in + B5_in + B6_in + B7_in + C1_in + C2_in + C3_in + C4_in + C5_in + C6_in + C7_in
# y = Y
#
# Ain = B1_in + B2_in + B3_in + B4_in + B5_in + B6_in + B7_in + C1_in + C2_in + C3_in + C4_in + C5_in + C6_in + C7_in
# B1in = A_in + B2_in + B3_in + B4_in + B5_in + B6_in + B7_in + C1_in + C2_in + C3_in + C4_in + C5_in + C6_in + C7_in
# B2in = A_in + B1_in + B3_in + B4_in + B5_in + B6_in + B7_in + C1_in + C2_in + C3_in + C4_in + C5_in + C6_in + C7_in
# B3in = A_in + B1_in + B2_in + B4_in + B5_in + B6_in + B7_in + C1_in + C2_in + C3_in + C4_in + C5_in + C6_in + C7_in
# B4in = A_in + B1_in + B2_in + B3_in + B5_in + B6_in + B7_in + C1_in + C2_in + C3_in + C4_in + C5_in + C6_in + C7_in
# B5in = A_in + B1_in + B2_in + B3_in + B4_in + B6_in + B7_in + C1_in + C2_in + C3_in + C4_in + C5_in + C6_in + C7_in
# B6in = A_in + B1_in + B2_in + B3_in + B4_in + B5_in + B7_in + C1_in + C2_in + C3_in + C4_in + C5_in + C6_in + C7_in
# B7in = A_in + B1_in + B2_in + B3_in + B4_in + B5_in + B6_in + C1_in + C2_in + C3_in + C4_in + C5_in + C6_in + C7_in
# C1in = A_in + B1_in + B2_in + B3_in + B4_in + B5_in + B6_in + B7_in + c1in + C2_in + C3_in + C4_in + C5_in + C6_in + C7_in
# C2in = A_in + B1_in + B2_in + B3_in + B4_in + B5_in + B6_in + B7_in + C1_in + c2in + C3_in + C4_in + C5_in + C6_in + C7_in
# C3in = A_in + B1_in + B2_in + B3_in + B4_in + B5_in + B6_in + B7_in + C1_in + C2_in + c3in + C4_in + C5_in + C6_in + C7_in
# C4in = A_in + B1_in + B2_in + B3_in + B4_in + B5_in + B6_in + B7_in + C1_in + C2_in + C3_in + c4in + C5_in + C6_in + C7_in
# C5in = A_in + B1_in + B2_in + B3_in + B4_in + B5_in + B6_in + B7_in + C1_in + C2_in + C3_in + C4_in + c5in + C6_in + C7_in
# C6in = A_in + B1_in + B2_in + B3_in + B4_in + B5_in + B6_in + B7_in + C1_in + C2_in + C3_in + C4_in + C5_in + c6in + C7_in
# C7in = A_in + B1_in + B2_in + B3_in + B4_in + B5_in + B6_in + B7_in + C1_in + C2_in + C3_in + C4_in + C5_in + C6_in + c7in
# flag = True
# i = 1
# while (flag):
#     Z = Y - F
#     D = np.column_stack((A_in, B1in, B2in, B3in, B4in, B5in, B6in, B7in, C1in, C2in, C3in, C4in, C5in, C6in, C7in))
#     det = np.linalg.inv(D.T.dot(D)).dot(D.T).dot(Z)
#     Ain = Ain + det[0]
#     B1in = B1in + det[1]
#     B2in = B2in + det[2]
#     B3in = B3in + det[3]
#     B4in = B4in + det[4]
#     B5in = B5in + det[5]
#     B6in = B6in + det[6]
#     B7in = B7in + det[7]
#     C1in = C1in + det[8]
#     C2in = C2in + det[9]
#     C3in = C3in + det[10]
#     C4in = C4in + det[11]
#     C5in = C5in + det[12]
#     C6in = C6in + det[13]
#     C7in = C7in + det[14]
#     if np.linalg.norm(det) > 0.001:
#         i = i + 1
#         flag = True
#     else:
#         flag = False
#
# print('F = ', F)
# print('D = ', D)
# print('Z = ', Z)
# print('det = ', det)
# print('w = ', w)
# print('h = ', h)
# print('i = ', i)
#
#
# def nonlinear_regression2(x1, x2, x3, x4, x5, x6, x7):
#     return
#
#
# def nonlinear_regression3(x1, x2, x3, x4, x5, x6, x7):
#     return
