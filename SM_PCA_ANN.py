import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

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
X_train = scaler.fit_transform(ext_data[:, 1:])
y_train = ext_data[:, 0]
X_test = scaler.transform(va_data[:, 1:])
y_test = va_data[:, 0]

# # Convert labels to discrete classes
# threshold = 0.5
# y_train_class = np.where(y_train >= threshold, 1, 0)
# y_test_class = np.where(y_test >= threshold, 1, 0)

# Perform PCA for dimensionality reduction
pca = PCA()  # Do not specify the number of components initially
X_train_pca = pca.fit_transform(X_train)

# Determine the number of components to keep based on explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
num_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1  # Keep components explaining 95% variance

# Access eigenvalues
eigenvalues = pca.explained_variance_
# Sort eigenvalues in descending order
sorted_eigenvalues = np.sort(eigenvalues)[::-1]
# Print eigenvalues
print(sorted_eigenvalues)

# Plot the elbow plot, choose the largest few PCs to train the model which has the highest valiability
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Elbow Plot for PCA')
plt.show()

# Apply PCA with the selected number of components
pca = PCA(n_components=num_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Convert the reduced PC set back to the original set and check the accuracy
X_train_reconstructed = np.dot(X_train_pca, pca.components_) + pca.mean_
X_test_reconstructed = np.dot(X_test_pca, pca.components_) + pca.mean_

# Checking accuracy: accuracy_score function from scikit-learn does not support multi-output or continuous targets.
# It is typically used for classification tasks with discrete labels.
# If you have continuous targets or multi-output regression, you need to use a different evaluation metric.
mse_train = mean_squared_error(X_train, X_train_reconstructed)
mse_test = mean_squared_error(X_test, X_test_reconstructed)
print("Training set MSE: ", mse_train)
print("Test set MSE: ", mse_test)

# plot for the reduced pc set and the original set
plt.figure(2)
plt.plot(X_train[:, 1], label='Actual data')
plt.plot(X_train_reconstructed[:, 1], label='PCA reconstruction')
# Set labels and title
plt.xlabel('Time step')
plt.title('Actual Vs. pca')
# Display the legend
plt.legend()
# Display the plot
plt.show()

# Train the neural network：train NN with MLP, the neural network has two hidden layers with 100 and 50 neurons,
# respectively. The relu activation function is used, and the adam solver is used for optimization. The max_iter
# parameter specifies the maximum number of iterations (epochs) for training.
# activation function: relu, tanh, logistic, identity; solver: lbfgs, sgd, adam
# model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500)
# # Fit and train the MLP classifier
# model.fit(X_train_pca, y_train)

# Define the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(70, activation='sigmoid', input_shape=(4,), use_bias=True),
    tf.keras.layers.Dense(60, activation='sigmoid', use_bias=True),
    tf.keras.layers.Dense(40, activation='sigmoid', use_bias=True),
    tf.keras.layers.Dense(30, activation='sigmoid', use_bias=True),
    tf.keras.layers.Dense(1, activation='linear', use_bias=True)
])

# Compile the model
model.compile(optimizer='Adamax', loss='mean_absolute_error')

# Train the model
model.fit(X_train_pca, y_train, epochs=4000, batch_size=50, verbose=1)

# Predict on the test set
y_pred = model.predict(X_test_pca)

# Evaluate the accuracy of the model
mse_pred = mean_squared_error(y_test, y_pred)
print("Test set MSE: ", mse_pred)

# Unscale the PCA set
predictions = y_pred * np.std(ext_data[:, 1:]) + np.mean(ext_data[:, 1:])

# plot
plt.plot(y_test, label='Actual data')
plt.plot(y_pred, label='predictions')
# Set labels and title
plt.xlabel('Time step')
plt.ylabel('Soil Moisture')
plt.title('Actual Vs. prediction')
# Display the legend
plt.legend()
# Display the plot
plt.show()

# Extract the explicit model equation
model_weights = []
for layer in model.layers:
    layer_weights = layer.get_weights()
    model_weights.append(layer_weights)


# Define the activation functions
def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# Define the model prediction function
def predict(x):
    layer1_weights, layer2_weights, layer3_weights, layer4_weights, layer5_weights = model_weights[0], model_weights[1], model_weights[2], model_weights[3], model_weights[4]
    layer1_output = sigmoid(np.dot(x, layer1_weights[0]) + layer1_weights[1])
    layer2_output = sigmoid(np.dot(layer1_output, layer2_weights[0]) + layer2_weights[1])
    layer3_output = sigmoid(np.dot(layer2_output, layer3_weights[0]) + layer3_weights[1])
    layer4_output = sigmoid(np.dot(layer3_output, layer4_weights[0]) + layer4_weights[1])
    layer5_output = sigmoid(np.dot(layer4_output, layer5_weights[0]) + layer5_weights[1])
    return layer5_output

    # Perform model prediction on X_test_pca
    predictions = [predict(x) for x in X_test_pca]

    return predictions


# Get the model predictions for X_test
model_prediction = predict(X_test_pca)
model_predictions = model_prediction * np.std(ext_data[:, 1:]) + np.mean(ext_data[:, 1:])
plt.plot(y_pred, ":r", label='inbuilt')
plt.plot(model_predictions, "b.", label='explicit_model')
# Set labels and title
plt.xlabel('Time step')
plt.ylabel('Soil Moisture')
plt.title('Inbuilt_prediction Vs. Explicit_model_predictions')
# Display the legend
plt.legend()
# Display the plot
plt.show()