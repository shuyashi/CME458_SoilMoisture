import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt

# Load the CSV data for training
data = pd.read_csv('training data.csv')
export_data = data.iloc[:, 2:10]
# matrix = np.array(data)
# Split the data into input and output
ext_data = np.array(export_data)
# y = data.iloc[:, 7].values

# Load the CSV data for validation
data_v = pd.read_csv('validation data.csv')
valid_data = data_v.iloc[:, 2:10]
va_data = np.array(valid_data)

# Normalize the input data and validation data
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(ext_data)
Xv_normalized = scaler.fit_transform(va_data)
X = X_normalized[:, 1:]
y = X_normalized[:, 0]
X_v = Xv_normalized[:, 1:]
y_v = Xv_normalized[:, 0]
X_train, X_test = X, X_v
y_train, y_test = y, y_v

# Create the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(110, activation='relu', input_shape=(7,), bias_initializer='zeros'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=6000, batch_size=600, verbose=1)
# using a larger batch size (e.g., 32, 64, or 128) is called mini-batch gradient descent. It provides a balance
# between the efficiency of larger batch sizes and the noise associated with small batch sizes. With larger batches,
# the gradient estimates become more stable and the updates to the model's weights are less frequent but more
# accurate.

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
predictions = model.predict(X_test)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)


def calculate_rmse(actual, predicted):
    """
    Calculates the Root Mean Square Error (RMSE) between actual and predicted data.

    Args:
        actual (numpy.ndarray): Array of actual values.
        predicted (numpy.ndarray): Array of predicted values.

    Returns:
        float: The RMSE value.
    """
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    return rmse


print("rmse:", calculate_rmse(X_test, predictions))

# unscale the data
predictions = predictions * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
Y_test = y_test * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]

# plot
plt.plot(Y_test, label='Actual data')
plt.plot(predictions, label='predictions')
# Set labels and title
plt.xlabel('Time step')
plt.ylabel('Soil Moisture')
plt.title('Actual Vs. prediction')
# Display the legend
plt.legend()
# Display the plot
plt.show()

# Neural Network Model
# Extract the model weights
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
    layer1_weights, layer2_weights, layer3_weights = model_weights[0], model_weights[1], model_weights[2]
    layer1_output = relu(np.dot(x, layer1_weights[0]) + layer1_weights[1])
    layer2_output = relu(np.dot(layer1_output, layer2_weights[0]) + layer2_weights[1])
    layer3_output = sigmoid(np.dot(layer2_output, layer3_weights[0]) + layer3_weights[1])
    return layer3_output

    # Perform model prediction on X_test
    predictions = [predict(x) for x in X_test]

    return predictions


# Get the model predictions for X_test
model_prediction = predict(X_test)
model_predictions = model_prediction * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
plt.plot(predictions, ":r", label='inbuilt')
plt.plot(model_predictions, "b.", label='explicit_model')
# Set labels and title
plt.xlabel('Time step')
plt.ylabel('Soil Moisture')
plt.title('Inbuilt_prediction Vs. Explicit_model_predictions')
# Display the legend
plt.legend()
# Display the plot
plt.show()

# print the model summary, the model function is y = w_i * x + b_i
print(model.summary())

# Extract the model configuration
model_config = model.get_config()
# Create a new model from the extracted configuration
new_model = tf.keras.models.Sequential.from_config(model_config)