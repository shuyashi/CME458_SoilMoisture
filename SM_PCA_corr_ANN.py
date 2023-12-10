# The second method to train: choose the PCs based on the correlation with the output
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
pca = PCA()
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Calculate the correlation matrix
correlation_matrix = np.corrcoef(X_train_pca.T, y_train)

# Extract the correlation values between features and the output variable
correlation_with_output = correlation_matrix[:-1, -1]  # Exclude the last row (correlation of output with itself)

# Sort the correlation values in descending order and get the corresponding feature indices
sorted_indices = np.argsort(np.abs(correlation_with_output))[::-1]

# Calculate the correlation between each PC and the output variable
correlations = []
for i in range(X_train_pca.shape[1]):
    corr = np.corrcoef(X_train_pca[:, i], y_train)[0, 1]
    correlations.append(corr)

# Calculate the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Determine the number of components to keep based on the elbow plot
num_components = np.argmax(np.cumsum(explained_variance_ratio) >= 0.95) + 1

# Sort the correlations and select the PCs with the highest absolute correlation values
correlations = np.abs(correlations)
selected_pcs = np.argsort(correlations)[::-1][:num_components]
print(selected_pcs)

# Plot the elbow plot
plt.plot(range(1, len(explained_variance_ratio) + 1), np.cumsum(explained_variance_ratio), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Elbow Plot')
plt.show()

# Extract the selected PCs from the transformed data
X_selected = X_train_pca[:, selected_pcs]
X_selected_test = X_test_pca[:, selected_pcs]

# Apply PCA with the selected number of components
pca = PCA(n_components=num_components)
X_train_pca = pca.fit_transform(X_selected)
X_test_pca = pca.transform(X_selected_test)

# similar steps as the general PCA file
# Convert the reduced PC set back to the original set and check the accuracy
X_train_reconstructed = np.dot(X_train_pca, pca.components_) + pca.mean_
X_test_reconstructed = np.dot(X_test_pca, pca.components_) + pca.mean_


# plot for the reduced pc set and the original set
plt.figure(2)
plt.plot(X_selected[:, 1], label='Actual data')
plt.plot(X_train_reconstructed[:, 1], label='PCA reconstruction')
# Set labels and title
plt.xlabel('Time step')
plt.title('Actual Vs. pca')
# Display the legend
plt.legend()
# Display the plot
plt.show()

# Train the neural network
# Define the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(85, activation='tanh', input_shape=(4,), use_bias=True),
    tf.keras.layers.Dense(65, activation='sigmoid', use_bias=True),
    tf.keras.layers.Dense(45, activation='sigmoid', use_bias=True),
    tf.keras.layers.Dense(30, activation='sigmoid', use_bias=True),
    tf.keras.layers.Dense(1, activation='sigmoid', use_bias=True)
])

# Compile the model
model.compile(optimizer='adamax', loss='mean_absolute_error')

# Train the model
model.fit(X_selected, y_train, epochs=2000, batch_size=100, verbose=1)

# Predict on the test set
y_pred = model.predict(X_selected_test)

# # Evaluate the accuracy of the model
# accuracy = accuracy_score(X_train, X_train_reconstructed)
# print("Training set accuracy: ", accuracy)
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