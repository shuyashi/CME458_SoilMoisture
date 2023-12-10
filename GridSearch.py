import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from kerastuner.tuners import GridSearch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from kerastuner import HyperParameters

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
X_val = Xv_normalized[:, 1:]
y_val = Xv_normalized[:, 0]
X_train, X_test = X, X_val
y_train, y_test = y, y_val


# Define the function to build the neural network model
def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units', min_value=32, max_value=128, step=32),
                    activation=hp.Choice('activation', values=['relu', 'tanh', 'sigmoid']), input_shape=(input_shape, 7)))

    # Add additional hidden layers (optional)
    for i in range(hp.Int('num_hidden_layers', 0, 5)):  # Vary the number of hidden layers from 0 to 5
        model.add(Dense(units=hp.Int(f'layer_{i}_units', min_value=32, max_value=128, step=32),
                        activation=hp.Choice(f'layer_{i}_activation', values=['relu', 'tanh', 'sigmoid'])))

    model.add(Dense(1, activation='sigmoid'))

    # Choose the optimizer and learning rate
    optimizer = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    # Choose the alpha (L2 regularization factor)
    alpha = hp.Float('alpha', min_value=0.0, max_value=0.1, step=0.01)

    if alpha > 0.0:
        # Add regularization if alpha > 0
        model.add(tf.keras.regularizers.l2(alpha))

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Assuming you have your training data (X_train, y_train) and validation data (X_val, y_val) ready
input_shape = X_train.shape[1]

# Create the Keras Tuner grid search object
tuner = GridSearch(
    build_model,
    objective='val_accuracy',  # Metric to optimize
    max_trials=10,  # Number of different combinations to try
    directory='grid_search_results',  # Directory to store results
    project_name='my_grid_search'  # Name of the tuning project
)
hp = HyperParameters()
# Perform the grid search with cross-validation
# tuner.search(x=X_train, y=y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
# Perform the grid search with cross-validation
tuner.search(X_train, y_train, epochs=hp.Int('epochs', min_value=10, max_value=50, step=10),
             batch_size=hp.Choice('batch_size', values=[16, 32, 64]), validation_data=(X_val, y_val))

# Get the best hyperparameters
best_hyperparameters = tuner.get_best_hyperparameters()[0]
print("Best Hyperparameters:")
print(best_hyperparameters)

# Build and train the final model with the best hyperparameters
final_model = tuner.hypermodel.build(best_hyperparameters)
# final_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))
final_model.fit(X_train, y_train, epochs=best_hyperparameters.get('epochs'),
                batch_size=best_hyperparameters.get('batch_size'), validation_data=(X_val, y_val))
# def build_model(hp):
#     model = Sequential()
#     model.add(Dense(units=hp.Int('units', min_value=32, max_value=256, step=32), activation='relu', input_shape=(input_shape,)))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(optimizer=hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop']),
#                   loss='binary_crossentropy', metrics=['accuracy'])
#     return model
#
#
# tuner = GridSearch(
#     build_model,
#     objective='val_accuracy',
#     max_trials=10,  # Number of different combinations to try
#     executions_per_trial=2,  # Number of models to train per trial (to reduce noise)
#     directory='grid_search_results',  # Directory to store results
#     project_name='my_grid_search'  # Name of the tuning project
# )
#
# tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
#
# # Get the best hyperparameters
# best_hyperparameters = tuner.get_best_hyperparameters()[0]
# final_model = build_model(best_hyperparameters)
# final_epochs = 50  # Replace with the desired number of epochs for the final training
# final_batch_size = 32  # Replace with the desired batch size for the final training
#
# final_model.fit(X_train, y_train, epochs=final_epochs, batch_size=final_batch_size, validation_data=(X_test, y_test))

predictions = final_model.predict(X_test)
# plot
plt.plot(y_test, label='Actual data')
plt.plot(predictions, label='predictions')
# Set labels and title
plt.xlabel('Time step')
plt.ylabel('Soil Moisture')
plt.title('Actual Vs. prediction')
# Display the legend
plt.legend()
plt.show()
