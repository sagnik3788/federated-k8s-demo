import flwr as fl
import tensorflow as tf
import h5py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Load custom dataset (assuming it's in .h5 format)
custom_data_path = "full_dataset_vectors.h5"

# Load the .h5 file
with h5py.File(custom_data_path, 'r') as f:
    # Inspect the keys of the file to see the structure
    print(f.keys())  # This will print the keys in the HDF5 file
    # Assuming the keys for training and testing data are 'X_train', 'y_train', 'X_test', 'y_test'
    X_train = np.array(f['X_train'])
    y_train = np.array(f['y_train'])
    X_test = np.array(f['X_test'])
    y_test = np.array(f['y_test'])

# Preprocess the data (Reshape and Normalize if necessary)
# If using 4096-D vectors, don't reshape into 28x28. Instead, treat it as a dense input.
X_train = X_train.astype("float32") / 255.0  # Normalize the data if it's in pixel range
X_test = X_test.astype("float32") / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define CNN model
def create_model():
    model = Sequential([
        Dense(512, activation='relu', input_shape=(4096,)),  # Using dense layer for 4096-D vector input
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')  # 10 classes for MNIST
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Flower client
class MnistClient(fl.client.NumPyClient):
    def __init__(self, model):
        self.model = model

    def get_parameters(self, config=None):
        """Return current model weights."""
        return self.model.get_weights()

    def set_parameters(self, parameters):
        print("Server received weights: ", [w.shape for w in parameters])
        self.model.set_weights(parameters)
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        history = self.model.fit(X_train, y_train, epochs=1, batch_size=16)
        metrics = {"accuracy": history.history["accuracy"][-1]}  # Use the final accuracy
        return self.get_parameters(config), len(X_train), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.model.evaluate(X_test, y_test)
        return loss, len(X_test), {"accuracy": accuracy}


model = create_model()

# Start the Flower client and connect to the server
fl.client.start_numpy_client(server_address="localhost:8080", client=MnistClient(model))
