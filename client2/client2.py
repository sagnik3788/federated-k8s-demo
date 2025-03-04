import flwr as fl
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load and preprocess data
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'dataset',  # Replace with the path to your training data
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

# Build the model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)  # Adjust for binary classification
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Define Flower client
class EfficientNetClient(fl.client.NumPyClient):
    def get_parameters(self, config=None):  # Add config parameter
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(train_generator, epochs=1, steps_per_epoch=len(train_generator))
        return model.get_weights(), len(train_generator), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(train_generator, steps=len(train_generator))
        return loss, len(train_generator), {"accuracy": accuracy}

# Start Flower client using the new start_client method
fl.client.start_client(
    server_address="localhost:8080",
    client=EfficientNetClient().to_client(),  # Convert NumPyClient to Client
)