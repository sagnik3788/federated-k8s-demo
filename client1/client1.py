import flwr as fl
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load and preprocess data
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'dataset',  
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Build the model
base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Define Flower client
class MobileNetClient(fl.client.NumPyClient):
    def get_parameters(self, config=None):  
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(train_generator, epochs=1, steps_per_epoch=len(train_generator))
        return model.get_weights(), len(train_generator), {"accuracy": model.history.history["accuracy"][-1]}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(train_generator, steps=len(train_generator))
        return loss, len(train_generator), {"accuracy": accuracy}

fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=MobileNetClient(),
)
