import flwr as fl
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, MobileNetV3Small
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, num_classes=2, test_data_dir=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes  # Number of output classes
        self.parameters = None  # To store aggregated parameters
        self.test_data_dir = test_data_dir  # Path to the test data folder
        self.model_type = None  # To track the model type (efficientnet or mobilenet)

    def aggregate_fit(self, server_round, results, failures):
        # Aggregate the model parameters using FedAvg's default aggregation
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        
        # Store the aggregated model parameters for later use
        if parameters_aggregated is not None:
            self.parameters = parameters_aggregated.tensors
        return parameters_aggregated, metrics_aggregated

    def build_model(self, model_type):
        """Build the model based on the specified architecture."""
        if model_type == "efficientnet":
            base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
        elif model_type == "mobilenet":
            base_model = MobileNetV3Small(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        base_model.trainable = False  # Freeze the base model
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(self.num_classes, activation="softmax"),  # Adjust for number of classes
        ])
        return model

    def evaluate_on_test_data(self, model):
        """Evaluate the model on the test dataset."""
        if self.test_data_dir is None:
            print("No test data directory provided. Skipping evaluation.")
            return
        
        # Load and preprocess the test data
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            self.test_data_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=False  # Ensure consistent evaluation
        )
        
        # Evaluate the model
        loss, accuracy = model.evaluate(test_generator, steps=len(test_generator))
        print(f"Test Loss: {loss}")
        print(f"Test Accuracy: {accuracy}")

# Define the server strategy with custom metrics aggregation
strategy = SaveModelStrategy(
    num_classes=2,  # Number of output classes (Pneumonia/Normal)
    test_data_dir="path_to_test_data",  # Path to the test data folder
    fit_metrics_aggregation_fn=lambda metrics: {
        "accuracy": np.mean([m.get("accuracy", 0.0) for _, m in metrics])
    },
    evaluate_metrics_aggregation_fn=lambda metrics: {
        "accuracy": np.mean([m.get("accuracy", 0.0) for _, m in metrics])
    },
)

# Define the function to start the server
def start_server():
    print("Starting the Flower server...")
    # Start the server with the defined strategy
    history = fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=1),  # Adjust num_rounds as needed
        strategy=strategy,
    )
    
    # After training, save the aggregated model and evaluate on test data
    if hasattr(strategy, 'parameters') and strategy.parameters is not None:
        # Convert aggregated parameters into model weights
        final_weights = [fl.common.parameter.bytes_to_ndarray(tensor) for tensor in strategy.parameters]

        # Build the model (should match client model architecture)
        model = strategy.build_model(strategy.model_type)
        
        # Compile the model (use the same optimizer and loss function as in clients)
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        
        # Set the final aggregated weights into the model
        model.set_weights(final_weights)
        
        # Evaluate the model on the test dataset
        strategy.evaluate_on_test_data(model)
        
        # Save the model to a file
        model.save("final_model.h5")
        print("Model saved successfully.")
    else:
        print("Training completed but no parameters were captured.")

# Start the server
if __name__ == "__main__":
    start_server()