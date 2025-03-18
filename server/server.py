import flwr as fl
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, num_classes=2, test_data_dir="test", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.test_data_dir = test_data_dir
        self.global_model = self.build_model()
        self.parameters = None

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate model updates."""
        if not results:
            return None, {}

        parameters_aggregated, _ = super().aggregate_fit(server_round, results, failures)
        self.parameters = parameters_aggregated.tensors if parameters_aggregated else None
        print("Aggregated MobileNet parameters.")
        return parameters_aggregated, {}

    def build_model(self):
        """Build MobileNetV3 model (Same as clients)."""
        base_model = MobileNetV3Small(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False  # Freeze base layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation="relu")(x)  # Match the client
        predictions = Dense(self.num_classes, activation="softmax")(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        return model


    def evaluate_on_test_data(self, model):
        """Evaluate the model on the test dataset."""
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            self.test_data_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )

        loss, accuracy = model.evaluate(test_generator, steps=len(test_generator))
        print(f"Test Loss: {loss}")
        print(f"Test Accuracy: {accuracy}")

# Define the server strategy
# strategy = SaveModelStrategy(
#     num_classes=2,
#     test_data_dir="test",
#     fit_metrics_aggregation_fn=lambda metrics: {
#         "accuracy": np.mean([m.get("accuracy", 0.0) for _, m in metrics])
#     },
#     evaluate_metrics_aggregation_fn=lambda metrics: {
#         "accuracy": np.mean([m.get("accuracy", 0.0) for _, m in metrics])
#     },
# )
strategy = SaveModelStrategy(
    num_classes=2,
    test_data_dir="test",
    min_available_clients=3,  # Ensure at least 3 clients must be available
    fit_metrics_aggregation_fn=lambda metrics: {
        "accuracy": np.mean([m.get("accuracy", 0.0) for _, m in metrics])
    },
    evaluate_metrics_aggregation_fn=lambda metrics: {
        "accuracy": np.mean([m.get("accuracy", 0.0) for _, m in metrics])
    },
)


def start_server():
    print("Starting the Flower server...")
    
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy,
    )

    # Save model and evaluate
    if strategy.parameters is not None:
        final_weights = [fl.common.parameter.bytes_to_ndarray(tensor) for tensor in strategy.parameters]
        model = strategy.global_model
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        model.set_weights(final_weights)
        strategy.evaluate_on_test_data(model)
        # model.save("final_model_mobilenet.h5")
        model.save('my_model.keras')
        print("Model saved successfully.")
    else:
        print("No parameters captured.")

if __name__ == "__main__":
    start_server()
