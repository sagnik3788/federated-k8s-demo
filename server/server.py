import flwr as fl
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, num_classes=2, test_data_dir="test", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.test_data_dir = test_data_dir
        self.global_model = self.build_model()
        self.global_model.compile(optimizer=Adam(learning_rate=0.0001), 
                                  loss="categorical_crossentropy", 
                                  metrics=["accuracy"])
        self.parameters = None
        self.history = {
            "accuracy": [],
            "loss": [],
            "precision": [],
            "recall": [],
            "f1": [],
        }

    def aggregate_fit(self, server_round, results, failures):
        parameters_aggregated, _ = super().aggregate_fit(server_round, results, failures)
        
        if parameters_aggregated:
            # Convert Flower parameters to model weights
            new_weights = [fl.common.parameter.bytes_to_ndarray(t) for t in parameters_aggregated.tensors]
            self.global_model.set_weights(new_weights)
            self.global_model.compile(optimizer=Adam(learning_rate=0.0001), 
                                      loss="categorical_crossentropy", 
                                      metrics=["accuracy"])
            self.parameters = parameters_aggregated.tensors

            print(f"[Round {server_round}] Aggregated new model weights. Evaluating...")

            # Evaluate after aggregation
            self.evaluate_on_test_data()
            self.plot_metrics()

        return parameters_aggregated, {}

    def build_model(self):
        base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

        for layer in base_model.layers[:300]:
            layer.trainable = False
        for layer in base_model.layers[300:]:
            layer.trainable = True
        
        x = GlobalAveragePooling2D()(base_model.output)
        x = BatchNormalization()(x)
        x = Dense(1024, activation="relu", kernel_regularizer=l2(0.001))(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation="relu", kernel_regularizer=l2(0.001))(x)
        x = Dropout(0.4)(x)
        predictions = Dense(self.num_classes, activation="softmax")(x)

        return Model(inputs=base_model.input, outputs=predictions)

    def evaluate_on_test_data(self):
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            self.test_data_dir, target_size=(224, 224), batch_size=32, 
            class_mode='categorical', shuffle=False
        )

        loss, accuracy = self.global_model.evaluate(test_generator, steps=len(test_generator))

        # Get predictions and true labels
        y_true = test_generator.classes
        y_pred_probs = self.global_model.predict(test_generator, steps=len(test_generator), verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Compute precision, recall, F1
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        print(f"🔍 Test Loss: {loss:.4f} | Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")

        # Update history
        self.history["accuracy"].append(accuracy)
        self.history["loss"].append(loss)
        self.history["precision"].append(precision)
        self.history["recall"].append(recall)
        self.history["f1"].append(f1)

    def plot_metrics(self):
        rounds = list(range(1, len(self.history["accuracy"]) + 1))

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(rounds, self.history["accuracy"], label="Accuracy", color='blue')
        plt.title("Test Accuracy")
        plt.xlabel("Rounds")
        plt.ylabel("Accuracy")
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(rounds, self.history["loss"], label="Loss", color='red')
        plt.title("Test Loss")
        plt.xlabel("Rounds")
        plt.ylabel("Loss")
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.plot(rounds, self.history["precision"], label="Precision", color='green')
        plt.plot(rounds, self.history["recall"], label="Recall", color='orange')
        plt.plot(rounds, self.history["f1"], label="F1 Score", color='purple')
        plt.title("Precision / Recall / F1")
        plt.xlabel("Rounds")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("training_metrics.png")
        print("📊 Updated plot saved as 'training_metrics.png'.")

# Define the strategy
strategy = SaveModelStrategy(
    num_classes=2,
    test_data_dir="test",
    fraction_fit=0.5,
    min_available_clients=2,
    min_evaluate_clients=2,
    min_fit_clients=2,
)

def start_server():
    print("🚀 Starting the Flower server...")

    fl.server.start_server(
        server_address="0.0.0.0:8080", 
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

    if strategy.parameters is not None:
        final_weights = [fl.common.parameter.bytes_to_ndarray(t) for t in strategy.parameters]
        strategy.global_model.compile(optimizer=Adam(learning_rate=0.0001), 
                                      loss="categorical_crossentropy", 
                                      metrics=["accuracy"])
        strategy.global_model.set_weights(final_weights)

        print("🏁 Final Model Evaluation:")
        strategy.evaluate_on_test_data()

        strategy.global_model.save('final_inception_model.keras')
        print("✅ Final model saved as 'final_inception_model.keras'.")
    else:
        print("⚠️ No model parameters available. Training might have failed.")

if __name__ == "__main__":
    start_server()